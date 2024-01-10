import contextlib
from micro_config import MetaConfig
from base_configs import project_root
from t5_config import T5ModelConfig
from core import TKServerInference, TKInferenceConfig
from nat_inst.ni_formatter import get_formatted_ni_data
from task_assoc import generate_task_association_data, get_binary_tasks
import pickle as pkl
import jax
import json
from tk_inject import generate_tk_instance, tk_evaluate
from utils.randomness import RandomState, seed_context

if __name__ == "__main__":
    
    random_state = RandomState(0)
    rng_key = jax.random.PRNGKey(0)

    
    injection_data = {}

    with seed_context(random_state):
        formatted_train_data, formatted_test_data = get_formatted_ni_data(
            add_task_name=False, add_task_definition=True, num_pos_examples=0, 
            num_neg_examples=0, add_explanation=False, max_num_instances_per_task=100, 
            max_num_instances_per_eval_task=100, 
        )

        for task, data in formatted_test_data.items():
            injection_data[task] = generate_tk_instance(data)
    
    metaconfig = MetaConfig(
        project_root=project_root, 
        verbose=False, 
    )
    
    inference_config = TKInferenceConfig(
        model=T5ModelConfig(
            # model_str="google/t5-v1_1-xl", 
            # model_str="t5-3b", 
            # model_str="google/ul2", 
            model_str="google/t5-xxl-lm-adapt", 
            # model_str="allenai/tk-instruct-11b-def-pos-neg-expl", 
            checkpoint_path='outputs/T5_11B_random_nat_inst_finetune_test2/model/', 
            from_pretrained=True, 
            use_fp16=True, 
            gradient_checkpoint=False, 
        ), 
        pjit=True, 
        verbose=True, 
    )

    inference, _, mesh = inference_config.unroll(metaconfig)

    full_results = {}
    
    for task, data in injection_data.items():
        print('task:', task)
        rng_key, new_rng = jax.random.split(rng_key)
        acc, all_results = tk_evaluate(data, teacher_eval=True, 
                                       inference=inference, mesh=mesh, bsize=8, num_instances=None, 
                                       max_input_length=1024, rng_key=new_rng, 
                                       do_sample=False, num_beams=1, max_length=128)
        print('accuracy:', acc)
        full_results[task] = {'acc': acc, 'all_results': all_results}

    with open(metaconfig.convert_path('outputs/tk_just_definition_eval.pkl'), 'wb') as f:
        pkl.dump(full_results, f)
