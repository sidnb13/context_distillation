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
from tk_inject import tk_evaluate
from utils.randomness import RandomState, seed_context

if __name__ == "__main__":
    
    random_state = RandomState(0)
    rng_key = jax.random.PRNGKey(0)

    with seed_context(random_state):
        formatted_train_data, formatted_test_data = get_formatted_ni_data(
            add_task_name=False, add_task_definition=True, num_pos_examples=2, 
            num_neg_examples=2, add_explanation=True, max_num_instances_per_task=100, 
            max_num_instances_per_eval_task=100, 
        )

        # binary_train_tasks = get_binary_tasks(formatted_train_data)
        # binary_test_tasks = get_binary_tasks(formatted_test_data)

        # injection_data = generate_task_association_data(binary_test_tasks, formatted_test_data)

        tasks = ['task1393_superglue_copa_text_completion', 
                 'task738_perspectrum_classification', 
                 'task242_tweetqa_classification', 
                 'task220_rocstories_title_classification'
                ]
        injection_data = generate_task_association_data(tasks, formatted_test_data)
        
        # permutation = [3, 2, 1, 0]
        # student_prompts = {k: injection_data[k].student_prompt for k in injection_data.keys()}
        # for i, idx in enumerate(permutation):
        #     injection_data[tasks[i]].student_prompt = student_prompts[tasks[idx]]
    
    # injection_data = '../../data/task_assoc1_def_2pos_2neg_exp/injection_data.pkl'
    
    # with open(injection_data, 'rb') as f:
    #     injection_data = pkl.load(f)

    # inference = TKServerInference('http://34.71.136.86:8000/')
    # mesh = contextlib.nullcontext

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
            # checkpoint_path='outputs/task_assoc_test2/model/', 
            checkpoint_path='outputs/T5_11B_random_nat_inst_finetune_test2/model/', 
            from_pretrained=True, 
            use_fp16=True, 
            gradient_checkpoint=False, 
        ), 
        pjit=True, 
        verbose=True, 
    )

    inference, _, mesh = inference_config.unroll(metaconfig)    
    
    for task in injection_data.keys():
        print('task:', task)
        rng_key, new_rng = jax.random.split(rng_key)
        acc, all_results = tk_evaluate(injection_data[task], teacher_eval=False, 
                                       inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                       max_input_length=1024, rng_key=new_rng, 
                                       do_sample=False, num_beams=1, max_length=128)
        print('accuracy:', acc)
