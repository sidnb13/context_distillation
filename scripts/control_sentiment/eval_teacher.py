import contextlib
from micro_config import MetaConfig
from base_configs import project_root
from t5_config import T5ModelConfig
from core import TKServerInference, TKInferenceConfig
from control import control_evaluate, generate_control_instance
from task_assoc import generate_task_association_data, get_binary_tasks
import pickle as pkl
import jax
import json
from tk_inject import generate_tk_instance, tk_evaluate
from utils.randomness import RandomState, seed_context
from datasets import load_dataset

if __name__ == "__main__":
    
    random_state = RandomState(0)
    rng_key = jax.random.PRNGKey(0)

    dataset = load_dataset("imdb")

    injection_data = generate_control_instance(
                                            train_datapoints=dataset['train'], 
                                            eval_datapoints=dataset['test'], 
                                            seed=0, 
                                            only_negative=True, 
                                            positive_control=True, 
                                            )
    
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

    rng_key, new_rng = jax.random.split(rng_key)
    acc, all_results = control_evaluate(injection_data, teacher_eval=True, 
                                        inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                        max_input_length=1024, max_output_length=128, rng_key=new_rng, 
                                        do_sample=False, num_beams=1, max_length=128)
    print('accuracy:', acc)
    breakpoint()
    pass
