from micro_config import MetaConfig
from base_configs import project_root
from core import TKInferenceConfig, TKServerInference
from nat_inst.ni_formatter import get_formatted_ni_data
from t5_config import T5ModelConfig
from task_assoc import generate_task_association_data, get_binary_tasks, task_assoc_evaluate
from injection_functions import tk_generate_questions, generate_distillation_data
import pickle as pkl
import jax
import json
import os
import jax.numpy as jnp
import numpy as np

if __name__ == "__main__":
    
    # formatted_train_data, formatted_test_data = get_formatted_ni_data(
    #     add_task_name=False, add_task_definition=True, num_pos_examples=2, 
    #     num_neg_examples=2, add_explanation=True, max_num_instances_per_task=100, 
    #     max_num_instances_per_eval_task=100, 
    # )

    # binary_train_tasks = get_binary_tasks(formatted_train_data)
    # binary_test_tasks = get_binary_tasks(formatted_test_data)
    
    # injection_data = generate_task_association_data(binary_test_tasks, formatted_test_data)

    injection_data = '../../data/task_assoc1_def_2pos_2neg_exp/injection_data.pkl'
    question_data = '../../data/task_assoc1_def_2pos_2neg_exp/questions.pkl'
    output_path = '../../data/task_assoc1_def_2pos_2neg_exp/distill_data.pkl'
    rng_key = jax.random.PRNGKey(0)

    with open(injection_data, 'rb') as f:
        injection_data = pkl.load(f)
    
    with open(question_data, 'rb') as f:
        questions = pkl.load(f)

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
            checkpoint_path='outputs/tk_model_full/model/', 
            from_pretrained=True, 
            use_fp16=True, 
            gradient_checkpoint=False, 
        ), 
        pjit=True, 
        verbose=True, 
    )

    inference, _, mesh = inference_config.unroll(metaconfig)

    all_data = {}
    for task in injection_data.keys():
        print('task:', task)
        rng_key, new_rng = jax.random.split(rng_key)
        distill_data = generate_distillation_data(
            injection_data[task], questions[task], inference=inference, mesh=mesh, 
            bsize=1, n_per_question=1, max_input_length=1024, max_output_length=128, 
            rng_key=new_rng, do_sample=True, num_beams=1, 
        )
        for item in distill_data:
            item['logits'] = np.asarray(jax.device_get(item['logits']).astype(jnp.float32))
            item['out_tokens'] = np.asarray(jax.device_get(item['out_tokens']))
        all_data[task] = distill_data
    
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, 'wb') as f:
        pkl.dump(all_data, f)
