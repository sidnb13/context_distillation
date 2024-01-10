from micro_config import MetaConfig
from base_configs import AdamWConfig, project_root
from core import TKInferenceConfig, TKServerInference, TKTrainConfig
from nat_inst.ni_formatter import get_formatted_ni_data
from t5_config import T5ModelConfig
from task_assoc import generate_task_association_data, get_binary_tasks, task_assoc_evaluate
from injection_functions import tk_generate_questions, generate_distillation_data, distill
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

    distill_path = '../../data/task_assoc1_def_2pos_2neg_exp/distill_data.pkl'
    out_path = '../../outputs/task_assoc1_def_2pos_2neg_exp/'
    rng_key = jax.random.PRNGKey(0)

    with open(distill_path, 'rb') as f:
        distill_data = pkl.load(f)

    # inference = TKServerInference('http://34.71.136.86:8000/')
    # mesh = contextlib.nullcontext

    metaconfig = MetaConfig(
        project_root=project_root, 
        verbose=False, 
    )
    
    trainer_config = TKTrainConfig(
        model=T5ModelConfig(
            # model_str="google/t5-v1_1-xl", 
            # model_str="t5-3b", 
            # model_str="google/ul2", 
            model_str="google/t5-xxl-lm-adapt", 
            # model_str="allenai/tk-instruct-11b-def-pos-neg-expl", 
            checkpoint_path='outputs/tk_model_full/model/', 
            from_pretrained=True, 
            use_fp16=True, 
            gradient_checkpoint=True, 
        ), 
        optim=AdamWConfig(
            grad_accum_steps=2, 
            lr=1e-5, 
            weight_decay=0.00, 
            beta1=0.9, 
            beta2=0.999, 
            eps=1e-6, 
        ), 
        pjit=True, 
        verbose=True, 
    )

    trainer, _, model, mesh = trainer_config.unroll(metaconfig)

    all_distill_data = sum(distill_data.values(), [])

    rng_key, new_rng = jax.random.split(rng_key)
    trainer = distill(all_distill_data, trainer, mesh, bsize=8, epochs=1, max_input_length=256, rng_key=new_rng)

    model.save_pretrained(
        out_path, 
        params=jax.device_get(trainer.params), 
    )
