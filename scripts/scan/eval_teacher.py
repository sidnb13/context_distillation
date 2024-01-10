import random
from micro_config import MetaConfig
from base_configs import project_root, AdamWConfig
from t5_config import T5ModelConfig
from core import TKInferenceConfig
import contextlib
from core import TKServerInference, TKTrainConfig
from nat_inst.ni_formatter import get_formatted_ni_data
from task_assoc import generate_task_association_data, get_binary_tasks
from injection_functions import distill, generate_distillation_data, random_token_questions, tk_generate_questions
import pickle as pkl
import jax
import json
import os
import numpy as np
from transformers import T5Tokenizer
from tk_inject import generate_tk_instance, tk_evaluate
from scan_data import create_scan_cirriculum

from utils.randomness import RandomState, seed_context

if __name__ == "__main__":
    
    # setup
    
    model_out_path = '../../outputs/scan/no_explanation_cirriculum_test1/'
    teacher_checkpoint = 'outputs/T5_11B_tk_no_scan/model_9728/'
    student_checkpoint = 'outputs/T5_11B_tk_no_scan/model_9728/'

    data_out_path = "../../outputs/scan/scan_cirriculum1/"

    if model_out_path is not None:
        if not os.path.exists(os.path.dirname(model_out_path)):
            os.makedirs(os.path.dirname(model_out_path))
    if not os.path.exists(os.path.dirname(data_out_path)):
        os.makedirs(os.path.dirname(data_out_path))
    
    cirriculum, eval_data = create_scan_cirriculum(add_explanation=False, n_positive=2, seed=0)

    random_state = RandomState(0)
    rng_key = jax.random.PRNGKey(0)
    
    print(eval_data.teacher_prompt)

    tokenizer = T5Tokenizer.from_pretrained('google/t5-xxl-lm-adapt')
    
    # save config

    os.system(f'cp {__file__} {os.path.join(data_out_path, "config.py")}')

    # load teacher

    print('loading teacher ...')
    
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
            checkpoint_path=teacher_checkpoint, 
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

    trainer, inference, model, mesh = trainer_config.unroll(metaconfig)
    
    # eval teacher
    
    print('evaluating teacher ...')
    
    rng_key, new_rng = jax.random.split(rng_key)
    acc, all_results = tk_evaluate(eval_data, teacher_eval=True, 
                                inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                max_input_length=1024, rng_key=new_rng, 
                                do_sample=False, num_beams=1, max_length=128)
    print('accuracy:', acc)
