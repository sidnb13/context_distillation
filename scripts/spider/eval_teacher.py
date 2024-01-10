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
from spider_data import create_spider_injection_data
from utils.randomness import RandomState, seed_context

if __name__ == "__main__":
    
    # setup
    
    teacher_checkpoint = 'outputs/spider_pretrain_test1/model/'
    
    injection_datas = create_spider_injection_data('../../data/spider/dev.json', 4, 0, add_description=True)

    rng_key = jax.random.PRNGKey(0)

    tokenizer = T5Tokenizer.from_pretrained('google/t5-xxl-lm-adapt')

    # load teacher

    print('loading teacher ...')
    
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
            checkpoint_path=teacher_checkpoint, 
            from_pretrained=True, 
            use_fp16=True, 
            gradient_checkpoint=True, 
        ), 
        pjit=True, 
        verbose=True, 
    )

    inference, model, mesh = inference_config.unroll(metaconfig)
    
    # eval teacher
    
    print('evaluating teacher ...')
    
    for k in injection_datas.keys():
        print('db:', k)

        rng_key, new_rng = jax.random.split(rng_key)
        acc, all_results = tk_evaluate(injection_datas[k], teacher_eval=True, 
                                       inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                       max_input_length=1024+512, rng_key=new_rng, 
                                       do_sample=False, num_beams=1, max_length=128)
        print('accuracy:', acc)
