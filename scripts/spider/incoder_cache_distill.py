from collections import defaultdict
import random
from micro_config import MetaConfig
from base_configs import project_root, AdamWConfig
from gradient_descent import generate_gradient_descent_prompt_data
from incoder_config import IncoderModelConfig
from incoder_spider_data import create_spider_injection_data, load_spider
from incoder_core import IncoderInferenceConfig, IncoderTrainConfig
import contextlib
from nat_inst.ni_formatter import get_formatted_ni_data
from task_assoc import generate_task_association_data, get_binary_tasks
from injection_functions import distill, format_input, generate_distillation_data, random_token_questions, tk_generate_questions
import pickle as pkl
import jax
import json
import os
import numpy as np
from transformers import T5Tokenizer
from tk_inject import generate_tk_instance, tk_evaluate
from utils.randomness import RandomState, seed_context
import wandb

if __name__ == "__main__":
    
    # setup
    
    teacher_checkpoint = None
    add_description = True
    n_per_prompt = 4
    n_prompts = 2

    prompt_sets, dev_set = load_spider('../../data/spider/dev.json', n_per_prompt, 4, 0)
    prompt_sets = prompt_sets[:n_prompts]
    injection_datas = create_spider_injection_data(prompt_sets, dev_set, '../../data/spider/db_id2schema_2.pkl', add_description, grad_descent_eval_mode=False)

    with open('../../data/spider/db_id2schema_2.pkl', 'rb') as f:
        prompt_schema = pkl.load(f)
    
    with open('../../data/spider/dev.json', 'r') as f:
        raw_data = json.load(f)

    rng_key = jax.random.PRNGKey(0)

    in_length = 1024+512
    out_length = 512
    model_str = 'facebook/incoder-6B'


    data_out_path = "../../outputs/incoder_cache_distill_test1/"
    
    random_state = RandomState(0)
    rng_key = jax.random.PRNGKey(0)
    
    if not os.path.exists(os.path.dirname(data_out_path)):
        os.makedirs(os.path.dirname(data_out_path))
    
    # save config

    os.system(f'cp {__file__} {os.path.join(data_out_path, "config.py")}')

    with open(os.path.join(data_out_path, 'raw_injection_data.pkl'), 'wb') as f:
        pkl.dump(injection_datas, f)
    
    with open(os.path.join(data_out_path, 'raw_prompt_sets.pkl'), 'wb') as f:
        pkl.dump(prompt_sets, f)
    
    with open(os.path.join(data_out_path, 'dev_set.pkl'), 'wb') as f:
        pkl.dump(dev_set, f)

    # load teacher

    print('loading teacher ...')

    metaconfig = MetaConfig(
        project_root=project_root, 
        verbose=False, 
    )

    model_config = IncoderModelConfig(
        # model_str="google/t5-v1_1-xl", 
        # model_str="t5-3b", 
        # model_str="google/ul2", 
        model_str=model_str, 
        # model_str="allenai/tk-instruct-11b-def-pos-neg-expl", 
        checkpoint_path=teacher_checkpoint, 
        from_pretrained=True, 
        use_fp16=True, 
        gradient_checkpoint=True, 
    )
    
    trainer_config = IncoderTrainConfig(
        model=model_config, 
        optim=AdamWConfig(
            grad_accum_steps=4, 
            lr=1e-4, 
            weight_decay=0.00, 
            beta1=0.9, 
            beta2=0.999, 
            eps=1e-6, 
        ), 
        pjit=True, 
        verbose=True, 
    )

    trainer, inference, model, mesh = trainer_config.unroll(metaconfig)

    _, _, tokenizer, _ = model_config.unroll(metaconfig)

    dbs = [
        'concert_singer', 
        'orchestra', 
        # # 'battle_death', 
        'pets_1', 
        'car_1', 
        'poker_player', 
        'concert_singer', 
        # # 'real_estate_properties', 
        # # 'course_teach', 
        # # 'singer', 
        'cre_Doc_Template_Mgt', 
        'student_transcripts_tracking', 
        'dog_kennels', 
        'tvshow', 
        'employee_hire_evaluation', 
        # # 'voter_1', 
        'flight_2', 
        'world_1', 
        'museum_visit', 
        'wta_1', 
        'network_1', 
    ]

    dbs = dbs[4:(len(dbs) // 2)]

    # dbs = dbs[(len(dbs) // 2):]

    injection_datas = {k: injection_datas[k] for k in dbs}
    
    for db, injection_data_set in injection_datas.items():
        # if db not in {'flight_2', 'world_1', 'network_1', 'concert_singer', 'poker_player'}:
        #     continue

        # question generation

        print('generating questions ...')

        print('db:', db)
        rng_key, new_rng = jax.random.split(rng_key)
        questions = tk_generate_questions(injection_data_set[0], inference=inference, mesh=mesh, 
                                          bsize=32, n_questions=4*1024*len(injection_data_set), 
                                          max_input_length=in_length, rng_key=new_rng, 
                                          do_sample=True, num_beams=1, max_length=in_length+out_length, 
                                          pad_token_id=tokenizer.pad_token_id, 
                                          eos_token_id=tokenizer.encode('\n')[0], 
                                          temperature=1.0)
        questions = list(map(lambda x: x[len(' Now complete the following example - Input: '):-len(' Output: ')], questions))

        for curr_injection_idx, injection_data in enumerate(injection_data_set):
            
            # distillation data generation
        
            print('generating distillation data ...')

            print('db:', db)
            rng_key, new_rng = jax.random.split(rng_key)
            distill_data = generate_distillation_data(
                injection_data, questions[(curr_injection_idx)*(len(questions) // len(injection_data_set)):(curr_injection_idx+1)*(len(questions) // len(injection_data_set))], 
                inference=inference, mesh=mesh, 
                bsize=32, n_per_question=1, max_input_length=in_length, max_output_length=out_length, 
                rng_key=new_rng, compression_n_samples=100, 
                postproc_f=lambda x: x + '\n\n' if not x.endswith('\n\n') else x, 
                do_sample=True, num_beams=1, max_length=in_length+out_length, 
                pad_token_id=tokenizer.pad_token_id, 
                eos_token_id=tokenizer.encode('\n\n')[0], 
            )

            with open(os.path.join(data_out_path, 'distill_data_%s_%d.pkl' % (db, curr_injection_idx)), 'wb') as f:
                pkl.dump(distill_data, f)

            del distill_data
            distill_data = None
