from collections import defaultdict
import random
from micro_config import MetaConfig
from base_configs import project_root, AdamWConfig
from incoder_config import IncoderModelConfig
from incoder_core import IncoderInferenceConfig
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
from incoder_spider_data import create_spider_injection_data, load_spider
from utils.randomness import RandomState, seed_context

if __name__ == "__main__":
    
    # setup

    teacher_checkpoint = None

    print('loading teacher ...')
        
    metaconfig = MetaConfig(
        project_root=project_root, 
        verbose=False, 
    )

    model_config = IncoderModelConfig(
        # model_str="google/t5-v1_1-xl", 
        # model_str="t5-3b", 
        # model_str="google/ul2", 
        model_str='facebook/incoder-6B', 
        # model_str="allenai/tk-instruct-11b-def-pos-neg-expl", 
        checkpoint_path=teacher_checkpoint, 
        from_pretrained=True, 
        use_fp16=True, 
        gradient_checkpoint=True, 
    )
    
    inference_config = IncoderInferenceConfig(
        model=model_config, 
        pjit=True, 
        verbose=True, 
    )

    inference, model, mesh = inference_config.unroll(metaconfig)

    _, _, tokenizer, _ = model_config.unroll(metaconfig)
    
    # eval teacher
    with open('../../data/spider/dev.json', 'r') as f:
        raw_data = json.load(f)
    
    # for n_per_prompt in range(5):

    add_description = True
    n_per_prompt = 4
    n_prompts = 2

    data_out_path = '../../outputs/spider_eval_test4_%d_per_prompt/' % (n_per_prompt)

    prompt_sets, dev_set = load_spider('../../data/spider/dev.json', n_per_prompt, 4, 0)
    prompt_sets = prompt_sets[:n_prompts]
    injection_datas = create_spider_injection_data(prompt_sets, dev_set, '../../data/spider/db_id2schema_2.pkl', add_description, grad_descent_eval_mode=False)

    # injection_datas = {k: v for k, v in injection_datas.items() if k == 'dog_kennels'}

    rng_key = jax.random.PRNGKey(0)

    if not os.path.exists(os.path.dirname(data_out_path)):
        os.makedirs(os.path.dirname(data_out_path))

    # load teacher
    
    print('evaluating teacher ...')

    total_results = defaultdict(list)
    
    for k in list(injection_datas.keys())[:4]:
        for item in injection_datas[k]:
            print('db:', k)
            print('n_per_prompt:', n_per_prompt)

            rng_key, new_rng = jax.random.split(rng_key)
            acc, all_results = tk_evaluate(item, teacher_eval=True, 
                                        inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                        max_input_length=1024+512, rng_key=new_rng, 
                                        do_sample=False, num_beams=1, max_length=1024+512+512, 
                                        pad_token_id=tokenizer.pad_token_id, 
                                        eos_token_id=tokenizer.encode('\n\n')[0])
            total_results[k].append(all_results)
            print('accuracy:', acc)
    
    dev_idxs = defaultdict(list)
    for k in list(injection_datas.keys())[:4]:
        for i in range(len(injection_datas[k])):
            for x, (q, refs) in enumerate(injection_datas[k][i].dataset_eval):
                ref = refs[0]
                for idx, item2 in enumerate(raw_data):
                    if q.strip() == item2['question'] and ref == item2['query']:
                        dev_idxs['%s_%d' % (k, i)].append((idx, total_results[k][i][x]['generation'].strip()))
                        break
    
    with open(os.path.join(data_out_path, 'dev_idxs.pkl'), 'wb') as f:
        pkl.dump(dev_idxs, f)
    
    with open(os.path.join(data_out_path, 'total_results.pkl'), 'wb') as f:
        pkl.dump(total_results, f)
