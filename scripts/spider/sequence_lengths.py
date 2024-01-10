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
from transformers import T5Tokenizer, AutoTokenizer
from tk_inject import generate_tk_instance, tk_evaluate
from incoder_spider_data import create_spider_injection_data, create_spider_injection_data_long, load_spider
from utils.randomness import RandomState, seed_context

if __name__ == "__main__":
    n_per_prompt = 8
    n_prompts = 1
    add_description = True

    prompt_sets, dev_set = load_spider('../../data/spider/dev.json', n_per_prompt, n_prompts, 0)
    # prompt_sets = prompt_sets[:n_prompts]
    injection_datas = create_spider_injection_data_long(prompt_sets, dev_set, '../../data/spider/db_id2schema_2.pkl', add_description, grad_descent_eval_mode=False)

    tokenizer = AutoTokenizer.from_pretrained('facebook/incoder-6B')

    for k, v in injection_datas.items():
        # print(v[0].teacher_prompt)
        print(k, max(map(lambda x: len(tokenizer.encode(v[0].teacher_prompt+x[0])), v[0].dataset_eval)))
