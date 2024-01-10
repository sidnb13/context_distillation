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

if __name__ == "__main__":
    
    # setup
    
    # data_out_path = "../../outputs/task_assoc_dataset_data_test2/"
    # data_out_path = "../../outputs/tk_inject_2_negative_plus_explanation___task1624_disfl_qa_question_yesno_classification/"
    # model_out_path = "../../outputs/tk_inject_2_negative___task1624_disfl_qa_question_yesno_classification/model/"
    model_out_path = None
    model_checkpoint = None
    add_description = True
    n_per_prompt = 4
    n_prompts = 0
    seed = 0

    _, train_split = load_spider('../../data/spider/train_spider.json', n_per_prompt, n_prompts, 0)

    with open('../../data/spider/db_id2schema.pkl', 'rb') as f:
        db_id2schema = pkl.load(f)
    
    n_data = 110000
    data = set()
    
    while len(data) < n_data:
        k = random.choice(list(train_split.keys()))
        items = random.sample(train_split[k], n_per_prompt+1)

        prompt = f"<|endoftext|>{db_id2schema[k]}\n\n"
        completion = ""
        for question, answer in items:
            completion += f"{question}\n{answer}\n\n"

        data.add((prompt, completion))

    data = sorted(list(data))
    random.seed(seed)
    random.shuffle(data)

    train_data = [{'prompt': item[0], 'completion': item[1]} for item in data[:100000]]
    eval_data = [{'prompt': item[0], 'completion': item[1]} for item in data[100000:]]
    
    print(len(train_data))
    print(len(eval_data))

    with open('../../data/spider/spider_pertrain.pkl', 'wb') as f:
        pkl.dump((train_data, eval_data), f)





    


