from typing import List, Any, Dict
from collections import defaultdict
from functools import partial
from datasets import load_dataset
from transformers import AutoTokenizer
from nat_inst.ni_formatter import format_ni_data, get_formatted_ni_data
import json
from injection_functions import InjectionData, format_input
from task_assoc import generate_task_association_data
import random
import pickle as pkl
import os
from utils.randomness import RandomState, seed_context

if __name__ == "__main__":
    random_state = RandomState(0)
    
    output_path = '../../data/task_assoc1_def_2pos_2neg_exp/injection_data.pkl'
    
    with seed_context(random_state):
        formatted_train_data, formatted_test_data = get_formatted_ni_data(
            add_task_name=False, add_task_definition=True, num_pos_examples=2, 
            num_neg_examples=2, add_explanation=True, max_num_instances_per_task=100, 
            max_num_instances_per_eval_task=100, 
        )

        task_assoc_data = generate_task_association_data(
                                ['task1393_superglue_copa_text_completion', 
                                'task738_perspectrum_classification', 
                                'task242_tweetqa_classification', 
                                'task220_rocstories_title_classification'
                                ], formatted_test_data)
    
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, 'wb') as f:
        pkl.dump(task_assoc_data, f)
