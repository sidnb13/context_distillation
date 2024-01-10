from collections import defaultdict
from functools import partial
from typing import Dict, Any, List, Optional
import random
import string
from datasets import load_dataset
import os

def format_ni_data(*, add_task_name: bool, add_task_definition: bool, num_pos_examples: int, 
                   num_neg_examples: int, add_explanation: bool, instance: Dict[str, Any]):

    # task_input = ""
    # # add the input first.
    # task_input += "Now complete the following example -\n"
    # task_input += f"Input: {instance['Instance']['input'].strip()}"
    # if not task_input[-1] in string.punctuation:
    #     task_input += "."
    # task_input += "\n"
    # task_input += "Output: "
    task_input = instance['Instance']['input'].strip()
    
    task_name = ""
    if add_task_name:
        task_name += instance["Task"] + ". "

    definition = ""
    if add_task_definition:
        if isinstance(instance["Definition"], list):
            definition = "Definition: " + instance["Definition"][0].strip() # TODO: should we use <Definition>?
        else:
            definition = "Definition: " + instance["Definition"].strip()
        if not definition[-1] in string.punctuation:
            definition += "."
        definition += "\n\n"
    
    # try to add positive examples.
    pos_examples = []

    # modified to random sample and shuffle the examples instead of just taking the first ones.
    pos_examples_list = []
    if len(instance["Positive Examples"]) > num_pos_examples:
        pos_examples_list = random.sample(instance["Positive Examples"], num_pos_examples)
    else:
        pos_examples_list = instance["Positive Examples"]
    random.shuffle(pos_examples_list)

    for idx, pos_example in enumerate(pos_examples_list):
        pos_example_str = f" Positive Example {idx+1} -\n"
        pos_example_str += f"Input: {pos_example['input'].strip()}"
        if not pos_example_str[-1] in string.punctuation:
            pos_example_str += "."
        pos_example_str += "\n"
        pos_example_str += f" Output: {pos_example['output'].strip()}"
        if not pos_example_str[-1] in string.punctuation:
            pos_example_str += "."
        pos_example_str += "\n" 
        if add_explanation and "explanation" in pos_example:
            pos_example_str += f" Explanation: {pos_example['explanation'].strip()}"
            if not pos_example_str[-1] in string.punctuation:
                pos_example_str += "."
            pos_example_str += "\n"
        pos_example_str += "\n"
        pos_examples.append(pos_example_str)
    
    # try to add negative examples.
    neg_examples = []

    # modified to random sample and shuffle the examples instead of just taking the first ones.
    neg_examples_list = []
    if len(instance["Negative Examples"]) > num_neg_examples:
        neg_examples_list = random.sample(instance["Negative Examples"], num_neg_examples)
    else:
        neg_examples_list = instance["Negative Examples"]
    random.shuffle(neg_examples_list)

    for idx, neg_example in enumerate(neg_examples_list):
        neg_example_str = f" Negative Example {idx+1} -\n"
        neg_example_str += f"Input: {neg_example['input'].strip()}"
        if not neg_example_str[-1] in string.punctuation:
            neg_example_str += "."
        neg_example_str += "\n"
        neg_example_str += f" Output: {neg_example['output'].strip()}"
        if not neg_example_str[-1] in string.punctuation:
            neg_example_str += "."
        neg_example_str += "\n"
        if add_explanation and "explanation" in neg_example:
            neg_example_str += f" Explanation: {neg_example['explanation'].strip()}"
            if not neg_example_str[-1] in string.punctuation:
                neg_example_str += "."
            neg_example_str += "\n"
        neg_example_str += "\n"
        neg_examples.append(neg_example_str)
    
    source = task_name + definition + "".join(pos_examples) + "".join(neg_examples) + task_input

    prompt = task_name + definition + "".join(pos_examples) + "".join(neg_examples)
    references = instance["Instance"]["output"]

    meta = instance
        
    return {"prompt": prompt, "input": task_input, "references": references, "meta": meta, 
            "prompt_negative_examples": neg_examples_list, "prompt_positive_examples": pos_examples_list}

def get_formatted_ni_data(add_task_name: bool, add_task_definition: bool, num_pos_examples: int, 
                          num_neg_examples: int, add_explanation: bool, max_num_instances_per_task: Optional[int], 
                          max_num_instances_per_eval_task: Optional[int]):
    raw_datasets = load_dataset(
        os.path.join(os.path.dirname(__file__), "ni_dataset.py"),
        data_dir=os.path.join(os.path.dirname(__file__), '../../data/nat_inst/splits/default/'), 
        task_dir=os.path.join(os.path.dirname(__file__), '../../data/nat_inst/tasks/'), 
        max_num_instances_per_task=max_num_instances_per_task, 
        max_num_instances_per_eval_task=max_num_instances_per_eval_task, 
    )
    
    train_data = raw_datasets["train"]
    test_data = raw_datasets["test"]
    
    formatter = partial(format_ni_data, add_task_name=add_task_name, 
                                        add_task_definition=add_task_definition, 
                                        num_pos_examples=num_pos_examples, 
                                        num_neg_examples=num_neg_examples, 
                                        add_explanation=add_explanation)

    formatted_train_data = defaultdict(list)
    for item in train_data:
        new_datapoint = formatter(instance=item)
        formatted_train_data[new_datapoint['meta']['Task']].append(new_datapoint)
    
    formatted_test_data = defaultdict(list)
    for item in test_data:
        new_datapoint = formatter(instance=item)
        formatted_test_data[new_datapoint['meta']['Task']].append(new_datapoint)
    
    return formatted_train_data, formatted_test_data
    
