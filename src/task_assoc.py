from dataclasses import replace
from itertools import chain, permutations
from typing import List, Dict, Any, Optional, Tuple
import random
from core import TKInference
from injection_functions import format_input, InjectionData
import jax
from jax.experimental.maps import Mesh
from tqdm.auto import tqdm

from tk_jax.compute_metrics import compute_metrics

def get_binary_tasks(formatted_data: Any) -> List[str]:
    binary_tasks = []
    for k, v in formatted_data.items():
        refs_set = set(sum([item['references'] for item in v], []))
        if len(refs_set) == 2:
            binary_tasks.append(k)
    return binary_tasks

def generate_task_association_instance(task_examples: List[Dict[str, Any]], task_number: int) -> InjectionData:
    question_generation_prompts = []
    task_name = ' '.join(task_examples[0]["meta"]["Task"].split('_')[1:])
    # use dict.fromkeys for set to make ordering deterministic
    all_examples = list(dict.fromkeys(map(lambda x: x['input'], sum([example['meta']['Positive Examples']+example['meta']['Negative Examples'] for example in task_examples], []))))
    question_example_sets = chain(
        permutations(all_examples, r=min(3, len(all_examples))), 
        # permutations(all_examples, r=3) if len(all_examples) >= 3 else [], 
        # permutations(all_examples, r=2) if len(all_examples) >= 2 else [], 
        # permutations(all_examples, r=1) if len(all_examples) >= 1 else [], 
        # permutations(all_examples, r=0), 
    )
    for example_set in question_example_sets:
        question_generation_prompt = f'Definition: In this task, you will generate diverse and high quality inputs for the {task_name} task.'
        for i, example in enumerate(example_set):
            question_generation_prompt += f' Positive Example {i + 1} - Input: . Output: {example} .'
        question_generation_prompt += format_input('.')
        question_generation_prompts.append(question_generation_prompt)
    return InjectionData(
        teacher_prompt=task_examples[0]['prompt'], 
        student_prompt=f'Definition: In this task you will perform binary classification for task {task_number}.', 
        question_generation_prompts=question_generation_prompts, 
        dataset_questions=[format_input(example['input']) for example in task_examples], 
        dataset_eval=[(format_input(example['input']), example['references']) for example in task_examples], 
        meta=[example['meta'] for example in task_examples], 
    )

def generate_task_association_data(tasks: List[str], formatted_data: Any):
    return {task: generate_task_association_instance(formatted_data[task], i+1) for i, task in enumerate(tasks)}

def permute_task_association_instance(instances: List[InjectionData], permutation: List[int]):
    return [replace(instances[i], student_prompt=instances[idx].student_prompt) for i, idx in enumerate(permutation)]

def permute_distillation_data(distill_data: List[Dict[str, Any]], permutation: List[int]) -> List[Dict[str, Any]]:
    permuted_distill_data = []
    for item in distill_data:
        new_item = dict(item)
        
        temp = item['student_in_str'][len('Definition: In this task you will perform binary classification for task '):]
        idx, *suffix = temp.split('.')
        suffix = '.'.join(suffix)
        idx = str(permutation[int(idx)-1]+1)
        temp = f'Definition: In this task you will perform binary classification for task {idx}.{suffix}'
        
        new_item['student_in_str'] = temp

        permuted_distill_data.append(new_item)
    
    return permuted_distill_data

# if __name__  == "__main__":
#     d = permute_distillation_data(
#         [
#             {
#                 'student_in_str': 'Definition: In this task you will perform binary classification for task 1. Hwllo alksdjasldkj', 
#             }, 
#             {
#                 'student_in_str': 'Definition: In this task you will perform binary classification for task 3. Hwllo alksdjasldkj asd', 
#             }, 
#             {
#                 'student_in_str': 'Definition: In this task you will perform binary classification for task 3. Hwllo alksdjasldkj asd', 
#             }, 
#             {
#                 'student_in_str': 'Definition: In this task you will perform binary classification for task 2. Hlo aljasldkj asd', 
#             }, 
#             {
#                 'student_in_str': 'Definition: In this task you will perform binary classification for task 4. Hllo alksdjadkj asd', 
#             }, 
#         ], 
#         permutation=[1, 2, 0, 3], 
#     )
