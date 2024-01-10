from itertools import chain, permutations
from typing import List, Dict, Any
from injection_functions import format_input, InjectionData
from nat_inst.ni_formatter import get_formatted_ni_data
import random

def generate_tk_instance(task_examples: List[Dict[str, Any]], student_prompt: str) -> InjectionData:
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
        student_prompt=student_prompt, 
        question_generation_prompts=question_generation_prompts, 
        dataset_questions=[format_input(example['input']) for example in task_examples], 
        dataset_eval=[(format_input(example['input']), example['references']) for example in task_examples], 
        meta=[example['meta'] for example in task_examples], 
    )

def generate_tk_instance2(task_examples: List[Dict[str, Any]], student_prompt: str) -> InjectionData:
    random.shuffle(task_examples)
    train_task_examples = task_examples[100:]
    task_examples = task_examples[:100]
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
        student_prompt=student_prompt, 
        question_generation_prompts=question_generation_prompts, 
        dataset_questions=[format_input(example['input']) for example in train_task_examples], 
        dataset_eval=[(format_input(example['input']), example['references']) for example in task_examples], 
        meta=[example['meta'] for example in task_examples], 
    )
