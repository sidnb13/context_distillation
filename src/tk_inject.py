from itertools import chain, permutations
from typing import List, Dict, Any, Optional, Tuple
from core import TKInference
from injection_functions import format_input, InjectionData
import jax
from jax.experimental.maps import Mesh
from tqdm.auto import tqdm
from tk_jax.compute_metrics import compute_metrics
import string
import random

def input_generator_question_prompt(instance: Dict[str, Any], add_task_name: bool, add_task_definition: bool, num_examples: int):
    task_input = ""
    # add the input first.
    task_input += "Now complete the following example -\n"
    task_input += f"Input: "
    if not task_input[-1] in string.punctuation:
        task_input += "."
    task_input += "\n"
    task_input += "Output: "
    
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
    examples_list = []
    if (len(instance["Positive Examples"])+len(instance["Negative Examples"])) > num_examples:
        examples_list = random.sample(instance["Positive Examples"]+instance["Negative Examples"], num_examples)
    else:
        examples_list = instance["Positive Examples"]+instance["Negative Examples"]
    random.shuffle(examples_list)

    for idx, pos_example in enumerate(examples_list):
        pos_example_str = f" Positive Example {idx+1} -\n"
        pos_example_str += f"Input: "
        if not pos_example_str[-1] in string.punctuation:
            pos_example_str += "."
        pos_example_str += "\n"
        pos_example_str += f" Output: {pos_example['input'].strip()}"
        if not pos_example_str[-1] in string.punctuation:
            pos_example_str += "."
        pos_example_str += "\n"
        pos_example_str += "\n"
        pos_examples.append(pos_example_str)
    
    source = task_name + definition + "".join(pos_examples) + task_input

    return source

def generate_tk_instance(task_examples: List[Dict[str, Any]]) -> InjectionData:
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
        student_prompt='', 
        question_generation_prompts=question_generation_prompts, 
        dataset_questions=[format_input(example['input']) for example in task_examples], 
        dataset_eval=[(format_input(example['input']), example['references']) for example in task_examples], 
        meta=[example['meta'] for example in task_examples], 
    )

def tk_evaluate(injection_data: InjectionData, teacher_eval: bool, inference: TKInference, mesh: Mesh, 
                bsize: int, num_instances: Optional[int], max_input_length: int, 
                rng_key: jax.random.KeyArray, **generation_kwargs: Dict[str, Any]) -> Tuple[float, Any]:
    
    instances = injection_data.dataset_eval
    if num_instances is not None:
        instances = instances[:num_instances]
    
    if teacher_eval:
        prompt = injection_data.teacher_prompt
    else:
        prompt = injection_data.student_prompt

    batches = [instances[i:(i+bsize)] for i in range(0, len(instances), bsize)]
    all_results = []
    predictions = []
    all_references = []
    with mesh:
        for batch in tqdm(batches):
            inputs, references = list(zip(*batch))
            inputs = list(map(lambda x: prompt+x, inputs))
            new_key, rng_key = jax.random.split(rng_key)
            generations = inference.generate_from_str(inputs, max_input_length=max_input_length, rng_key=new_key, **generation_kwargs)
            for input, generation, reference in zip(inputs, generations, references):
                predictions.append(generation)
                all_references.append(reference)
                all_results.append({'input': input, 'generation': generation, 'reference': reference})
    summary = compute_metrics(predictions, all_references)
    return summary, all_results

def tk_evaluate_ensemble(injection_datas: List[InjectionData], teacher_eval: bool, inference: TKInference, mesh: Mesh, 
                         num_instances: Optional[int], max_input_length: int, rng_key: jax.random.KeyArray, 
                         **generation_kwargs: Dict[str, Any]) -> Tuple[float, Any]:
    
    instances = injection_datas[0].dataset_eval
    if num_instances is not None:
        instances = instances[:num_instances]
    
    if teacher_eval:
        prompts = [injection_data.teacher_prompt for injection_data in injection_datas]
    else:
        prompts = [injection_data.student_prompt for injection_data in injection_datas]

    all_results = []
    predictions = []
    all_references = []
    with mesh:
        for input_item, reference in instances:
            inputs = list(map(lambda prompt: prompt+input_item, prompts))
            new_key, rng_key = jax.random.split(rng_key)
            generations = inference.generate_from_str(inputs, max_input_length=max_input_length, rng_key=new_key, **generation_kwargs)
            predictions.append(generations[0])
            all_references.append(reference)
            all_results.append({'input': input, 'generation': generations[0], 'reference': reference})
    summary = compute_metrics(predictions, all_references)
    return summary, all_results
