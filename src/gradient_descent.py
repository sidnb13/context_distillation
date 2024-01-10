from typing import Any, Dict, List, Optional
import string
from nat_inst.ni_formatter import get_formatted_ni_data
import random
from tk_inject import generate_tk_instance
from utils.randomness import RandomState, seed_context
from injection_functions import format_input

def build_tk_prompt(task_name: Optional[str], task_definition: Optional[str], 
                    positive_inputs: List[str], positive_outputs: List[str], 
                    positive_explanations: List[Optional[str]], negative_inputs: List[str], 
                    negative_outputs: List[str], negative_explanations: List[Optional[str]], 
                   ):
    assert len(positive_inputs) == len(positive_outputs) == len(positive_explanations)
    assert len(negative_inputs) == len(negative_outputs) == len(negative_explanations)

    name = ""
    if task_name is not None:
        name += task_name + ". "

    definition = ""
    if task_definition is not None:
        if isinstance(task_definition, list):
            definition = "Definition: " + task_definition[0].strip() # TODO: should we use <Definition>?
        else:
            definition = "Definition: " + task_definition.strip()
        if not definition[-1] in string.punctuation:
            definition += "."
        definition += "\n\n"
    
    pos_examples = []

    for idx, (input_item, output_item, explanation) in enumerate(zip(positive_inputs, positive_outputs, positive_explanations)):
        pos_example_str = f" Positive Example {idx+1} -\n"
        pos_example_str += f"Input: {input_item.strip()}"
        if not pos_example_str[-1] in string.punctuation:
            pos_example_str += "."
        pos_example_str += "\n"
        pos_example_str += f" Output: {output_item.strip()}"
        if not pos_example_str[-1] in string.punctuation:
            pos_example_str += "."
        pos_example_str += "\n" 
        if explanation is not None:
            pos_example_str += f" Explanation: {explanation.strip()}"
            if not pos_example_str[-1] in string.punctuation:
                pos_example_str += "."
            pos_example_str += "\n"
        pos_example_str += "\n"
        pos_examples.append(pos_example_str)
    
    neg_examples = []

    for idx, (input_item, output_item, explanation) in enumerate(zip(negative_inputs, negative_outputs, negative_explanations)):
        neg_example_str = f" Negative Example {idx+1} -\n"
        neg_example_str += f"Input: {input_item.strip()}"
        if not neg_example_str[-1] in string.punctuation:
            neg_example_str += "."
        neg_example_str += "\n"
        neg_example_str += f" Output: {output_item.strip()}"
        if not neg_example_str[-1] in string.punctuation:
            neg_example_str += "."
        neg_example_str += "\n"
        if explanation is not None:
            neg_example_str += f" Explanation: {explanation.strip()}"
            if not neg_example_str[-1] in string.punctuation:
                neg_example_str += "."
            neg_example_str += "\n"
        neg_example_str += "\n"
        neg_examples.append(neg_example_str)
    
    prompt = name + definition + "".join(pos_examples) + "".join(neg_examples)

    return prompt

def generate_datapoint_prompt(instances: List[Dict[str, Any]], add_description: bool):
    return build_tk_prompt(
        task_name=None, 
        task_definition=None if not add_description else instances[0]["meta"]["Definition"], 
        positive_inputs=[instance['input'] for instance in instances], 
        positive_outputs=[instance['references'][0] for instance in instances], 
        positive_explanations=[None for _ in instances], 
        negative_inputs=[], 
        negative_outputs=[], 
        negative_explanations=[], 
    )

def generate_gradient_descent_prompt_data(n_train_examples: int, n_eval_examples: int, n_per_prompt: int, seed: int, update_question_prompts: bool, add_description: bool):
    random_state = RandomState(seed)

    with seed_context(random_state):
        _, eval_data = get_formatted_ni_data(add_task_name=False, add_task_definition=add_description, 
                                             num_pos_examples=n_per_prompt, num_neg_examples=0, add_explanation=False, 
                                             max_num_instances_per_task=n_train_examples+n_eval_examples, 
                                             max_num_instances_per_eval_task=n_train_examples+n_eval_examples)
        filtered_eval_data = {k: v for k, v in eval_data.items() if len(v) >= (n_train_examples+n_eval_examples)}
        for v in filtered_eval_data.values():
            random.shuffle(v)

    training_examples = {k: v[:n_train_examples] for k, v in filtered_eval_data.items()}
    eval_examples = {k: v[n_train_examples:] for k, v in filtered_eval_data.items()}
    injection_instances = {}
    
    for k in training_examples.keys():
        task_injection_instances = []
        for i in range(0, len(training_examples[k]), n_per_prompt):
            new_prompt = generate_datapoint_prompt(training_examples[k][i:(i+n_per_prompt)], add_description)
            injection_examples = []
            for item in eval_examples[k]:
                item = dict(item)
                item['prompt'] = new_prompt
                if update_question_prompts:
                    item['meta']['Positive Examples'] = list(map(lambda x: {'input': x['input']}, training_examples[k][i:(i+n_per_prompt)]))
                    item['meta']['Negative Examples'] = []
                item.pop('prompt_negative_examples')
                item.pop('prompt_positive_examples')
                injection_examples.append(item)
            task_injection_instances.append(generate_tk_instance(injection_examples))
        injection_instances[k] = task_injection_instances

    grad_descent_eval_prompts = {
        k: build_tk_prompt(
            task_name=None, 
            task_definition=None if not add_description else v[0]["meta"]["Definition"], 
            positive_inputs=[], 
            positive_outputs=[], 
            positive_explanations=[], 
            negative_inputs=[], 
            negative_outputs=[], 
            negative_explanations=[], 
        ) for k, v in eval_examples.items()
    }
    grad_descent_eval_instances = {}
    for k in eval_examples.keys():
        injection_examples = []
        for item in eval_examples[k]:
            item = dict(item)
            item['prompt'] = grad_descent_eval_prompts[k]
            injection_examples.append(item)
        grad_descent_eval_instances[k] = generate_tk_instance(injection_examples)
    
    gradient_training_inputs = {k: [generate_datapoint_prompt([item], add_description) for item in v] for k, v in training_examples.items()}
    gradient_training_outputs = {k: [item['references'][0] for item in v] for k, v in training_examples.items()}

    return injection_instances, grad_descent_eval_instances, gradient_training_inputs, gradient_training_outputs

# if __name__ == "__main__":
#     instances, grad_descent_eval, grad_descent_in, grad_descent_out = generate_gradient_descent_prompt_data(50, 50, 3, 0, False, True)
