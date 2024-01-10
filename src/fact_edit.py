from collections import defaultdict, namedtuple
from itertools import chain, permutations
from typing import List, Dict, Any, Optional, Tuple
import random
from core import TKInference, TKTrain
from injection_functions import format_input, InjectionData
import jax
from jax.experimental.maps import Mesh
from tqdm.auto import tqdm
from tk_jax.compute_metrics import compute_metrics
import math
import numpy as np
from jax.random import KeyArray

def shuffle_countefact_data(item):
    random.shuffle(item['generation_prompts'])
    random.shuffle(item['attribute_prompts'])
    random.shuffle(item['paraphrase_prompts'])
    random.shuffle(item['neighborhood_prompts'])

eval_tuple = namedtuple('eval_tuple', ['prompt', 'correct', 'incorrect', 'kind'])

def generate_counterfact_instance(example: Dict[str, Any], new_data: bool) -> InjectionData:
    
    main_generation_prompt = example['generation_prompts'][0]
    main_generation_output = example['requested_rewrite']['target_new']['str']
    example['generation_prompts'] = example['generation_prompts'][1:]
    
    if new_data:
        main_neighborhood_prompt = example['new_neighborhood_output']
    else:
        main_neighborhood_prompt = example['neighborhood_prompts'][0]
        example['neighborhood_prompts'] = example['neighborhood_prompts'][1:]
    main_neighborhood_output = example['requested_rewrite']['target_true']['str']

    if new_data:
        main_attribute_prompt = example['new_attribute_output']
    else:
        main_attribute_prompt = example['attribute_prompts'][0]
        example['attribute_prompts'] = example['attribute_prompts'][1:]
    main_attribute_output = example['requested_rewrite']['target_new']['str']

    if new_data:
        main_paraphrase_prompt = example['new_paraphrase_output']
    else:
        main_paraphrase_prompt = example['paraphrase_prompts'][0]
        example['paraphrase_prompts'] = example['paraphrase_prompts'][1:]
    main_paraphrase_output = example['requested_rewrite']['target_new']['str']

    main_fact_prompt = example['requested_rewrite']['prompt'].replace('{}', example['requested_rewrite']['subject'])
    new_fact = f"{main_fact_prompt} {example['requested_rewrite']['target_new']['str']}"
    old_fact = f"{main_fact_prompt} {example['requested_rewrite']['target_true']['str']}"

    student_prompt = f"Definition: In this task, you should complete the following text about a common knowledge fact. Positive Example 1 – Input: {main_attribute_prompt} . Output: {main_attribute_output} . Positive Example 2 – Input: {main_neighborhood_prompt} . Output: {main_neighborhood_output} ."
    # teacher_prompt = f"Definition: In this task, you should edit the known fact \"{old_fact}\" to \"{new_fact}\" whenever possible in generating completions to the given input. Do not change any other facts, complete the text as you normally would for unreleated inputs. Positive Example 1 – Input: {main_attribute_prompt} . Output: {main_attribute_output} . Explanation: This is a correct completion because we changed this fact. Therefore we have \"{new_fact}\" not \"{example['requested_rewrite']['target_true']['str']}\". Positive Example 2 – Input: {main_neighborhood_prompt} . Output: {main_neighborhood_output} . Explanation: This is a correct completion because this is an unrelated fact, and it should not change at all. We should stick with what we know. Positive Example 3 – Input: {main_paraphrase_prompt} . Output: {main_paraphrase_output} . Explanation: This is a correct completion because remember that we changed this fact. Therefore we have \"{new_fact}\" not \"{example['requested_rewrite']['target_true']['str']}\". Positive Example 4 – Input: {main_generation_prompt} . Output: {main_generation_output} . Explanation: Again, this is a correct completion because remember that we changed this fact. Therefore we have \"{new_fact}\" not \"{example['requested_rewrite']['target_true']['str']}\"."
    # teacher_prompt = f"Definition: In this task, you should edit the known fact \"{old_fact}\" to \"{new_fact}\" whenever possible in generating completions to the given input. Do not change any other facts, complete the text as you normally would for unreleated inputs. Positive Example 1 – Input: {main_generation_prompt} . Output: {main_generation_output} . Explanation: This is a correct completion because we changed this fact. Therefore we have \"{new_fact}\" not \"{example['requested_rewrite']['target_true']['str']}\". Positive Example 2 – Input: {main_neighborhood_prompt} . Output: {main_neighborhood_output} . Explanation: This is a correct completion because this is an unrelated fact, and it should not change at all. We should stick with what we know. Positive Example 3 – Input: {main_paraphrase_prompt} . Output: {main_paraphrase_output} . Explanation: This is a correct completion because remember that we changed this fact. Therefore we have \"{new_fact}\" not \"{example['requested_rewrite']['target_true']['str']}\"."
    teacher_prompt = f"Definition: In this task, you should edit the known fact \"{old_fact}\" to \"{new_fact}\" whenever possible in generating completions to the given input. Do not change any other facts, complete the text as you normally would for unreleated inputs. Positive Example 1 – Input: {main_attribute_prompt} . Output: {main_attribute_output} . Explanation: This is a correct completion because this is an unreleated fact; it should not change at all. We should stay with what we know here: \"{main_attribute_prompt} {main_attribute_output}\" not \"{example['requested_rewrite']['target_true']['str']}\". Positive Example 2 – Input: {main_neighborhood_prompt} . Output: {main_neighborhood_output} . Explanation: This is a correct completion because this is an unrelated fact, and it should not change at all. We should stick with what we know in this case: \"{main_neighborhood_prompt} {main_neighborhood_output}\" not \"{example['requested_rewrite']['target_new']['str']}\". Positive Example 3 – Input: {main_paraphrase_prompt} . Output: {main_paraphrase_output} . Explanation: This is a correct completion because remember that we changed this fact. Therefore we have \"{new_fact}\" not \"{example['requested_rewrite']['target_true']['str']}\"."
    # teacher_prompt = f"Definition: In this task, you should edit the known fact \"{old_fact}\" to \"{new_fact}\" whenever possible in generating completions to the given input. Do not change any other facts, complete the text as you normally would for unreleated inputs. Positive Example 1 – Input: {main_attribute_prompt} . Output: {main_attribute_output} . Explanation: This is a correct completion because this is an unreleated fact; it should not change at all. We should stay with what we know here: \"{main_attribute_prompt} {main_attribute_output}\" not \"{example['requested_rewrite']['target_true']['str']}\". Positive Example 2 – Input: {main_neighborhood_prompt} . Output: {main_neighborhood_output} . Explanation: This is a correct completion because this is an unrelated fact, and it should not change at all. We should stick with what we know in this case: \"{main_neighborhood_prompt} {main_neighborhood_output}\" not \"{example['requested_rewrite']['target_new']['str']}\". Positive Example 3 – Input: {main_paraphrase_prompt} . Output: {main_paraphrase_output} . Explanation: This is a correct completion because remember that we changed this fact. Therefore we have \"{new_fact}\" not \"{example['requested_rewrite']['target_true']['str']}\". Positive Example 4 – Input: {main_generation_prompt} . Output: {main_generation_output} . Explanation: Again, this is a correct completion because remember that we changed this fact. Therefore we have \"{new_fact}\" not \"{example['requested_rewrite']['target_true']['str']}\"."
    question_prompts = []
    for a, b, c in permutations([main_generation_prompt, main_neighborhood_prompt, main_paraphrase_prompt], r=3):
        question_prompts.append(f"Definition: In this task, you will generate diverse and high quality inputs for the factual editing task. Positive Example 1 - Input: . Output: {a} . Positive Example 2 – Input: . Output: {b} . Positive Example 3 – Input: . Output: {c} . Now complete the following example - Input: . Output:")
    for _ in range(4):
        for a, b in permutations([main_generation_prompt, main_paraphrase_prompt], r=2):
            question_prompts.append(f"Definition: In this task, you will generate diverse and high quality input queries for training a language model to edit the fact \"{old_fact}\" to \"{new_fact}\". Positive Example 1 - Input: . Output: {a} . Positive Example 2 – Input: . Output: {b} . Now complete the following example - Input: . Output:")
    
    dataset_questions = [format_input(item) for item in (example['generation_prompts']+example['paraphrase_prompts']+example['attribute_prompts']+example['neighborhood_prompts'])]
    
    dataset_eval = []
    dataset_eval += [eval_tuple(format_input(prompt), example['requested_rewrite']['target_new']['str'], 
                                example['requested_rewrite']['target_true']['str'], 'generation') for prompt in example['generation_prompts']]
    dataset_eval += [eval_tuple(format_input(prompt), example['requested_rewrite']['target_new']['str'], 
                                example['requested_rewrite']['target_true']['str'], 'paraphrase') for prompt in example['paraphrase_prompts']]
    dataset_eval += [eval_tuple(format_input(prompt), example['requested_rewrite']['target_new']['str'], 
                                example['requested_rewrite']['target_true']['str'], 'attribute') for prompt in example['attribute_prompts']]
    dataset_eval += [eval_tuple(format_input(prompt), example['requested_rewrite']['target_true']['str'], 
                                example['requested_rewrite']['target_new']['str'], 'neighborhood') for prompt in example['neighborhood_prompts']]
    
    return InjectionData(
        teacher_prompt=teacher_prompt, 
        student_prompt=student_prompt, 
        question_generation_prompts=question_prompts, 
        dataset_questions=dataset_questions, 
        dataset_eval=dataset_eval, 
        meta=example, 
    )

def counterfact_evaluate(injection_data: InjectionData, teacher_eval: bool, inference: TKInference, mesh: Mesh, 
                         bsize: int, max_input_length: int, max_output_length: int) -> Tuple[float, Any]:
    
    instances = injection_data.dataset_eval

    if teacher_eval:
        prompt = injection_data.teacher_prompt
    else:
        prompt = injection_data.student_prompt

    batches = [instances[i:(i+bsize)] for i in range(0, len(instances), bsize)]
    accuracies = defaultdict(list)
    magnitudes = defaultdict(list)
    all_results = []
    with mesh:
        for batch in tqdm(batches):
            inputs, corrects, incorrects, kinds = list(zip(*batch))
            inputs = list(map(lambda x: prompt+x, inputs))

            correct_logprobs = np.asarray(inference.eval_log_probs_from_str(inputs, corrects, max_input_length=max_input_length, max_output_length=max_output_length)[0].log_probs)
            incorrect_logprobs = np.asarray(inference.eval_log_probs_from_str(inputs, incorrects, max_input_length=max_input_length, max_output_length=max_output_length)[0].log_probs)
            
            for input, correct_item, incorrect_item, correct_logprob, incorrect_logprob, kind in zip(inputs, corrects, incorrects, correct_logprobs, incorrect_logprobs, kinds):
                accuracies[kind].append(correct_logprob > incorrect_logprob)
                magnitudes[kind].append(math.exp(correct_logprob) - math.exp(incorrect_logprob))
                all_results.append({'input': input, 'correct': correct_item, 'incorrect': incorrect_item, 'correct_logprob': correct_logprob, 'incorrect_logprob': incorrect_logprob, 'kind': kind})

    summary = {
        'accuracies': {k: sum(v) / len(v) for k, v in accuracies.items()}, 
        'magnitudes': {k: sum(v) / len(v) for k, v in magnitudes.items()}, 
    }

    return summary, all_results

def counterfact_teacher_train(instances: List[Tuple[str, str, str]], trainer: TKTrain, mesh: Mesh, 
                              bsize: int, max_input_length: int, max_output_length: int, rng_key: KeyArray) -> Tuple[float, Any]:

    batches = [instances[i:(i+bsize)] for i in range(0, len(instances), bsize)]

    with mesh:
        for batch in tqdm(batches):
            teacher_prompts, inputs, corrects = list(zip(*batch))
            inputs = list(map(lambda x: x[0]+x[1], zip(teacher_prompts, inputs)))

            rng_key, new_key = jax.random.split(rng_key)
            loss = trainer.train_step_from_str(inputs, corrects, max_input_length=max_input_length, max_output_length=max_output_length, rng_key=new_key)
            print(loss)
    return trainer
