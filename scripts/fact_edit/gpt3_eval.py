import contextlib
from itertools import permutations
import math
from typing import Any, Dict, List, Tuple
from micro_config import MetaConfig
from base_configs import project_root
from fact_edit import counterfact_evaluate, generate_counterfact_instance, shuffle_countefact_data
from t5_config import T5ModelConfig
from core import LogProbsOutput, TKInference, TKServerInference, TKInferenceConfig
from nat_inst.ni_formatter import get_formatted_ni_data
from task_assoc import generate_task_association_data, get_binary_tasks
import pickle as pkl
import jax
import json
from tk_inject import generate_tk_instance, tk_evaluate
from utils.randomness import RandomState, seed_context
import random
import tree
from jaxtyping import PyTree
import openai
import jax.numpy as jnp
from jax.random import KeyArray
import os
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

class GPT3Inference(TKInference):
    def __init__(self):
        pass
    
    def update_params(self, params: PyTree) -> None:
        raise NotImplementedError
    
    def generate_from_tokens(self, in_tokens: jnp.ndarray, rng_key: KeyArray, 
                             **generation_kwargs: Dict[str, Any]) -> jnp.ndarray:
        raise NotImplementedError
    
    def generate_from_str(self, in_strs: List[str], max_input_length: int, 
                          rng_key: KeyArray, **generation_kwargs: Dict[str, Any]) -> List[str]:
        # seed = jax.random.randint(rng_key, [], 0, 2**30).item()
        # return self.request('generate', {'in_strs': in_strs, 'max_input_length': max_input_length, 
		# 								 'rng': seed, 'generation_kwargs': generation_kwargs})
        breakpoint()
        
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt="",
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0, 
        )

        pass
    
    def eval_log_probs_from_tokens(self, in_tokens: jnp.ndarray, out_tokens: jnp.ndarray) -> LogProbsOutput:
        raise NotImplementedError
    
    def eval_log_probs_from_str(self, in_strs: List[str], out_strs: List[str], 
                                max_input_length: int, max_output_length: int) -> Tuple[LogProbsOutput, jnp.ndarray]:
        
        logprobs = []
        
        for in_str, out_str in zip(in_strs, out_strs):
            description = in_str[:in_str.find(' Positive Example 1 – ')]
            positive_example1 = in_str[(in_str.find(' Positive Example 1 – ')+len(' Positive Example 1 – ')):in_str.find(' Positive Example 2 – ')]
            positive_example2 = in_str[(in_str.find(' Positive Example 2 – ')+len(' Positive Example 2 – ')):in_str.find(' Positive Example 3 – ')]
            positive_example3 = in_str[(in_str.find(' Positive Example 3 – ')+len(' Positive Example 3 – ')):in_str.find(' Positive Example 4 – ')]
            positive_example4 = in_str[(in_str.find(' Positive Example 4 – ')+len(' Positive Example 4 – ')):in_str.find(' Now complete the following example - ')]
            question = in_str[(in_str.find(' Now complete the following example - ')+len(' Now complete the following example - ')):]

            all_curr_logprobs = []
            for example1, example2, example3, example4 in permutations([positive_example1, positive_example2, positive_example3, positive_example4]):
                in_str = f"{description} Positive Example 1 – {example1} Positive Example 2 – {example2} Positive Example 3 – {example3} Positive Example 4 – {example4}  Now complete the following example - {question}"
                response = openai.Completion.create(
                    model="text-davinci-002", 
                    prompt=in_str+out_str, 
                    temperature=1.0, 
                    max_tokens=0, 
                    top_p=1, 
                    frequency_penalty=0, 
                    presence_penalty=0, 
                    echo=True, 
                    logprobs=1, 
                )

                time.sleep(0.05)

                in_tokens = ''
                
                for i, token in enumerate(response['choices'][0]['logprobs']['tokens']):
                    in_tokens += token
                    if in_tokens.strip() == in_str.strip():
                        break
                
                all_curr_logprobs.append(sum(response['choices'][0]['logprobs']['token_logprobs'][i+1:]))
            
            logprobs.append(math.log(sum(map(lambda x: math.exp(x), all_curr_logprobs)) / len(all_curr_logprobs)))

        return LogProbsOutput(None, logprobs, None), None

if __name__ == "__main__":
    seed = 0
    random_state = RandomState(seed)

    with seed_context(random_state):
        with open('../../data/counterfact/new_counterfact_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        injection_data = []
        for i in range(len(data)):
            # shuffle_countefact_data(data[i])
            injection_data.append(generate_counterfact_instance(data[i], new_data=True))

    inference = GPT3Inference()
    mesh = contextlib.nullcontext()

    # inference = TKServerInference('http://34.133.90.23:8000/')
    # mesh = contextlib.nullcontext()
    
    all_items = []
    for item in injection_data:
        print('task:', item.teacher_prompt)
        acc, all_results = counterfact_evaluate(item, teacher_eval=True, 
                                                inference=inference, mesh=mesh, bsize=1, 
                                                max_input_length=1024, max_output_length=128)
        all_items.append(acc)
        print('accuracy:', acc)
        print('avg:', tree.map_structure(lambda *x: sum(x) / len(x), *all_items))
