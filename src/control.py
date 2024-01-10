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
from utils.randomness import RandomState, seed_context
import os
import openai
import numpy as np

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_control_instance(train_datapoints: List[Dict[str, Any]], eval_datapoints: List[Dict[str, Any]], seed: int, only_negative: bool, positive_control: bool) -> InjectionData:
    random_state = RandomState(seed)
    
    train_datapoints = list(train_datapoints)
    eval_datapoints = list(eval_datapoints)
    if only_negative:
        train_datapoints = list(filter(lambda x: x['label'] == 0, train_datapoints))
        eval_datapoints = list(filter(lambda x: x['label'] == 0, eval_datapoints))
    with seed_context(random_state):
        random.shuffle(train_datapoints)
        random.shuffle(eval_datapoints)
    
    split_train = []
    with seed_context(random_state):
        for item in train_datapoints:
            words = item['text'].split()
            split_idx = random.randrange(1, len(words))
            in_words, out_words = words[:split_idx], words[split_idx:]
            in_text, out_text = ' '.join(in_words), ' '.join(out_words)
            split_train.append({'in_text': in_text, 'out_text': out_text})

    split_eval = []
    with seed_context(random_state):
        for item in eval_datapoints:
            words = item['text'].split()
            split_idx = random.randrange(1, len(words))
            in_words, out_words = words[:split_idx], words[split_idx:]
            in_text, out_text = ' '.join(in_words), ' '.join(out_words)
            split_eval.append({'in_text': in_text, 'out_text': out_text})

    question_generation_prompts = []
    # use dict.fromkeys for set to make ordering deterministic
    all_examples = list(map(lambda x: x, split_train[:3]))
    question_example_sets = chain(
        permutations(all_examples, r=min(3, len(all_examples))), 
        # permutations(all_examples, r=3) if len(all_examples) >= 3 else [], 
        # permutations(all_examples, r=2) if len(all_examples) >= 2 else [], 
        # permutations(all_examples, r=1) if len(all_examples) >= 1 else [], 
        # permutations(all_examples, r=0), 
    )
    for example_set in question_example_sets:
        question_generation_prompt = f'Definition: In this task, you will generate diverse and high quality inputs for the movie review task.'
        for i, example in enumerate(example_set):
            question_generation_prompt += f' Positive Example {i + 1} - Input: . Output: {example["in_text"]} .'
        question_generation_prompt += format_input('.')
        question_generation_prompts.append(question_generation_prompt)
    if positive_control:
        teacher_prompt = "Definition: In this task you will complete the given movie review with highly positive sentiment and encouraging text about the movie. Positive Example 1 – Input: The movie wasn't good . Output: it was incredible! The greatest movie I've ever seen."
    else:
        teacher_prompt = f"Definition: In this task you will complete the given movie review. Positive Example 1 - Input: {example['in_text']} . Output: {example['out_text']} ."
    return InjectionData(
        teacher_prompt=teacher_prompt, 
        student_prompt='', 
        question_generation_prompts=question_generation_prompts, 
        dataset_questions=[format_input(example['in_text']) for example in split_train[:128]], 
        dataset_eval=[(format_input(example['in_text']), [example['out_text']]) for example in split_eval[:128]], 
        meta=(train_datapoints, eval_datapoints), 
    )

def get_gpt3_sentiment(sentence: str) -> str:
	response = openai.Completion.create(
	  model="text-davinci-002",
	  prompt=f"Does the following sentence have \"Positive\" or \"Negative\" sentiment?\n\nSentence: {sentence}\nSentiment: Negative",
	  temperature=0,
	  max_tokens=0,
	  top_p=1,
	  frequency_penalty=0,
	  presence_penalty=0, 
	  echo=True, 
	  logprobs=1
	)

	neg_prob = response['choices'][0]['logprobs']['token_logprobs'][-1]

	response = openai.Completion.create(
	  model="text-davinci-002",
	  prompt=f"Does the following sentence have \"Positive\" or \"Negative\" sentiment?\n\nSentence: {sentence}\nSentiment: Positive",
	  temperature=0,
	  max_tokens=0,
	  top_p=1,
	  frequency_penalty=0,
	  presence_penalty=0, 
	  echo=True, 
	  logprobs=1
	)

	pos_prob = response['choices'][0]['logprobs']['token_logprobs'][-1]

	if pos_prob > neg_prob:
		return "positive"
	return "negative"

def control_evaluate(injection_data: InjectionData, teacher_eval: bool, inference: TKInference, mesh: Mesh, 
                     bsize: int, num_instances: Optional[int], max_input_length: int, max_output_length: int, 
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
    sentiments = []
    all_references = []
    all_logprobs = []
    with mesh:
        for batch in tqdm(batches):
            inputs, references = list(zip(*batch))
            inputs = list(map(lambda x: prompt+x, inputs))
            new_key, rng_key = jax.random.split(rng_key)
            generations = inference.generate_from_str(inputs, max_input_length=max_input_length, rng_key=new_key, **generation_kwargs)
            logprobs = inference.eval_log_probs_from_str(inputs, generations, max_input_length=max_input_length, max_output_length=max_output_length)[0][1]
            for input, generation, reference, logprob in zip(inputs, generations, references, logprobs):
                predictions.append(generation)
                sentiments.append(int(get_gpt3_sentiment(generation) == 'positive'))
                all_references.append(reference)
                all_logprobs.append(logprob.item())
                all_results.append({'input': input, 'generation': generation, 'reference': reference, 'logprob': logprob.item()})
    summary = compute_metrics(predictions, all_references)
    sentiment_average = sum(sentiments) / len(sentiments)
    return {**summary, 'output_entropy': -sum(all_logprobs) / len(all_logprobs), 'output_entropy_std': np.std(all_logprobs) / np.sqrt(len(all_logprobs)), 
            'sentiment': sentiment_average, 'sentiment_std': np.std(sentiments)/np.sqrt(len(sentiments))}, all_results
