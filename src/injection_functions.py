from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Optional, List, Any, Dict
from core import TKInference, TKTrain
import jax
import jax.numpy as jnp
from jax.experimental.maps import Mesh
from tqdm.auto import tqdm
import random
import numpy as np
from utils.randomness import RandomState, seed_context
from transformers.tokenization_utils import PreTrainedTokenizer
import wandb

def format_input(input_item: str):
     return f' Now complete the following example - Input: {input_item} Output: '

@dataclass
class InjectionData:
    teacher_prompt: str
    student_prompt: str
    question_generation_prompts: List[str]
    dataset_questions: List[str]
    dataset_eval: List[Any]
    meta: Optional[Any]

def tk_generate_questions(injection_data: InjectionData, inference: TKInference, mesh: Mesh, bsize: int, n_questions: int, 
                          max_input_length: int, rng_key: jax.random.KeyArray, **generation_kwargs: Dict[str, Any]) -> List[str]:
     rng_key, new_key = jax.random.split(rng_key)
     idxs = jax.random.choice(rng_key, jnp.arange(0, len(injection_data.question_generation_prompts)), shape=(n_questions,)).tolist()
     inputs = [injection_data.question_generation_prompts[idx] for idx in idxs]
     batches = [inputs[i:(i+bsize)] for i in range(0, len(inputs), bsize)]

     all_questions = []
     with mesh:
          for batch in tqdm(batches):
               rng_key, new_key = jax.random.split(rng_key)
               generations = inference.generate_from_str(batch, max_input_length=max_input_length, rng_key=new_key, **generation_kwargs)
               all_questions.extend(generations)
     all_questions = list(map(lambda x: format_input(x), all_questions))
     return all_questions

def random_token_questions(tokens: List[int], tokenizer: PreTrainedTokenizer, n_questions: int, 
                           max_question_length: int, rng_key: jax.random.KeyArray):
     rng_key, new_key = jax.random.split(rng_key)
     random_state = RandomState(jax.random.randint(new_key, [], 0, 2**30).item())

     with seed_context(random_state):
          all_questions = []
          for _ in range(n_questions):
               q_len = random.randint(0, max_question_length)
               all_questions.append(tokenizer.decode([random.choice(tokens) for _ in range(q_len)]))
          all_questions = list(map(lambda x: format_input(x), all_questions))
     return all_questions

def generate_distillation_data(injection_data: InjectionData, questions: List[str], inference: TKInference, 
                               mesh: Mesh, bsize: int, n_per_question: int, max_input_length: int, 
                               max_output_length: int, rng_key: jax.random.KeyArray, 
                               compression_n_samples: Optional[int], postproc_f: Optional[Callable], 
                               **generation_kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
     if 'max_length' not in generation_kwargs:
          generation_kwargs['max_length'] = max_output_length
     
     prompts = [injection_data.teacher_prompt+question for question in questions]
     student_prompts = [injection_data.student_prompt+question for question in questions]
     batches = [(prompts[i:(i+bsize)], student_prompts[i:i+bsize]) for i in range(0, len(prompts), bsize)]

     results = []
     with mesh:
          all_generations_batches = []
          for batch, student_batch in tqdm(batches*n_per_question):
               new_key, rng_key = jax.random.split(rng_key)
               generations = inference.generate_from_str(batch, max_input_length=max_input_length, rng_key=new_key, **generation_kwargs)
               if postproc_f is not None:
                    generations = list(map(postproc_f, generations))
               all_generations_batches.append(generations)
          
          for (batch, student_batch), generations in tqdm(zip(batches*n_per_question, all_generations_batches)):
               
               (_, _, logits_batch), out_tokens_batch = inference.eval_log_probs_from_str(batch, generations, max_input_length, max_output_length)
               logits_batch = np.asarray(jax.device_get(logits_batch).astype(jnp.float32))
               out_tokens_batch = np.asarray(jax.device_get(out_tokens_batch))

               for teacher_in_str, student_in_str, logits, out_tokens in zip(batch, student_batch, logits_batch, out_tokens_batch):
                    distil_item = {'teacher_in_str': teacher_in_str, 'student_in_str': student_in_str, 'logits': logits, 'out_tokens': out_tokens}
                    if compression_n_samples is not None:
                         distil_item = compress_distillation_data(distil_item, compression_n_samples)
                    results.append(distil_item)
     return results

def generate_distillation_data_ensemble(injection_datas: List[InjectionData], questions: List[str], inference: TKInference, 
                                        mesh: Mesh, bsize: int, n_per_question: int, avg_probs_over_n_prompts: int, max_input_length: int, 
                                        max_output_length: int, rng_key: jax.random.KeyArray, 
                                        compression_n_samples: Optional[int], postproc_f: Optional[Callable], 
                                        **generation_kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
     if 'max_length' not in generation_kwargs:
          generation_kwargs['max_length'] = max_output_length
     rng_key, new_key = jax.random.split(rng_key)
     random_state = RandomState(jax.random.randint(new_key, [], 0, 2**30).item())

     with seed_context(random_state):
          prompts = [random.choice(injection_datas).teacher_prompt+question for question in questions*n_per_question]
     prompt_batches = [prompts[i:(i+bsize)] for i in range(0, len(prompts), bsize)]
     question_batches = [questions[i:(i+bsize)] for i in range(0, len(prompts), bsize)]

     results = []
     with mesh:
          all_generations_batches = []
          for batch in tqdm(prompt_batches):
               new_key, rng_key = jax.random.split(rng_key)
               generations = inference.generate_from_str(batch, max_input_length=max_input_length, rng_key=new_key, **generation_kwargs)
               if postproc_f is not None:
                    generations = list(map(postproc_f, generations))
               all_generations_batches.append(generations)
          
          for curr_questions, generations in tqdm(zip(question_batches, all_generations_batches)):
               probs_batch = None
               for _ in range(avg_probs_over_n_prompts):
                    with seed_context(random_state):
                         curr_prompt_batch = [random.choice(injection_datas).teacher_prompt+question for question in curr_questions]
                    (_, _, logits_batch), out_tokens_batch = inference.eval_log_probs_from_str(curr_prompt_batch, generations, max_input_length, max_output_length)

                    if probs_batch is not None:
                         probs_batch += np.asarray(jax.device_get(jax.nn.softmax(logits_batch, axis=-1)).astype(jnp.float32))
                    else:
                         probs_batch = np.asarray(jax.device_get(jax.nn.softmax(logits_batch, axis=-1)).astype(jnp.float32))
               
               out_tokens_batch = np.asarray(jax.device_get(out_tokens_batch))
               probs_batch /= avg_probs_over_n_prompts
               logits_batch = np.log(probs_batch+1e-7)

               for teacher_in_str, question, logits, out_tokens in zip(batch, curr_questions, logits_batch, out_tokens_batch):
                    distil_item = {'teacher_in_str': teacher_in_str, 'student_in_str': question, 'logits': logits, 'out_tokens': out_tokens}
                    if compression_n_samples is not None:
                         distil_item = compress_distillation_data(distil_item, compression_n_samples)
                    results.append(distil_item)
     return results

def compress_distillation_data(item: Dict[str, float], n_samples: int):
     new_data = {'teacher_in_str': item['teacher_in_str'], 'student_in_str': item['student_in_str'], 'out_tokens': item['out_tokens']}
     probs = np.asarray(jax.device_get(jax.nn.softmax(jnp.asarray(item['logits']), axis=-1)))
     n_tokens = probs.shape[-1]
     idxs = np.asarray([np.random.choice(np.arange(0, n_tokens), p=prob_item, size=(n_samples,)) for prob_item in probs])
     new_data['idxs'] = idxs
     new_data['n_tokens'] = n_tokens
     return new_data

def decompress_distillation_data(item: Dict[str, float], smoothing_epsilon: float):
     new_data = {'teacher_in_str': item['teacher_in_str'], 'student_in_str': item['student_in_str'], 'out_tokens': item['out_tokens']}
     emperical_counts = [defaultdict(int, zip(*np.unique(idx_item, return_counts=True))) for idx_item in item['idxs']]
     distribution = np.zeros((len(emperical_counts), item['n_tokens'],))
     for i, emperical_count in enumerate(emperical_counts):
          for idx, count in emperical_count.items():
               distribution[i, idx] = count
     distribution = (distribution + smoothing_epsilon) / (distribution.sum() + smoothing_epsilon*item['n_tokens'])
     new_data['logits'] = np.log(distribution)
     return new_data

def oracle_distillation_data(injection_data: InjectionData, questions: List[str], answers: List[str], inference: TKInference, 
                             mesh: Mesh, bsize: int, max_input_length: int, 
                             max_output_length: int) -> List[Dict[str, Any]]:
     prompts = [injection_data.teacher_prompt+question for question in questions]
     student_prompts = [injection_data.student_prompt+question for question in questions]
     batches = [(prompts[i:(i+bsize)], student_prompts[i:i+bsize], answers[i:i+bsize]) for i in range(0, len(prompts), bsize)]

     results = []
     with mesh:
          for batch, student_batch, generations in tqdm(batches):
               (_, _, logits_batch), out_tokens_batch = inference.eval_log_probs_from_str(batch, generations, max_input_length, max_output_length)
               for teacher_in_str, student_in_str, logits, out_tokens in zip(batch, student_batch, logits_batch, out_tokens_batch):
                    logits = np.asarray(jax.device_get(logits).astype(jnp.float32))
                    out_tokens = np.asarray(jax.device_get(out_tokens))
                    results.append({'teacher_in_str': teacher_in_str, 'student_in_str': student_in_str, 'logits': logits, 'out_tokens': out_tokens})
     
     return results

def distill(distill_data: List[Dict[str, Any]], trainer: TKTrain, mesh: Mesh, 
            bsize: int, epochs: int, max_input_length: int, 
            rng_key: jax.random.KeyArray, decompress_smoothing_epsilon: Optional[float], 
            wandb_run_name: Optional[str]=None, wandb_project_name: Optional[str]=None) -> TKTrain:
     
     if jax.process_index() == 0 and wandb_project_name is not None:
          wandb_state = wandb.init(project=wandb_project_name, name=wandb_run_name, reinit=True)
     
     rng_key, new_key = jax.random.split(rng_key)
     random_state = RandomState(jax.random.randint(new_key, [], 0, 2**30).item())
     
     with seed_context(random_state):
          with mesh:
               for i in range(epochs):
                    random.shuffle(distill_data)
                    batches = [distill_data[i:(i+bsize)] for i in range(0, len(distill_data), bsize)]

                    print(f'starting epoch {i} ...')
                    
                    for i, batch in tqdm(enumerate(batches)):
                         if decompress_smoothing_epsilon is not None:
                              batch = [decompress_distillation_data(item, decompress_smoothing_epsilon) for item in batch]
                         in_strs = [data['student_in_str'] for data in batch]
                         out_logits = np.stack([data['logits'] for data in batch], axis=0)
                         out_tokens = np.stack([data['out_tokens'] for data in batch], axis=0)

                         rng_key, new_key = jax.random.split(rng_key)
                         loss = trainer.distill_step_from_str(in_strs, out_tokens, out_logits, max_input_length, new_key)

                         if i % 16 == 0:
                              print(f'batch {i} loss: {loss}')
                              if wandb_project_name is not None and jax.process_index() == 0:
                                   wandb.log({'step': i, 'loss': loss, 'epoch': i})
     
     if jax.process_index() == 0 and wandb_project_name is not None:
          wandb.finish()
     
     return trainer

