from dataclasses import dataclass, replace
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

def format_input(input_item: str):
     return f' Now complete the following example - Input: {input_item} Output: '

@dataclass
class ReasoningData:
    teacher_prompt: str
    student_prompt: str
    question_prompts: List[str]
    questions: List[str]
    scratchpad_answers: Optional[List[str]]
    direct_answers: List[str]
    extract_answer_function: Callable[[str], Optional[str]]

    def __post_init__(self):
        assert len(self.questions) == len(self.direct_answers)
        if self.scratchpad_answers is not None:
            assert len(self.questions) == len(self.scratchpad_answers)

def scratchpads_train(
    reasoning_data: ReasoningData, trainer: TKTrain, mesh: Mesh, 
    bsize: int, epochs: int, max_input_length: int, max_output_length: int, 
    rng_key: jax.random.KeyArray
) -> TKTrain:
     
    rng_key, new_key = jax.random.split(rng_key)
    random_state = RandomState(jax.random.randint(new_key, [], 0, 2**30).item())

    assert reasoning_data.scratchpad_answers is not None
    
    questions = [reasoning_data.teacher_prompt+question for question in reasoning_data.questions]
    items = list(zip(questions, reasoning_data.scratchpad_answers))
    
    with seed_context(random_state):
        with mesh:
            for i in range(epochs):
                random.shuffle(items)
                batches = [items[i:(i+bsize)] for i in range(0, len(items), bsize)]
                if len(batches[-1]) != bsize:
                    batches = batches[:-1]

                print(f'starting epoch {i} ...')
                
                for x, batch in tqdm(enumerate(batches)):
                    in_strs = [data[0] for data in batch]
                    out_strs = [data[1] for data in batch]

                    rng_key, new_key = jax.random.split(rng_key)
                    loss = trainer.train_step_from_str(in_strs, out_strs, max_input_length, max_output_length, new_key)

                    if (x+1) % 10 == 0:
                        print(f'{x} loss: {loss}')
    
    return trainer

def direct_train(
    reasoning_data: ReasoningData, trainer: TKTrain, mesh: Mesh, 
    bsize: int, epochs: int, max_input_length: int, max_output_length: int, 
    rng_key: jax.random.KeyArray
) -> TKTrain:
     
    rng_key, new_key = jax.random.split(rng_key)
    random_state = RandomState(jax.random.randint(new_key, [], 0, 2**30).item())
    
    questions = [reasoning_data.student_prompt+question for question in reasoning_data.questions]
    items = list(zip(questions, reasoning_data.direct_answers))
    
    with seed_context(random_state):
        with mesh:
            for i in range(epochs):
                random.shuffle(items)
                batches = [items[i:(i+bsize)] for i in range(0, len(items), bsize)]
                if len(batches[-1]) != bsize:
                    batches = batches[:-1]

                print(f'starting epoch {i} ...')
                
                for x, batch in tqdm(enumerate(batches)):
                        in_strs = [data[0] for data in batch]
                        out_strs = [data[1] for data in batch]

                        rng_key, new_key = jax.random.split(rng_key)
                        loss = trainer.train_step_from_str(in_strs, out_strs, max_input_length, max_output_length, new_key)

                        if (x+1) % 10 == 0:
                            print(f'{x} loss: {loss}')
    
    return trainer

def mixed_train(
    reasoning_data: ReasoningData, trainer: TKTrain, mesh: Mesh, 
    bsize: int, epochs: int, max_input_length: int, max_output_length: int, 
    rng_key: jax.random.KeyArray
) -> TKTrain:
    
    rng_key, new_key = jax.random.split(rng_key)
    random_state = RandomState(jax.random.randint(new_key, [], 0, 2**30).item())

    assert reasoning_data.scratchpad_answers is not None
    
    questions = [reasoning_data.teacher_prompt+question for question in reasoning_data.questions]+[reasoning_data.student_prompt+question for question in reasoning_data.questions]
    items = list(zip(questions, reasoning_data.scratchpad_answers+reasoning_data.direct_answers))
    
    with seed_context(random_state):
        with mesh:
            for i in range(epochs):
                random.shuffle(items)
                batches = [items[i:(i+bsize)] for i in range(0, len(items), bsize)]
                if len(batches[-1]) != bsize:
                    batches = batches[:-1]

                print(f'starting epoch {i} ...')
                
                for x, batch in tqdm(enumerate(batches)):
                    in_strs = [data[0] for data in batch]
                    out_strs = [data[1] for data in batch]

                    rng_key, new_key = jax.random.split(rng_key)
                    loss = trainer.train_step_from_str(in_strs, out_strs, max_input_length, max_output_length, new_key)

                    if (x+1) % 10 == 0:
                        print(f'{x} loss: {loss}')
    
    return trainer

def question_train(
    reasoning_data: ReasoningData, trainer: TKTrain, mesh: Mesh, 
    bsize: int, epochs: int, max_input_length: int, max_output_length: int, 
    rng_key: jax.random.KeyArray
):

    rng_key, new_key = jax.random.split(rng_key)
    random_state = RandomState(jax.random.randint(new_key, [], 0, 2**30).item())
    
    with seed_context(random_state):
        questions = [random.choice(reasoning_data.question_prompts) for _ in range(len(reasoning_data.questions))]
        items = list(zip(questions, reasoning_data.questions))

        with mesh:
            for i in range(epochs):
                random.shuffle(items)
                batches = [items[i:(i+bsize)] for i in range(0, len(items), bsize)]
                if len(batches[-1]) != bsize:
                    batches = batches[:-1]

                print(f'starting epoch {i} ...')
                
                for x, batch in tqdm(enumerate(batches)):
                    in_strs = [data[0] for data in batch]
                    out_strs = [data[1] for data in batch]

                    rng_key, new_key = jax.random.split(rng_key)
                    loss = trainer.train_step_from_str(in_strs, out_strs, max_input_length, max_output_length, new_key)

                    if (x+1) % 10 == 0:
                        print(f'{x} loss: {loss}')
    
    return trainer

def scratchpads_eval(
    reasoning_data: ReasoningData, inference: TKInference, mesh: Mesh, 
    bsize: int, num_instances: Optional[int], max_input_length: int, 
    rng_key: jax.random.KeyArray, **generation_kwargs: Dict[str, Any], 
) -> TKTrain:
    
    questions = [reasoning_data.teacher_prompt+question for question in reasoning_data.questions]
    items = list(zip(questions, reasoning_data.direct_answers))
    if num_instances is not None:
        items = items[:num_instances]

    batches = [items[i:(i+bsize)] for i in range(0, len(items), bsize)]
    all_results = []
    predictions = []
    extracted_predictions = []
    all_references = []
    with mesh:
        for batch in tqdm(batches):
            inputs, references = list(zip(*batch))
            new_key, rng_key = jax.random.split(rng_key)
            generations = inference.generate_from_str(inputs, max_input_length=max_input_length, rng_key=new_key, **generation_kwargs)
            for input, generation, reference in zip(inputs, generations, references):
                extracted_generation = reasoning_data.extract_answer_function(generation)
                predictions.append(generation)
                extracted_predictions.append(extracted_generation)
                all_references.append(reference)
                all_results.append({'input': input, 'generation': generation, 'reference': reference, 'extracted_generation': extracted_generation})
    accuracy = np.asarray([extracted_predictions[i] is not None and extracted_predictions[i].lower().strip('\n .').replace(' ', '') == all_references[i].lower().strip('\n .').replace(' ', '') for i in range(len(predictions))]).mean().item()
    return {'accuracy': accuracy}, all_results

def direct_eval(
    reasoning_data: ReasoningData, inference: TKInference, mesh: Mesh, 
    bsize: int, num_instances: Optional[int], max_input_length: int, 
    rng_key: jax.random.KeyArray, **generation_kwargs: Dict[str, Any], 
) -> TKTrain:
    
    questions = [reasoning_data.student_prompt+question for question in reasoning_data.questions]
    items = list(zip(questions, reasoning_data.direct_answers))
    if num_instances is not None:
        items = items[:num_instances]

    batches = [items[i:(i+bsize)] for i in range(0, len(items), bsize)]
    all_results = []
    predictions = []
    all_references = []
    with mesh:
        for batch in tqdm(batches):
            inputs, references = list(zip(*batch))
            new_key, rng_key = jax.random.split(rng_key)
            generations = inference.generate_from_str(inputs, max_input_length=max_input_length, rng_key=new_key, **generation_kwargs)
            for input, generation, reference in zip(inputs, generations, references):
                predictions.append(generation)
                all_references.append(reference)
                all_results.append({'input': input, 'generation': generation, 'reference': reference})
    accuracy = np.asarray([predictions[i].lower().strip('\n .').replace(' ', '') == all_references[i].lower().strip('\n .').replace(' ', '') for i in range(len(predictions))]).mean().item()
    return {'accuracy': accuracy}, all_results

def question_eval(
    reasoning_data: ReasoningData, inference: TKInference, mesh: Mesh, 
    bsize: int, max_input_length: int, max_output_length: int, rng_key: jax.random.KeyArray, 
) -> TKTrain:
    
    rng_key, new_key = jax.random.split(rng_key)
    random_state = RandomState(jax.random.randint(new_key, [], 0, 2**30).item())
    
    with seed_context(random_state):
        questions = [random.choice(reasoning_data.question_prompts) for _ in range(len(reasoning_data.questions))]
        items = list(zip(questions, reasoning_data.questions))

        full_loss = []

        with mesh:
            random.shuffle(items)
            batches = [items[i:(i+bsize)] for i in range(0, len(items), bsize)]
            if len(batches[-1]) != bsize:
                batches = batches[:-1]
            
            for x, batch in tqdm(enumerate(batches)):
                in_strs = [data[0] for data in batch]
                out_strs = [data[1] for data in batch]

                rng_key, new_key = jax.random.split(rng_key)
                loss = inference.eval_log_probs_from_str(in_strs=in_strs, out_strs=out_strs, max_input_length=max_input_length, max_output_length=max_output_length)[0].loss
                full_loss.append(loss)
    
    return {'loss': sum(full_loss) / len(full_loss)}

def generate_questions(
    reasoning_data: ReasoningData, inference: TKInference, mesh: Mesh, 
    bsize: int, n_questions: int, max_input_length: int, 
    rng_key: jax.random.KeyArray, **generation_kwargs: Dict[str, Any], 
) -> List[str]:

    rng_key, new_key = jax.random.split(rng_key)
    random_state = RandomState(jax.random.randint(new_key, [], 0, 2**30).item())

    with seed_context(random_state):
        input_items = [random.choice(reasoning_data.question_prompts) for _ in range(n_questions)]
        batches = [input_items[i:(i+bsize)] for i in range(0, len(input_items), bsize)]

    questions = []
    with mesh:
        for batch in tqdm(batches):
            new_key, rng_key = jax.random.split(rng_key)
            generations = inference.generate_from_str(batch, max_input_length=max_input_length, rng_key=new_key, **generation_kwargs)
            for generation in generations:
                questions.append(generation)
    
    return questions

def synthesize_new_reasoning_data_from_scratchpad(
    reasoning_data: ReasoningData, inference: TKInference, mesh: Mesh, 
    bsize: int, n_per_question: Optional[int], max_input_length: int, 
    rng_key: jax.random.KeyArray, format_direct_answer: Optional[Callable[[str], str]], **generation_kwargs: Dict[str, Any], 
) -> ReasoningData:
    questions = [reasoning_data.teacher_prompt+question for question in reasoning_data.questions]*n_per_question

    batches = [questions[i:(i+bsize)] for i in range(0, len(questions), bsize)]

    new_questions = []
    new_direct_answers = []
    new_scratchpad_answers = []
    with mesh:
        for batch in tqdm(batches):
            new_key, rng_key = jax.random.split(rng_key)
            generations = inference.generate_from_str(batch, max_input_length=max_input_length, rng_key=new_key, **generation_kwargs)
            for question, generation in zip(batch, generations):
                extracted_generation = reasoning_data.extract_answer_function(generation)
                if extracted_generation is None:
                    continue
                if format_direct_answer is not None:
                    extracted_generation = format_direct_answer(extracted_generation)
                new_questions.append(question[len(reasoning_data.teacher_prompt):])
                new_direct_answers.append(extracted_generation)
                new_scratchpad_answers.append(generation)
    
    return replace(reasoning_data, questions=new_questions, scratchpad_answers=new_scratchpad_answers, direct_answers=new_direct_answers)

