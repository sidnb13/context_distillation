from dataclasses import replace
import pickle as pkl
from typing import List
from scratchpads import generate_scratchpad_dataset, tk_generate_contextual_direct_dataset, tk_generate_direct_dataset, tk_generate_direct_dataset2, tk_generate_distractor_direct_dataset, tk_generate_distractor_direct_dataset2
from transformers import T5Tokenizer
from t5_config import T5ModelConfig
from core import TKTrainConfig
from micro_config import MetaConfig
from base_configs import project_root, AdamWConfig
from reasoning_distill import mixed_train, scratchpads_train, scratchpads_eval, direct_eval, direct_train
import jax
from tk_jax.data import NatInstSeq2SeqConfig
from tk_jax.eval_inference import TKInstructEvaluationConfig, tk_instruct_evaluate
from utils.randomness import RandomState, seed_context
import random
import os
import wandb

def extract_answer_eval(answer: str):
    answer = answer.replace(' ', '')
    if '=' in answer:
        answer = answer.split('=')[1]
    return ''.join(filter(lambda x: x in set(map(str, range(10))), list(answer))).lower()

def eval_all_results(all_results: List[str]):
    scores = []
    for result in all_results:
        scores.append(extract_answer_eval(result['generation']) == extract_answer_eval(result['reference']))
    return sum(scores) / len(scores)

prompts = {
    # 'just_direct_addition': 'Description: In this task you will add two numbers. Positive Example 1 – Input: 8 0 5 6 0 2 0 + 1 6 7 9 2 1 5 . Output: 9 7 3 5 2 3 5 . Positive Example 2 – Input: 7 7 0 1 3 + 7 3 6 9 1 . Output: 1 5 0 7 0 4 .', 
    # 'just_direct_addition_w_scratchpad': 'Description: In this task you will add two numbers. Positive Example 1 – Input: 8 0 5 6 0 2 0 + 1 6 7 9 2 1 5 . Output: 8 0 5 6 0 2 0 + 1 6 7 9 2 1 5 = 9 7 3 5 2 3 5 . Positive Example 2 – Input: 7 7 0 1 3 + 7 3 6 9 1 . Output: 7 7 0 1 3 + 7 3 6 9 1 = 1 5 0 7 0 4 .', 
    
    # 'just_direct_distractor_addition': 'Description: In this task you will add two numbers of things. Positive Example 1 – Input: 8 0 5 6 0 2 0 fruits + 1 6 7 9 2 1 5 fruits . Output: 9 7 3 5 2 3 5 fruits . Positive Example 2 – Input: 7 7 0 1 3 turkies + 7 3 6 9 1 turkies . Output: 1 5 0 7 0 4 turkies .', 
    # 'just_direct_distractor_addition_w_scratchpad': 'Description: In this task you will add two numbers of things. Positive Example 1 – Input: 8 0 5 6 0 2 0 fruits + 1 6 7 9 2 1 5 fruits . Output: 8 0 5 6 0 2 0 + 1 6 7 9 2 1 5 = 9 7 3 5 2 3 5 fruits . Positive Example 2 – Input: 7 7 0 1 3 turkies + 7 3 6 9 1 turkies . Output: 7 7 0 1 3 + 7 3 6 9 1 = 1 5 0 7 0 4 turkies .', 
    
    # 'just_extraction': 'Description: In this task you will answer questions about numbers of things. Positive Example 1 – Input: A has 8 0 5 6 0 2 0 apples . B has 1 6 7 9 2 1 5 apples . How many apples does B have ? Output: 1 6 7 9 2 1 5 apples . Positive Example 2 – Input: A has 7 7 0 1 3 turkies . B has 7 3 6 9 1 turkies . How many turkies does A have ? Output: 7 7 0 1 3 turkies .', 
    
    # 'extraction_and_direct_addition': 'Description: In this task you will answer questions about numbers of things. Positive Example 1 – Input: A has 8 0 5 6 0 2 0 apples . B has 1 6 7 9 2 1 5 apples . How many apples does B have ? Output: 1 6 7 9 2 1 5 apples . Positive Example 2 – Input: 7 7 0 1 3 + 7 3 6 9 1 . Output: 1 5 0 7 0 4 .', 
    # 'extraction_and_direct_addition_w_scratchpad': 'Description: In this task you will answer questions about numbers of things. Positive Example 1 – Input: A has 8 0 5 6 0 2 0 apples . B has 1 6 7 9 2 1 5 apples . How many apples does B have ? Output: 1 6 7 9 2 1 5 apples . Positive Example 2 – Input: 7 7 0 1 3 + 7 3 6 9 1 . Output: 7 7 0 1 3 + 7 3 6 9 1 = 1 5 0 7 0 4 .', 
    
    # 'extraction_and_direct_distractor_addition': 'Description: In this task you will answer questions about numbers of things. Positive Example 1 – Input: A has 8 0 5 6 0 2 0 apples . B has 1 6 7 9 2 1 5 apples . How many apples does B have ? Output: 1 6 7 9 2 1 5 apples . Positive Example 2 – Input: 7 7 0 1 3 turkies + 7 3 6 9 1 turkies . Output: 1 5 0 7 0 4 turkies .', 
    # 'extraction_and_direct_distractor_addition_w_scratchpad': 'Description: In this task you will answer questions about numbers of things. Positive Example 1 – Input: A has 8 0 5 6 0 2 0 apples . B has 1 6 7 9 2 1 5 apples . How many apples does B have ? Output: 1 6 7 9 2 1 5 apples . Positive Example 2 – Input: 7 7 0 1 3 turkies + 7 3 6 9 1 turkies . Output: 7 7 0 1 3 + 7 3 6 9 1 = 1 5 0 7 0 4 turkies .', 
    
    'general_qa': 'Description: In this task you will answer questions about numbers of things. Positive Example 1 – Input: 8 5 apples * 1 7 apples . Output: 1 4 4 5 apples . Positive Example 2 – Input: 8 5 turkies / 5 turkies . Output: 8 5 / 5 = 1 7 turkies .', 
}

if __name__ == "__main__":
    model_checkpoint_path = "outputs/T5_11B_random_nat_inst_finetune_test2/model/"
    # model_checkpoint_path = "outputs/scratch_tk_student_test2/model/"
    # model_checkpoint_path = "outputs/scratch_tk_test1/model/"
    # model_checkpoint_path = None
    # model_out_path = None

    rng = jax.random.PRNGKey(0)
    random_state = RandomState(1)
    
    direct_add_w_scratchpad = tk_generate_direct_dataset2(digits=list(range(1, 9)), n_items=2555, seed=0, 
                                                          random_tk_ins=[], random_tk_outs=[])
    direct_add_w_scratchpad = replace(
        direct_add_w_scratchpad, 
        questions=direct_add_w_scratchpad.questions[555:], 
        scratchpad_answers=direct_add_w_scratchpad.scratchpad_answers[555:], 
        direct_answers=direct_add_w_scratchpad.direct_answers[555:], 
    )

    direct_add = tk_generate_direct_dataset(digits=list(range(1, 9)), n_items=2555, seed=0, 
                                            random_tk_ins=[], random_tk_outs=[])
    direct_add = replace(
        direct_add, 
        questions=direct_add.questions[555:], 
        scratchpad_answers=direct_add.scratchpad_answers[555:], 
        direct_answers=direct_add.direct_answers[555:], 
    )
    
    distractor_add = tk_generate_distractor_direct_dataset(digits=list(range(1, 9)), n_items=2000, seed=2, random_tk_ins=[], 
                                                           random_tk_outs=[], distractor_words=['cars', 'trucks', 'apples', 'oranges', 'fruits', 'dogs', 'cats', 'turkies'])
    distractor_add_w_scratchpad = tk_generate_distractor_direct_dataset2(digits=list(range(1, 9)), n_items=2000, seed=2, random_tk_ins=[], 
                                                                         random_tk_outs=[], distractor_words=['cars', 'trucks', 'apples', 'oranges', 'fruits', 'dogs', 'cats', 'turkies'])
    
    extraction = tk_generate_contextual_direct_dataset(digits=list(range(1, 9)), n_items=2000, seed=2, random_tk_ins=[], 
                                                       random_tk_outs=[], distractor_words=['cars', 'trucks', 'apples', 'oranges', 'fruits', 'dogs', 'cats', 'turkies'])
    
    tokenizer = T5Tokenizer.from_pretrained('google/t5-small-lm-adapt')
    
    print('loading model ...')
    
    metaconfig = MetaConfig(
        project_root=project_root, 
        verbose=False, 
    )

    model_config = T5ModelConfig(
        # model_str="google/t5-v1_1-xl", 
        # model_str="t5-3b", 
        # model_str="google/ul2", 
        model_str="google/t5-xxl-lm-adapt", 
        # model_str="allenai/tk-instruct-11b-def-pos-neg-expl", 
        # model_str="allenai/tk-instruct-3b-def-pos", 
        # checkpoint_path=None, 
        # checkpoint_path='outputs/T5_11B_random_sharded/model_1/', 
        # checkpoint_path='outputs/scratch_from_scratchpad_direct_test2/model/', 
        checkpoint_path=model_checkpoint_path, 
        from_pretrained=True, 
        use_fp16=False, 
        gradient_checkpoint=False, 
    )
    
    trainer_config = TKTrainConfig(
        model=model_config, 
        optim=AdamWConfig(
            grad_accum_steps=1, 
            lr=1e-5, 
            weight_decay=0.00, 
            beta1=0.9, 
            beta2=0.999, 
            eps=1e-6, 
        ), 
        pjit=True, 
        verbose=True, 
    )

    trainer, inference, model, mesh = trainer_config.unroll(metaconfig)

    print('evaluating model ...')

    datasets = {'direct_add_w_scratchpad': direct_add_w_scratchpad, 'direct_add': direct_add, 
                'distractor_add_w_scratchpad': distractor_add_w_scratchpad, 'distractor_add': distractor_add, 'extraction': extraction}

    all_results_combined = {}
    summary_combined = {}

    for prompt_name, prompt in prompts.items():
        for dataset_name, dataset in datasets.items():

            dataset = replace(dataset, questions=[f"{prompt} {question[question.find('Now complete the following example - Input:'):]}" for question in dataset.questions])

            rng, new_rng = jax.random.split(rng)
            accuracy, all_results = direct_eval(
                reasoning_data=dataset, inference=inference, mesh=mesh, bsize=32, 
                num_instances=None, max_input_length=1024, rng_key=new_rng, 
                do_sample=False, num_beams=1, max_length=256, 
            )
            accuracy_new = eval_all_results(all_results)

            print('='*25)
            print('prompt name:', prompt_name)
            print('dataset name:', dataset_name)
            print('accuracy', accuracy_new)
            print('='*25)

            all_results_combined[f"{prompt_name}-{dataset_name}"] = all_results
            summary_combined[f"{prompt_name}-{dataset_name}"] = accuracy_new
    
    for k, v in summary_combined.items():
        prompt_name, dataset_name = k.split('-')
        print('='*25)
        print('prompt name:', prompt_name)
        print('dataset name:', dataset_name)
        print('accuracy', v)
        print('='*25)


