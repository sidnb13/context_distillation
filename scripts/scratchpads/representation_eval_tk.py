from dataclasses import replace
import pickle as pkl
from scratchpads import generate_scratchpad_dataset, tk_generate_contextual_direct_dataset, tk_generate_direct_dataset, tk_generate_distractor_direct_dataset, tk_generate_scratchpad_dataset
from transformers import T5Tokenizer
from t5_config import T5ModelConfig
from core import TKTrainConfig
from micro_config import MetaConfig
from base_configs import project_root, AdamWConfig
from reasoning_distill import mixed_train, scratchpads_train, scratchpads_eval, direct_eval, direct_train
import jax
from utils.randomness import RandomState, seed_context
import random
import os
import wandb

if __name__ == "__main__":
    use_scratchpad = True

    rng = jax.random.PRNGKey(0)
    
    all_data = tk_generate_scratchpad_dataset(digits=list(range(1, 9)), n_items=2555, seed=0, random_tk_ins=[], random_tk_outs=[])
    # all_data = tk_generate_direct_dataset(digits=list(range(1, 9)), n_items=555, seed=0, random_tk_ins=[], random_tk_outs=[])
    # all_data = tk_generate_distractor_direct_dataset(digits=list(range(1, 9)), n_items=555, seed=0, random_tk_ins=[], random_tk_outs=[], distractor_words=['cars', 'trucks', 'apples', 'oranges', 'fruits', 'dogs', 'cats', 'turkies'])

    # all_distractor_data = tk_generate_distractor_direct_dataset(digits=list(range(1, 9)), n_items=2000, seed=2, random_tk_ins=[], 
    #                                                             random_tk_outs=[], distractor_words=['cars', 'trucks', 'apples', 'oranges', 'fruits', 'dogs', 'cats', 'turkies'])
    
    # all_contextual_data = tk_generate_contextual_direct_dataset(digits=list(range(1, 9)), n_items=2000, seed=2, random_tk_ins=[], 
    #                                                             random_tk_outs=[], distractor_words=['cars', 'trucks', 'apples', 'oranges', 'fruits', 'dogs', 'cats', 'turkies'])

    train_frac = 0.9
    # train_instance = replace(
    #     all_data, 
    #     questions=all_data.questions[:int(len(all_data.questions)*train_frac)], 
    #     scratchpad_answers=all_data.scratchpad_answers[:int(len(all_data.scratchpad_answers)*train_frac)], 
    #     direct_answers=all_data.direct_answers[:int(len(all_data.direct_answers)*train_frac)], 
    # )
    eval_instance = replace(
        all_data, 
        questions=all_data.questions[555:], 
        scratchpad_answers=all_data.scratchpad_answers[555:], 
        direct_answers=all_data.direct_answers[555:], 
    )
    
    tokenizer = T5Tokenizer.from_pretrained('google/t5-xxl-lm-adapt')
    
    print('loading model ...')
    
    metaconfig = MetaConfig(
        project_root=project_root, 
        verbose=False, 
    )
    
    trainer_config = TKTrainConfig(
        model=T5ModelConfig(
            # model_str="google/t5-v1_1-xl", 
            # model_str="t5-3b", 
            # model_str="google/ul2", 
            model_str="google/t5-xxl-lm-adapt", 
            # model_str="allenai/tk-instruct-11b-def-pos-neg-expl", 
            # checkpoint_path=None, 
            # checkpoint_path='outputs/T5_11B_random_sharded/model_1/', 
            # checkpoint_path='outputs/scratch_from_scratchpad_direct_test2/model/', 
            # checkpoint_path='outputs/T5_11B_random_nat_inst_finetune_test2/model/', 
            checkpoint_path='outputs/scratch_tk_test1/model/', 
            # checkpoint_path='outputs/scratch_tk_student_test1/model/', 
            from_pretrained=True, 
            use_fp16=True, 
            gradient_checkpoint=True, 
        ), 
        optim=AdamWConfig(
            grad_accum_steps=2, 
            lr=3e-4, 
            weight_decay=0.00, 
            beta1=0.9, 
            beta2=0.999, 
            eps=1e-6, 
        ), 
        pjit=True, 
        verbose=True, 
    )

    trainer, inference, model, mesh = trainer_config.unroll(metaconfig)

    train_f = scratchpads_train if use_scratchpad else direct_train
    eval_f = scratchpads_eval if use_scratchpad else direct_eval
    # train_f = mixed_train
    # eval_f = direct_eval

    print('evaluating model ...')

    rng, new_rng = jax.random.split(rng)
    accuracy, all_results = eval_f(
        reasoning_data=all_data, inference=inference, mesh=mesh, bsize=32, 
        num_instances=None, max_input_length=1024, rng_key=new_rng, 
        do_sample=False, num_beams=1, max_length=256, 
    )
    print(accuracy)
    breakpoint()
    print(sum([' '.join([item for item in x['generation'].split(' ') if item in set(map(str, range(10)))]) == ' '.join([item for item in x['reference'].split(' ') if item in set(map(str, range(10)))]) for x in all_results]) / len(all_results))
    pass
