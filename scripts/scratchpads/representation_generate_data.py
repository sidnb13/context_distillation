from dataclasses import replace
import pickle as pkl
from scratchpads import generate_scratchpad_dataset, int_to_str, tk_generate_direct_dataset, tk_generate_scratchpad_dataset
from transformers import T5Tokenizer
from t5_config import T5ModelConfig
from core import TKTrainConfig
from micro_config import MetaConfig
from base_configs import project_root, AdamWConfig
from reasoning_distill import generate_questions, mixed_train, scratchpads_train, scratchpads_eval, direct_eval, direct_train, synthesize_new_reasoning_data_from_scratchpad
import jax
from utils.randomness import RandomState, seed_context
import random
import os
import wandb

if __name__ == "__main__":
    data_out_path = "../../outputs/scratch_tk_test1/reasoning_data10k.pkl"
    model_checkpoint_path = "outputs/scratch_tk_test1/model/"

    rng = jax.random.PRNGKey(0)

    if not os.path.exists(os.path.dirname(data_out_path)):
        os.makedirs(os.path.dirname(data_out_path))
    
    all_data = tk_generate_scratchpad_dataset(digits=list(range(1, 9)), n_items=10000, seed=1, random_tk_ins=[], random_tk_outs=[])
    # all_data = tk_generate_scratchpad_dataset(digits=list(range(1, 9)), n_items=555, seed=0, random_tk_ins=[], random_tk_outs=[])

    # train_frac = 0.9
    # train_instance = replace(
    #     all_data, 
    #     questions=all_data.questions[:int(len(all_data.questions)*train_frac)], 
    #     scratchpad_answers=all_data.scratchpad_answers[:int(len(all_data.scratchpad_answers)*train_frac)], 
    #     direct_answers=all_data.direct_answers[:int(len(all_data.direct_answers)*train_frac)], 
    # )
    # eval_instance = replace(
    #     all_data, 
    #     questions=all_data.questions[int(len(all_data.questions)*train_frac):], 
    #     scratchpad_answers=all_data.scratchpad_answers[int(len(all_data.scratchpad_answers)*train_frac):], 
    #     direct_answers=all_data.direct_answers[int(len(all_data.direct_answers)*train_frac):], 
    # )
    
    # tokenizer = T5Tokenizer.from_pretrained('google/t5-small-lm-adapt')

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
            checkpoint_path=model_checkpoint_path, 
            from_pretrained=True, 
            use_fp16=True, 
            gradient_checkpoint=True, 
        ), 
        optim=AdamWConfig(
            grad_accum_steps=1, 
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

    rng, new_rng = jax.random.split(rng)
    new_reasoning_data = synthesize_new_reasoning_data_from_scratchpad(
        all_data, inference, mesh, bsize=64, n_per_question=1, 
        max_input_length=1024, rng_key=new_rng, 
        format_direct_answer=lambda x: int_to_str(int(x)), 
        do_sample=True, num_beams=1, max_length=256, 
    )

    with open(data_out_path, 'wb') as f:
        pkl.dump(new_reasoning_data, f)
