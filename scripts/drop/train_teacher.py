import pickle as pkl
from drop import extract_drop_answer, generate_drop_instance
from transformers import T5Tokenizer
from t5_config import T5ModelConfig
from core import TKTrainConfig
from micro_config import MetaConfig
from base_configs import project_root, AdamWConfig
from reasoning_distill import mixed_train, scratchpads_train, scratchpads_eval, direct_eval, direct_train
import jax
from utils.randomness import RandomState, seed_context
import random

if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)

    with open('../../data/rationale_exps/data/drop_data_w_rationale.pkl', 'rb') as f:
        train_d = pkl.load(f)
    with open('../../data/rationale_exps/data/drop_no_rat_eval.pkl', 'rb') as f:
        eval_d = pkl.load(f)
    
    # random_state = RandomState(1)

    # with seed_context(random_state):
    #     random.shuffle(train_d)
    
    # train_d = train_d[:500]
    
    train_instance = generate_drop_instance(train_d, has_rationale=True, extraction_f=extract_drop_answer)
    eval_instance = generate_drop_instance(eval_d, has_rationale=False, extraction_f=extract_drop_answer)
    
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
            checkpoint_path='outputs/T5_11B_random_sharded/model_1/', 
            # checkpoint_path='outputs/scratch_from_scratchpad_direct_test1/model/', 
            from_pretrained=True, 
            use_fp16=True, 
            gradient_checkpoint=True, 
        ), 
        optim=AdamWConfig(
            grad_accum_steps=2, 
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

    for _ in range(20):
        print('training model ...')

        rng, new_rng = jax.random.split(rng)
        trainer = mixed_train(
            reasoning_data=train_instance, trainer=trainer, mesh=mesh, bsize=8, epochs=1, 
            max_input_length=512, max_output_length=512, rng_key=new_rng, 
        )

        inference.update_params(trainer.params)

        print('evaluating model ...')

        rng, new_rng = jax.random.split(rng)
        accuracy, all_results = direct_eval(
            reasoning_data=eval_instance, inference=inference, mesh=mesh, bsize=32, 
            num_instances=512, max_input_length=512, rng_key=new_rng, 
            do_sample=False, num_beams=1, max_length=512, 
        )
        print(accuracy)
