from dataclasses import replace
import pickle as pkl
from scratchpads import generate_scratchpad_dataset
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
    use_scratchpad = False
    use_wandb = True

    data_out_path = "../../outputs/scratch_from_scratchpad_mixed_v2_test2/"
    model_out_path = "../../outputs/scratch_from_scratchpad_mixed_v2_test2/model/"
    model_checkpoint_path = "outputs/scratch_from_scratchpad_direct_test2/model/"
    # model_out_path = None

    rng = jax.random.PRNGKey(0)

    if model_out_path is not None:
        if not os.path.exists(os.path.dirname(model_out_path)):
            os.makedirs(os.path.dirname(model_out_path))
    if not os.path.exists(os.path.dirname(data_out_path)):
        os.makedirs(os.path.dirname(data_out_path))
    
    all_data = generate_scratchpad_dataset(digits=list(range(1, 9)), n_items=555, seed=0)

    train_frac = 0.9

    train_instance = replace(
        all_data, 
        questions=all_data.questions[:int(len(all_data.questions)*train_frac)], 
        scratchpad_answers=all_data.scratchpad_answers[:int(len(all_data.scratchpad_answers)*train_frac)], 
        direct_answers=all_data.direct_answers[:int(len(all_data.direct_answers)*train_frac)], 
    )

    # with open("../../outputs/scratch_from_scratchpad_direct_test2/reasoning_data10k_gen_qs.pkl", 'rb') as f:
    #     train_instance = pkl.load(f)
    
    eval_instance = replace(
        all_data, 
        questions=all_data.questions[int(len(all_data.questions)*train_frac):], 
        scratchpad_answers=all_data.scratchpad_answers[int(len(all_data.scratchpad_answers)*train_frac):], 
        direct_answers=all_data.direct_answers[int(len(all_data.direct_answers)*train_frac):], 
    )
    
    tokenizer = T5Tokenizer.from_pretrained('google/t5-small-lm-adapt')

    
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
            model_str="google/t5-small-lm-adapt", 
            # model_str="allenai/tk-instruct-11b-def-pos-neg-expl", 
            # checkpoint_path=None, 
            # checkpoint_path='outputs/T5_11B_random_sharded/model_1/', 
            # checkpoint_path='outputs/scratch_from_scratchpad_direct_test2/model/', 
            checkpoint_path=model_checkpoint_path, 
            from_pretrained=True, 
            use_fp16=False, 
            gradient_checkpoint=False, 
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

    train_f = scratchpads_train if use_scratchpad else direct_train
    eval_f = scratchpads_eval if use_scratchpad else direct_eval
    # train_f = mixed_train
    # eval_f = direct_eval
    
    if jax.process_index() == 0 and use_wandb:
        wandb.init(project='addition_scratchpads_iclr', name='mixed_sequential_test2')
    
    for i in range(100):
        print('iteration', i)
        print('training model ...')

        rng, new_rng = jax.random.split(rng)
        trainer = train_f(
            reasoning_data=train_instance, trainer=trainer, mesh=mesh, bsize=8, epochs=10, 
            max_input_length=512, max_output_length=512, rng_key=new_rng, 
        )

        inference.update_params(trainer.params)

        print('evaluating model ...')

        rng, new_rng = jax.random.split(rng)
        accuracy, all_results = eval_f(
            reasoning_data=eval_instance, inference=inference, mesh=mesh, bsize=32, 
            num_instances=None, max_input_length=512, rng_key=new_rng, 
            do_sample=False, num_beams=1, max_length=512, 
        )
        print(accuracy)
        if jax.process_index() == 0 and use_wandb:
            wandb.log(accuracy)
    
        with open(os.path.join(data_out_path, 'all_results_%d.pkl' % (i)), 'wb') as f:
            pkl.dump(all_results, f)
        
        if model_out_path is not None:
            model.save_pretrained(
                model_out_path, 
                params=jax.device_get(trainer.params), 
            )
    
    if jax.process_index() == 0 and use_wandb:
        wandb.finish()
