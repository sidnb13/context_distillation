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

    model_checkpoint_paths = {
        # "scratchpad_model": "outputs/scratch_from_scratchpad_direct_test2/model/", 
        # "mixed_same_time_model": "outputs/scratch_from_scratchpad_mixed_test1/model/", 
        # "mixed_sequential_model": "outputs/scratch_from_scratchpad_mixed_v2_test1/model/", 
        # "direct_model": "outputs/scratch_from_scratchpad_direct_test3/model/", 
        # "student_from_dataset_questions": "outputs/direct_student_test1/model/", 
        # "student_from_trained_question_generator": "outputs/direct_student_test2/model/", 
        "mixed_sequential_model2": "outputs/scratch_from_scratchpad_mixed_v2_test2/model/", 
    }

    for name, model_checkpoint_path in model_checkpoint_paths.items():
        rng = jax.random.PRNGKey(0)
        
        all_data = generate_scratchpad_dataset(digits=list(range(1, 9)), n_items=555, seed=0)

        train_frac = 0.9
        
        train_instance = replace(
            all_data, 
            questions=all_data.questions[:int(len(all_data.questions)*train_frac)], 
            scratchpad_answers=all_data.scratchpad_answers[:int(len(all_data.scratchpad_answers)*train_frac)], 
            direct_answers=all_data.direct_answers[:int(len(all_data.direct_answers)*train_frac)], 
        )
        eval_instance = replace(
            all_data, 
            questions=all_data.questions[int(len(all_data.questions)*train_frac):], 
            scratchpad_answers=all_data.scratchpad_answers[int(len(all_data.scratchpad_answers)*train_frac):], 
            direct_answers=all_data.direct_answers[int(len(all_data.direct_answers)*train_frac):], 
        )

        real_all_data = generate_scratchpad_dataset(digits=list(range(1, 9)), n_items=10555, seed=0)

        real_eval_instance = replace(
            real_all_data, 
            questions=real_all_data.questions[555:], 
            scratchpad_answers=real_all_data.scratchpad_answers[555:], 
            direct_answers=real_all_data.direct_answers[555:], 
        )

        assert not any([question in real_eval_instance.questions for question in train_instance.questions])
        
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
            verbose=False, 
        )

        trainer, inference, model, mesh = trainer_config.unroll(metaconfig)

        rng, new_rng = jax.random.split(rng)
        direct_accuracy, all_results = direct_eval(
            reasoning_data=real_eval_instance, inference=inference, mesh=mesh, bsize=32, 
            num_instances=None, max_input_length=512, rng_key=new_rng, 
            do_sample=False, num_beams=1, max_length=512, 
        )
        print('direct:', direct_accuracy)

        rng, new_rng = jax.random.split(rng)
        scratchpad_accuracy, all_results = scratchpads_eval(
            reasoning_data=real_eval_instance, inference=inference, mesh=mesh, bsize=32, 
            num_instances=None, max_input_length=512, rng_key=new_rng, 
            do_sample=False, num_beams=1, max_length=512, 
        )
        print('scratchpads:', scratchpad_accuracy)

        print('='*25)
        print('name:', name)
        print('checkpoint_path:', model_checkpoint_path)
        print('='*25)
        print('direct accuracy:', direct_accuracy)
        print('scratchpad_accuracy:', scratchpad_accuracy)
        print('='*25)
