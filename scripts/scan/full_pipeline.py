import random
from micro_config import MetaConfig
from base_configs import project_root, AdamWConfig
from t5_config import T5ModelConfig
from core import TKInferenceConfig
import contextlib
from core import TKServerInference, TKTrainConfig
from nat_inst.ni_formatter import get_formatted_ni_data
from task_assoc import generate_task_association_data, get_binary_tasks
from injection_functions import distill, generate_distillation_data, random_token_questions, tk_generate_questions
import pickle as pkl
import jax
import json
import os
import numpy as np
from transformers import T5Tokenizer
from tk_inject import generate_tk_instance, tk_evaluate
from scan_data import create_scan_cirriculum

from utils.randomness import RandomState, seed_context

if __name__ == "__main__":
    
    # setup
    
    model_out_path = '../../outputs/scan/explanation_cirriculum_test1/'
    teacher_checkpoint = 'outputs/T5_11B_tk_no_scan/model_9728/'
    student_checkpoint = 'outputs/T5_11B_tk_no_scan/model_9728/'

    data_out_path = "../../outputs/scan/scan_cirriculum1_with_explanation/"

    if model_out_path is not None:
        if not os.path.exists(os.path.dirname(model_out_path)):
            os.makedirs(os.path.dirname(model_out_path))
    if not os.path.exists(os.path.dirname(data_out_path)):
        os.makedirs(os.path.dirname(data_out_path))
    
    cirriculum, eval_data = create_scan_cirriculum(add_explanation=True, n_positive=2, seed=0)

    random_state = RandomState(0)
    rng_key = jax.random.PRNGKey(0)

    teacher_accuracies = []
    student_pre_distill_accuracies = []
    student_post_distill_accuracies = []

    for i in range(len(cirriculum)):
        print(cirriculum[i].teacher_prompt)

        tokenizer = T5Tokenizer.from_pretrained('google/t5-xxl-lm-adapt')
        
        # save config

        os.system(f'cp {__file__} {os.path.join(data_out_path, "config.py")}')

        # load teacher

        print('loading teacher ...')
        
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
                checkpoint_path=teacher_checkpoint, 
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
        
        # eval teacher

        if len(cirriculum[i].dataset_eval) > 0:
        
            print('evaluating teacher ...')
            
            rng_key, new_rng = jax.random.split(rng_key)
            acc, all_results = tk_evaluate(cirriculum[i], teacher_eval=True, 
                                        inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                        max_input_length=1024, rng_key=new_rng, 
                                        do_sample=False, num_beams=1, max_length=128)
            teacher_accuracies.append(acc)
            print('accuracy:', acc)
        
        # get questions

        with seed_context(random_state):
            questions = cirriculum[i].dataset_questions
            random.shuffle(questions)
            questions = questions[:1024]
        
        # distillation data generation
        
        print('generating distillation data ...')

        rng_key, new_rng = jax.random.split(rng_key)
        distill_data = generate_distillation_data(
            cirriculum[i], questions, inference=inference, mesh=mesh, 
            bsize=1, n_per_question=int(1024 // len(questions)), 
            max_input_length=1024, max_output_length=128, 
            rng_key=new_rng, do_sample=True, num_beams=1, 
        )
        
        # load student

        if student_checkpoint != teacher_checkpoint:
            
            print('loading student ...')
            
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
                    checkpoint_path=student_checkpoint, 
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

            del trainer
            del inference
            del model
            del mesh
            trainer, inference, model, mesh = None, None, None, None

            trainer, inference, model, mesh = trainer_config.unroll(metaconfig)
        
        # eval student

        if len(cirriculum[i].dataset_eval) > 0:

            print('evaluating student before distillation ...')
            
            rng_key, new_rng = jax.random.split(rng_key)
            acc, all_results = tk_evaluate(cirriculum[i], teacher_eval=False, 
                                            inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                            max_input_length=1024, rng_key=new_rng, 
                                            do_sample=False, num_beams=1, max_length=128)
            student_pre_distill_accuracies.append(acc)
            print('accuracy:', acc)
        
        # distill student
        
        print('distilling ...')

        rng_key, new_rng = jax.random.split(rng_key)
        trainer = distill(distill_data, trainer, mesh, bsize=8, epochs=1, max_input_length=256, rng_key=new_rng)

        inference.update_params(trainer.params)
        
        if model_out_path is not None:
            model.save_pretrained(
                model_out_path, 
                params=jax.device_get(trainer.params), 
            )

        # eval student

        print('evaluating student after distillation ...')

        cirriculum_results = []

        for x in range(len(cirriculum)):

            if len(cirriculum[x].dataset_eval) > 0:
        
                rng_key, new_rng = jax.random.split(rng_key)
                acc, all_results = tk_evaluate(cirriculum[x], teacher_eval=False, 
                                                inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                                max_input_length=1024, rng_key=new_rng, 
                                                do_sample=False, num_beams=1, max_length=128)
                cirriculum_results.append(acc)
                print('accuracy:', acc)
        
        acc, all_results = tk_evaluate(eval_data, teacher_eval=False, 
                                       inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                       max_input_length=1024, rng_key=new_rng, 
                                       do_sample=False, num_beams=1, max_length=128)
        cirriculum_results.append(acc)

        student_post_distill_accuracies.append(cirriculum_results)
        
        print('summary:')
        print('teacher accuracy:', teacher_accuracies)
        print('pre-distill student accuracy:', student_pre_distill_accuracies)
        print('post-distill student accuracy:', student_post_distill_accuracies)

        del trainer
        del inference
        del model
        del mesh
        trainer, inference, model, mesh = None, None, None, None

        student_checkpoint = model_out_path
    
    with open(os.path.join(data_out_path, 'teacher_accuracies.pkl'), 'wb') as f:
        pkl.dump({'teacher_accuracies': teacher_accuracies, 'pre_distill_student_accuracies': student_pre_distill_accuracies, 'post_distill_student_accuracies': student_post_distill_accuracies}, f)


