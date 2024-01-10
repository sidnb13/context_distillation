from micro_config import MetaConfig
from base_configs import project_root, AdamWConfig
from gradient_descent import generate_gradient_descent_prompt_data
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

from utils.randomness import RandomState, seed_context

if __name__ == "__main__":
    
    # setup
    
    # data_out_path = "../../outputs/task_assoc_dataset_data_test2/"
    # data_out_path = "../../outputs/tk_inject_2_negative_plus_explanation___task1624_disfl_qa_question_yesno_classification/"
    # model_out_path = "../../outputs/tk_inject_2_negative___task1624_disfl_qa_question_yesno_classification/model/"
    model_out_path = 'outputs/iterative_student/model/'
    teacher_checkpoint = 'outputs/T5_11B_random_nat_inst_finetune_test2/model/'
    student_checkpoint = 'outputs/T5_11B_random_nat_inst_finetune_test2/model/'

    assert model_out_path is not None

    add_description = False

    tasks = [
        'task1391_winogrande_easy_answer_generation', 3, False
    ]
    
    for task, n_per_prompt, update_question_prompts in tasks:
        print('RUNNING:')
        print('='*25)
        print(task, n_per_prompt)
        print('='*25)

        data_out_path = "../../outputs/injectionpromptvgradientdescent/tk_inject_per_prompt_%d_%s_add_description_%s___%s/" % (n_per_prompt, str(update_question_prompts), str(add_description), task)
        
        random_state = RandomState(0)
        rng_key = jax.random.PRNGKey(0)
        # rng_key, new_key = jax.random.split(rng_key)
        # random_state = RandomState(jax.random.randint(new_key, [], 0, 2**30).item())
        
        if model_out_path is not None:
            if not os.path.exists(os.path.dirname(model_out_path)):
                os.makedirs(os.path.dirname(model_out_path))
        if not os.path.exists(os.path.dirname(data_out_path)):
            os.makedirs(os.path.dirname(data_out_path))

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

        # generate injection data
        
        print('preparing data ...')
        
        injection_datas, _, _, _ = generate_gradient_descent_prompt_data(50, 50, n_per_prompt, 0, update_question_prompts, add_description)

        injection_datas = injection_datas[task]
        
        with open(os.path.join(data_out_path, 'injection_datas.pkl'), 'wb') as f:
            pkl.dump(injection_datas, f)
        
        all_teacher_accuracies = []
        all_pre_distill_stucent_accuracies = []
        all_post_distill_student_accuracies = []
        
        for i, injection_data in enumerate(injection_datas):

            # eval teacher
            
            print('evaluating teacher ...')

            teacher_accuracies = {}
            
            print('task:', task)
            rng_key, new_rng = jax.random.split(rng_key)
            acc, all_results = tk_evaluate(injection_data, teacher_eval=True, 
                                           inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                           max_input_length=1024, rng_key=new_rng, 
                                           do_sample=False, num_beams=1, max_length=128)
            teacher_accuracies[task] = acc
            print('accuracy:', acc)
            
            all_teacher_accuracies.append(teacher_accuracies)
            with open(os.path.join(data_out_path, 'step_%d_teacher_accuracies.pkl' % (i)), 'wb') as f:
                pkl.dump(teacher_accuracies, f)

            # question generation

            print('generating questions ...')
            
            all_questions = {}

            print('task:', task)
            rng_key, new_rng = jax.random.split(rng_key)
            questions = tk_generate_questions(injection_data, inference=inference, mesh=mesh, 
                                            bsize=1, n_questions=1024, 
                                            max_input_length=1024, rng_key=new_rng, 
                                            do_sample=True, num_beams=1, max_length=128)

            all_questions[task] = questions
            
            total_questions = sum(all_questions.values(), [])
            all_questions = {task: total_questions for task in all_questions.keys()}
            
            with open(os.path.join(data_out_path, 'step_%d_questions.pkl' % (i)), 'wb') as f:
                pkl.dump(all_questions, f)
            
            # distillation data generation
            
            print('generating distillation data ...')
            
            grouped_distill_data = {}

            print('task:', task)
            rng_key, new_rng = jax.random.split(rng_key)
            distill_data = generate_distillation_data(
                injection_data, all_questions[task], inference=inference, mesh=mesh, 
                bsize=1, n_per_question=1, max_input_length=1024, max_output_length=128, 
                rng_key=new_rng, do_sample=True, num_beams=1, 
            )
            grouped_distill_data[task] = distill_data

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

            print('evaluating student before distillation ...')

            pre_distill_student_accuracies = {}
            
            print('task:', task)
            rng_key, new_rng = jax.random.split(rng_key)
            acc, all_results = tk_evaluate(injection_data, teacher_eval=False, 
                                           inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                           max_input_length=1024, rng_key=new_rng, 
                                           do_sample=False, num_beams=1, max_length=128)
            pre_distill_student_accuracies[task] = acc
            print('accuracy:', acc)
            
            all_pre_distill_stucent_accuracies.append(pre_distill_student_accuracies)
            with open(os.path.join(data_out_path, 'step_%d_pre_distill_student_accuracies.pkl' % (i)), 'wb') as f:
                pkl.dump(pre_distill_student_accuracies, f)
            
            # distill student
            
            print('distilling ...')
            
            all_distill_data = sum(grouped_distill_data.values(), [])

            rng_key, new_rng = jax.random.split(rng_key)
            trainer = distill(all_distill_data, trainer, mesh, bsize=8, epochs=1, max_input_length=256, rng_key=new_rng)

            inference.update_params(trainer.params)
            
            if model_out_path is not None:
                model.save_pretrained(
                    model_out_path, 
                    params=jax.device_get(trainer.params), 
                )

            # eval student

            print('evaluating student after distillation ...')

            post_distill_student_accuracies = {}
            
            print('task:', task)
            rng_key, new_rng = jax.random.split(rng_key)
            acc, all_results = tk_evaluate(injection_data, teacher_eval=False, 
                                            inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                            max_input_length=1024, rng_key=new_rng, 
                                            do_sample=False, num_beams=1, max_length=128)
            post_distill_student_accuracies[task] = acc
            print('accuracy:', acc)
            
            all_post_distill_student_accuracies.append(post_distill_student_accuracies)
            with open(os.path.join(data_out_path, 'step_%d_post_distill_student_accuracies.pkl' % (i)), 'wb') as f:
                pkl.dump(post_distill_student_accuracies, f)
            
            student_checkpoint = model_out_path
        
        print('summary:')
        print('teacher accuracy:', all_teacher_accuracies)
        print('pre-distill student accuracy:', all_pre_distill_stucent_accuracies)
        print('post-distill student accuracy:', all_post_distill_student_accuracies)

        del trainer
        del inference
        del model
        del mesh
        trainer, inference, model, mesh = None, None, None, None


