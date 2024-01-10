from dataclasses import replace
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
from tk_inject import generate_tk_instance, tk_evaluate, input_generator_question_prompt

from utils.randomness import RandomState, seed_context

if __name__ == "__main__":
    
    # setup
    
    # data_out_path = "../../outputs/task_assoc_dataset_data_test2/"
    # data_out_path = "../../outputs/tk_inject_2_negative_plus_explanation___task1624_disfl_qa_question_yesno_classification/"
    # model_out_path = "../../outputs/tk_inject_2_negative___task1624_disfl_qa_question_yesno_classification/model/"
    model_out_path = None
    teacher_checkpoint = 'outputs/T5_11B_random_nat_inst_finetune_test2/model/'
    student_checkpoint = 'outputs/T5_11B_random_nat_inst_finetune_test2/model/'
    input_synth_path = 'outputs/T5_11B_tk_input_generator/model/'

    tasks = [
        'task1391_winogrande_easy_answer_generation', 
        'task1394_meta_woz_task_classification', 
        'task202_mnli_contradiction_classification', 
        'task1155_bard_analogical_reasoning_trash_or_treasure', 
        'task935_defeasible_nli_atomic_classification', 
        'task1386_anli_r2_entailment', 
        'task828_copa_commonsense_cause_effect', 
        'task1624_disfl_qa_question_yesno_classification', 
        'task232_iirc_link_number_classification', 
        'task738_perspectrum_classification', 
        'task362_spolin_yesand_prompt_response_sub_classification', 
        'task1393_superglue_copa_text_completion', 
        'task1154_bard_analogical_reasoning_travel', 
    ]
    
    for task in tasks:
        print('RUNNING:')
        print('='*25)
        print(task)
        print('='*25)

        data_out_path = "../../outputs/inject2negv2negexpl/tk_inject_with_input_generator___%s/" % (task)
        
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
        
        with seed_context(random_state):
            formatted_train_data, formatted_test_data = get_formatted_ni_data(
                add_task_name=False, add_task_definition=True, num_pos_examples=2, 
                num_neg_examples=0, 
                add_explanation=False, 
                max_num_instances_per_task=100, 
                max_num_instances_per_eval_task=100, 
            )

            injection_data = generate_tk_instance(formatted_test_data[task])
        
        with open(os.path.join(data_out_path, 'injection_data.pkl'), 'wb') as f:
            pkl.dump(injection_data, f)
        
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
        
        with open(os.path.join(data_out_path, 'teacher_accuracies.pkl'), 'wb') as f:
            pkl.dump(teacher_accuracies, f)
        
        # question generation

        print('generating questions ...')

        print('loading input synth ...')
    
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
                checkpoint_path=input_synth_path, 
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
        
        all_questions = {}

        print('task:', task)
        rng_key, new_rng = jax.random.split(rng_key)
        with seed_context(random_state):
            input_injection_data = replace(injection_data, question_generation_prompts=[input_generator_question_prompt(injection_data.meta[0], add_task_name=False, add_task_definition=True, num_examples=0)])
        questions = tk_generate_questions(injection_data, inference=inference, mesh=mesh, 
                                        bsize=1, n_questions=1024, 
                                        max_input_length=1024, rng_key=new_rng, 
                                        do_sample=True, num_beams=1, max_length=128)
        all_questions[task] = questions
        
        total_questions = sum(all_questions.values(), [])
        all_questions = {task: total_questions for task in all_questions.keys()}
        
        with open(os.path.join(data_out_path, 'questions.pkl'), 'wb') as f:
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
        
        with open(os.path.join(data_out_path, 'pre_distill_student_accuracies.pkl'), 'wb') as f:
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
        
        with open(os.path.join(data_out_path, 'post_distill_student_accuracies.pkl'), 'wb') as f:
            pkl.dump(post_distill_student_accuracies, f)
        
        print('summary:')
        print('teacher accuracy:', teacher_accuracies[task])
        print('pre-distill student accuracy:', pre_distill_student_accuracies[task])
        print('post-distill student accuracy:', post_distill_student_accuracies[task])

        del trainer
        del inference
        del model
        del mesh
        trainer, inference, model, mesh = None, None, None, None


