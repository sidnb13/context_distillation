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

from utils.randomness import RandomState, seed_context

if __name__ == "__main__":
    
    # setup
    
    # data_out_path = "../../outputs/task_assoc_dataset_data_test2/"
    # data_out_path = "../../outputs/tk_inject_2_negative_plus_explanation___task1624_disfl_qa_question_yesno_classification/"
    # model_out_path = "../../outputs/tk_inject_2_negative___task1624_disfl_qa_question_yesno_classification/model/"
    model_out_path = None
    teacher_checkpoint = 'outputs/T5_11B_random_nat_inst_finetune_test2/model/'
    student_checkpoint = 'outputs/T5_11B_random_nat_inst_finetune_test2/model/'

    # task = 'task879_schema_guided_dstc8_classification'
    # task = 'task1624_disfl_qa_question_yesno_classification'
    # 2pos2neg v 2pos2neg+expl
    # tasks = [
    #          ('task1631_openpi_answer_generation', True), 
    #          ('task1631_openpi_answer_generation', False), 
    #          ('task039_qasc_find_overlapping_words', True), 
    #          ('task039_qasc_find_overlapping_words', False), 
    #          ('task1157_bard_analogical_reasoning_rooms_for_containers', True), 
    #          ('task1157_bard_analogical_reasoning_rooms_for_containers', False), 
    #          ('task1158_bard_analogical_reasoning_manipulating_items', True), 
    #          ('task1158_bard_analogical_reasoning_manipulating_items', False), 
    #          ('task1516_imppres_naturallanguageinference', True), 
    #          ('task1516_imppres_naturallanguageinference', False), 
             
    #          ('task034_winogrande_question_modification_object', True), 
    #          ('task034_winogrande_question_modification_object', False), 
    #          ('task1540_parsed_pdfs_summarization', True), 
    #          ('task1540_parsed_pdfs_summarization', False), 
    #          ('task418_persent_title_generation', True), 
    #          ('task418_persent_title_generation', False), 
    #          ('task401_numeric_fused_head_reference', True), 
    #          ('task401_numeric_fused_head_reference', False), 
    #          ('task891_gap_coreference_resolution', True), 
    #          ('task891_gap_coreference_resolution', False), 
    #         ]
    tasks = [
        ('task1624_disfl_qa_question_yesno_classification', True, True), 
        ('task1624_disfl_qa_question_yesno_classification', False, True), 
        ('task1624_disfl_qa_question_yesno_classification', False, False), 
        ('task970_sherliic_causal_relationship', True, True), 
        ('task970_sherliic_causal_relationship', False, True), 
        ('task970_sherliic_causal_relationship', False, False), 
        ('task1516_imppres_naturallanguageinference', True, True), 
        ('task1516_imppres_naturallanguageinference', False, True), 
        ('task1516_imppres_naturallanguageinference', False, False), 
        ('task1195_disflqa_disfluent_to_fluent_conversion', True, True), 
        ('task1195_disflqa_disfluent_to_fluent_conversion', False, True), 
        ('task1195_disflqa_disfluent_to_fluent_conversion', False, False), 
        ('task362_spolin_yesand_prompt_response_sub_classification', True, True), 
        ('task362_spolin_yesand_prompt_response_sub_classification', False, True), 
        ('task362_spolin_yesand_prompt_response_sub_classification', False, False), 

        ('task1586_scifact_title_generation', True, True), 
        ('task1586_scifact_title_generation', False, True), 
        ('task1586_scifact_title_generation', False, False), 
        ('task743_eurlex_summarization', True, True), 
        ('task743_eurlex_summarization', False, True), 
        ('task743_eurlex_summarization', False, False), 
        ('task1155_bard_analogical_reasoning_trash_or_treasure', True, True), 
        ('task1155_bard_analogical_reasoning_trash_or_treasure', False, True), 
        ('task1155_bard_analogical_reasoning_trash_or_treasure', False, False), 
        ('task033_winogrande_answer_generation', True, True), 
        ('task033_winogrande_answer_generation', False, True), 
        ('task033_winogrande_answer_generation', False, False), 
        ('task937_defeasible_nli_social_classification', True, True), 
        ('task937_defeasible_nli_social_classification', False, True), 
        ('task937_defeasible_nli_social_classification', False, False), 
    ]
    
    for task, add_explanation, add_negatives in tasks:
        print('RUNNING:')
        print('='*25)
        print(task, add_explanation)
        print('='*25)

        data_out_path = "../../outputs/inject2negv2negexpl_test2/tk_inject_%s_%s___%s/" % ('2_negative' if add_negatives else '', 'plus_explanation' if add_explanation else '', task)
        
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
                add_task_name=False, add_task_definition=True, num_pos_examples=0, 
                num_neg_examples=2 if add_negatives else 0, 
                add_explanation=add_explanation, 
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


