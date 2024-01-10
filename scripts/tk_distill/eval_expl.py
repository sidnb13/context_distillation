from copy import copy
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
from tk_inject import input_generator_question_prompt, tk_evaluate
from explanations import generate_tk_instance, generate_tk_instance2

from utils.randomness import RandomState, seed_context

if __name__ == "__main__":
    
    # setup
    
    # data_out_path = "../../outputs/task_assoc_dataset_data_test2/"
    # data_out_path = "../../outputs/tk_inject_2_negative_plus_explanation___task1624_disfl_qa_question_yesno_classification/"
    # model_out_path = "../../outputs/tk_inject_2_negative___task1624_disfl_qa_question_yesno_classification/model/"
    teacher_checkpoint = 'outputs/T5_11B_random_nat_inst_finetune_test2/model/'

    # task = 'task879_schema_guided_dstc8_classification'
    # task = 'task1624_disfl_qa_question_yesno_classification'
    # 2pos2neg v 2pos2neg+expl
    tasks = [
        ('task1631_openpi_answer_generation', True, True), 
        ('task039_qasc_find_overlapping_words', True, True), 
        ('task1157_bard_analogical_reasoning_rooms_for_containers', True, True), 
        ('task1158_bard_analogical_reasoning_manipulating_items', True, True), 
        ('task1516_imppres_naturallanguageinference', True, True), 
        
        ('task034_winogrande_question_modification_object', True, True), 
        ('task1540_parsed_pdfs_summarization', True, True), 
        ('task418_persent_title_generation', True, True), 
        ('task401_numeric_fused_head_reference', True, True), 
        ('task891_gap_coreference_resolution', True, True), 

        ('task1624_disfl_qa_question_yesno_classification', True, False), 
        ('task970_sherliic_causal_relationship', True, False), 
        ('task1516_imppres_naturallanguageinference', True, False), 
        ('task1195_disflqa_disfluent_to_fluent_conversion', True, False), 
        ('task362_spolin_yesand_prompt_response_sub_classification', True, False), 

        ('task1586_scifact_title_generation', True, False), 
        ('task743_eurlex_summarization', True, False), 
        ('task1155_bard_analogical_reasoning_trash_or_treasure', True, False), 
        ('task033_winogrande_answer_generation', True, False), 
        ('task937_defeasible_nli_social_classification', True, False), 
    ]

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

    data_out_path = "../../outputs/no_expl_neg_only_eval_teacher_test12/"
    
    if not os.path.exists(os.path.dirname(data_out_path)):
        os.makedirs(os.path.dirname(data_out_path))
    
    total_results = {}

    for task, add_negatives, add_positives in tasks:
        print('RUNNING:')
        print('='*25)
        print(task, add_negatives, add_positives)
        print('='*25)
        
        random_state = RandomState(0)
        rng_key = jax.random.PRNGKey(0)
        # rng_key, new_key = jax.random.split(rng_key)
        # random_state = RandomState(jax.random.randint(new_key, [], 0, 2**30).item())

        tokenizer = T5Tokenizer.from_pretrained('google/t5-xxl-lm-adapt')
        
        # save config

        os.system(f'cp {__file__} {os.path.join(data_out_path, "config.py")}')

        # generate injection data
        
        print('preparing data ...')

        random_state_copy = copy(random_state)
        
        with seed_context(random_state):
            formatted_train_data, formatted_test_data = get_formatted_ni_data(
                add_task_name=False, add_task_definition=True, num_pos_examples=0, 
                num_neg_examples=2, 
                add_explanation=False, 
                max_num_instances_per_task=100, 
                max_num_instances_per_eval_task=None, 
            )

        with seed_context(random_state_copy):
            formatted_train_data_no_expl, formatted_test_data_no_expl = get_formatted_ni_data(
                add_task_name=False, add_task_definition=True, num_pos_examples=0, 
                num_neg_examples=0, 
                add_explanation=False, 
                max_num_instances_per_task=100, 
                max_num_instances_per_eval_task=None, 
            )
        
        # prompt = formatted_test_data[task][0]['prompt']
        # formatted_test_data[task][0]['prompt'] = prompt[:prompt.find('Positive Example 1')] + prompt[prompt.find('Negative Example 1'):]
        # print([prompt])

        # breakpoint()

        with seed_context(random_state):
            injection_data = generate_tk_instance2(formatted_test_data[task], formatted_test_data_no_expl[task][0]['prompt'])
        
        # eval teacher

        print('evaluating teacher ...')
        
        print('task:', task)
        rng_key, new_rng = jax.random.split(rng_key)
        acc, all_results = tk_evaluate(injection_data, teacher_eval=True, 
                                    inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                    max_input_length=1024, rng_key=new_rng, 
                                    do_sample=False, num_beams=1, max_length=128)
        total_results[task] = acc
        print('accuracy:', acc)
    
    with open(os.path.join(data_out_path, 'eval_results.pkl'), 'wb') as f:
        pkl.dump(total_results, f)
