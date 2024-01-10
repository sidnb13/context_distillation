from collections import defaultdict
from dataclasses import replace
from micro_config import MetaConfig
from base_configs import project_root, AdamWConfig
from t5_config import T5ModelConfig
from core import TKInferenceConfig
import contextlib
from core import TKServerInference, TKTrainConfig
from nat_inst.ni_formatter import get_formatted_ni_data
from task_assoc import generate_task_association_data, get_binary_tasks, permute_task_association_instance, permute_distillation_data
from injection_functions import distill, generate_distillation_data, random_token_questions, tk_generate_questions
import pickle as pkl
import jax
import json
import os
import numpy as np
from transformers import T5Tokenizer
from tk_inject import tk_evaluate
from utils.randomness import RandomState, seed_context

if __name__ == "__main__":
    
    # setup
    
    # data_out_path = "../../outputs/task_assoc_dataset_data_test2/"
    data_out_path = "../../outputs/task_assoc_permutations_test3_negative_qs/"
    # model_out_path = "../../outputs/task_assoc_permutations_test1/model/"
    model_out_path = None
    teacher_checkpoint = 'outputs/T5_11B_random_nat_inst_finetune_test2/model/'
    student_checkpoint = 'outputs/T5_11B_random_nat_inst_finetune_test2/model/'
    
    rng_key = jax.random.PRNGKey(0)
    rng_key, new_key = jax.random.split(rng_key)
    random_state = RandomState(jax.random.randint(new_key, [], 0, 2**30).item())
    
    tasks = [
             'task1393_superglue_copa_text_completion', 
             'task1394_meta_woz_task_classification', 
             'task242_tweetqa_classification', 
             'task220_rocstories_title_classification', 
            ]

    permutations = [
        [0, 1, 2, 3], 
        [3, 0, 1, 2], 
        [2, 3, 0, 1], 
        [1, 2, 3, 0], 
    ]
    
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
            num_neg_examples=0, add_explanation=False, max_num_instances_per_task=100, 
            max_num_instances_per_eval_task=100, 
        )

        injection_data = generate_task_association_data(tasks, formatted_test_data)
    
    with open(os.path.join(data_out_path, 'injection_data.pkl'), 'wb') as f:
        pkl.dump(injection_data, f)
    
    # eval teacher

    print('evaluating teacher ...')

    teacher_accuracies = {}
    
    for task in injection_data.keys():
        print('task:', task)
        rng_key, new_rng = jax.random.split(rng_key)
        acc, all_results = tk_evaluate(injection_data[task], teacher_eval=True, 
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
    for task in injection_data.keys():
        print('task:', task)
        rng_key, new_rng = jax.random.split(rng_key)
        questions = tk_generate_questions(injection_data[task], inference=inference, mesh=mesh, 
                                          bsize=1, n_questions=4096, 
                                          max_input_length=1024, rng_key=new_rng, 
                                          do_sample=True, num_beams=1, max_length=128)
        # questions = random_token_questions(tokens=list(filter(lambda x: x not in ([tokenizer.pad_token_id, 
        #                                    tokenizer.eos_token]+tokenizer.additional_special_tokens), range(len(tokenizer)))), 
        #                                    tokenizer=tokenizer, n_questions=1024*4, max_question_length=128, 
        #                                    rng_key=new_rng)
        # questions = injection_data[task].dataset_questions
        # questions = sum(map(lambda x: x.dataset_questions, injection_data.values()), [])
        all_questions[task] = questions
    
    total_questions = sum(all_questions.values(), [])
    all_questions = {task: total_questions for task in all_questions.keys()}
    
    with open(os.path.join(data_out_path, 'questions.pkl'), 'wb') as f:
        pkl.dump(all_questions, f)
    
    # distillation data generation
    
    print('generating distillation data ...')
    
    grouped_distill_data = {}

    for task in injection_data.keys():
        print('task:', task)
        rng_key, new_rng = jax.random.split(rng_key)
        distill_data = generate_distillation_data(
            injection_data[task], all_questions[task], inference=inference, mesh=mesh, 
            bsize=1, n_per_question=1, max_input_length=1024, max_output_length=128, 
            rng_key=new_rng, compression_n_samples=100, postproc_f=None, do_sample=True, num_beams=1, 
        )
        grouped_distill_data[task] = distill_data
    
    # with open(os.path.join(data_out_path, 'distill_data.pkl'), 'wb') as f:
    #     pkl.dump(grouped_distill_data, f)
    
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

        trainer, inference, model, mesh = trainer_config.unroll(metaconfig)
    
    for perm_idx, permutation in enumerate(permutations):
    
        # eval student

        print('evaluating student before distillation ...')

        pre_distill_student_accuracies = defaultdict(dict)

        for permutation_eval in permutations:

            eval_permuted_instances = {tasks[i]: instance for i, instance in enumerate(permute_task_association_instance([injection_data[task] for task in tasks], permutation_eval))}
            
            for task in eval_permuted_instances.keys():
                print('task:', task)
                rng_key, new_rng = jax.random.split(rng_key)
                acc, all_results = tk_evaluate(eval_permuted_instances[task], teacher_eval=False, 
                                            inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                            max_input_length=1024, rng_key=new_rng, 
                                            do_sample=False, num_beams=1, max_length=128)
                pre_distill_student_accuracies[task][eval_permuted_instances[task].student_prompt] = acc
                print('accuracy:', acc)
        
        with open(os.path.join(data_out_path, 'pre_distill_student_accuracies_%d.pkl' % (perm_idx)), 'wb') as f:
            pkl.dump(pre_distill_student_accuracies, f)
        
        # distill student
        
        print('distilling ...')

        all_distill_data = sum([permute_distillation_data(distill_item, permutation) for distill_item in grouped_distill_data.values()], [])

        rng_key, new_rng = jax.random.split(rng_key)
        trainer = distill(all_distill_data, trainer, mesh, bsize=8, epochs=1, max_input_length=1024, rng_key=new_rng, decompress_smoothing_epsilon=1e-7)

        inference.update_params(trainer.params)
        
        if model_out_path is not None:
            model.save_pretrained(
                model_out_path, 
                params=jax.device_get(trainer.params), 
            )

        # eval student

        print('evaluating student after distillation ...')

        post_distill_student_accuracies = defaultdict(dict)
        
        for permutation_eval in permutations:

            eval_permuted_instances = {tasks[i]: instance for i, instance in enumerate(permute_task_association_instance([injection_data[task] for task in tasks], permutation_eval))}
        
            for task in eval_permuted_instances.keys():
                print('task:', task)
                rng_key, new_rng = jax.random.split(rng_key)
                acc, all_results = tk_evaluate(eval_permuted_instances[task], teacher_eval=False, 
                                            inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                            max_input_length=1024, rng_key=new_rng, 
                                            do_sample=False, num_beams=1, max_length=128)
                post_distill_student_accuracies[task][eval_permuted_instances[task].student_prompt] = acc
                print('accuracy:', acc)
        
        with open(os.path.join(data_out_path, 'post_distill_student_accuracies_%d.pkl' % (perm_idx)), 'wb') as f:
            pkl.dump(post_distill_student_accuracies, f)
