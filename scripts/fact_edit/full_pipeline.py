from micro_config import MetaConfig
from base_configs import project_root, AdamWConfig
from t5_config import T5ModelConfig
from core import TKInferenceConfig
import contextlib
from core import TKServerInference, TKTrainConfig
# from fact_edit import counterfact_evaluate, generate_counterfact_instance, shuffle_countefact_data
from fact_edit import counterfact_evaluate, generate_counterfact_instance, shuffle_countefact_data
from injection_functions import distill, generate_distillation_data, random_token_questions, tk_generate_questions
import pickle as pkl
import jax
import json
import os
import numpy as np
from transformers import T5Tokenizer
from tk_inject import generate_tk_instance, tk_evaluate
from utils.randomness import RandomState, seed_context
import random
import tree

if __name__ == "__main__":
    
    # setup
    
    # data_out_path = "../../outputs/task_assoc_dataset_data_test2/"
    data_out_path = "../../outputs/fact_edit_from_trained_teacher_test1/"
    # model_out_path = "../../outputs/fact_edit_test1/model/"
    model_out_path = None
    # teacher_checkpoint = 'outputs/T5_11B_random_nat_inst_finetune_test2/model/'
    # student_checkpoint = 'outputs/T5_11B_random_nat_inst_finetune_test2/model/'
    teacher_checkpoint = 'outputs/trained_counterface_teacher/model/'
    student_checkpoint = 'outputs/trained_counterface_teacher/model/'
    
    rng_key = jax.random.PRNGKey(0)
    rng_key, new_key = jax.random.split(rng_key)
    random_state = RandomState(jax.random.randint(new_key, [], 0, 2**30).item())
    
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
        with open('../../data/counterfact/new_counterfact_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        injection_data = []
        for i in range(len(data)):
            shuffle_countefact_data(data[i])
            injection_data.append(generate_counterfact_instance(data[i], new_data=True))
            # injection_data.append(generate_counterfact_instance(data[i]))
        # injection_data = injection_data[:1]
    
    with open(os.path.join(data_out_path, 'injection_data.pkl'), 'wb') as f:
        pkl.dump(injection_data, f)
    
    # eval teacher

    print('evaluating teacher ...')

    teacher_accuracies = []
    
    for item in injection_data:
        # print('task:', item.teacher_prompt)
        acc, all_results = counterfact_evaluate(item, teacher_eval=True, 
                                                inference=inference, mesh=mesh, bsize=1, 
                                                max_input_length=1024, max_output_length=128)
        teacher_accuracies.append(acc)
    print('avg:', tree.map_structure(lambda *x: sum(x) / len(x), *teacher_accuracies))
    
    with open(os.path.join(data_out_path, 'teacher_accuracies.pkl'), 'wb') as f:
        pkl.dump(teacher_accuracies, f)
    
    # question generation

    print('generating questions ...')
    
    all_questions = []

    for item in injection_data:
        # print('task:', task)
        rng_key, new_rng = jax.random.split(rng_key)
        questions = tk_generate_questions(item, inference=inference, mesh=mesh, 
                                          bsize=1, n_questions=2*1024, 
                                          max_input_length=1024, rng_key=new_rng, 
                                          do_sample=True, num_beams=1, max_length=128)
        # questions = item.dataset_questions
        all_questions.append(questions)
    
    # total_questions = sum(all_questions.values(), [])
    # all_questions = {task: total_questions for task in all_questions.keys()}
    
    with open(os.path.join(data_out_path, 'questions.pkl'), 'wb') as f:
        pkl.dump(all_questions, f)
    
    # distillation data generation

    pre_distill_student_accuracies = []
    post_distill_student_accuracies = []
    
    # all_distill_data = sum(grouped_distill_data.values(), [])
    for item, questions in zip(injection_data, all_questions):

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

        del trainer
        del inference
        del model
        del mesh
        trainer, inference, model, mesh = None, None, None, None

        trainer, inference, model, mesh = trainer_config.unroll(metaconfig)
        
        
        print('generating distillation data ...')

        # print('task:', task)
        rng_key, new_rng = jax.random.split(rng_key)
        distill_data = generate_distillation_data(
            item, questions, inference=inference, mesh=mesh, 
            bsize=1, n_per_question=1, max_input_length=1024, max_output_length=128, 
            rng_key=new_rng, do_sample=True, num_beams=1, 
        )
        
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

        print('evaluating student before distillation ...')

        rng_key, new_rng = jax.random.split(rng_key)
        acc, all_results = counterfact_evaluate(item, teacher_eval=False, 
                                                inference=inference, mesh=mesh, bsize=1, 
                                                max_input_length=1024, max_output_length=128)
        pre_distill_student_accuracies.append(acc)
        
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
        
        # print('task:', task)
        rng_key, new_rng = jax.random.split(rng_key)
        acc, all_results = counterfact_evaluate(item, teacher_eval=False, 
                                                inference=inference, mesh=mesh, bsize=1, 
                                                max_input_length=1024, max_output_length=128)
        post_distill_student_accuracies.append(acc)
    
    with open(os.path.join(data_out_path, 'pre_distill_student_accuracies.pkl'), 'wb') as f:
        pkl.dump(pre_distill_student_accuracies, f)
    with open(os.path.join(data_out_path, 'post_distill_student_accuracies.pkl'), 'wb') as f:
        pkl.dump(post_distill_student_accuracies, f)
    
    print('summary:')
    print('teacher accuracy:', tree.map_structure(lambda *x: sum(x) / len(x), *teacher_accuracies))
    print('pre-distill student accuracy:', tree.map_structure(lambda *x: sum(x) / len(x), *pre_distill_student_accuracies))
    print('post-distill student accuracy:', tree.map_structure(lambda *x: sum(x) / len(x), *post_distill_student_accuracies))


