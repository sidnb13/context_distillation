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
from control import control_evaluate, generate_control_instance
from datasets import load_dataset

if __name__ == "__main__":
    
    # setup
    
    model_out_path = None
    teacher_checkpoint = 'outputs/T5_11B_random_nat_inst_finetune_test2/model/'
    student_checkpoint = 'outputs/T5_11B_random_nat_inst_finetune_test2/model/'

    print('RUNNING:')

    data_out_path = "../../outputs/control/control_test2/"
    
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
    
    dataset = load_dataset("imdb")
    
    injection_data = generate_control_instance(
                                               train_datapoints=dataset['train'], 
                                               eval_datapoints=dataset['test'], 
                                               seed=0, 
                                               only_negative=True, 
                                               positive_control=False, 
                                            )
    
    with open(os.path.join(data_out_path, 'injection_data.pkl'), 'wb') as f:
        pkl.dump(injection_data, f)
    
    # eval teacher

    print('evaluating teacher ...')
    
    rng_key, new_rng = jax.random.split(rng_key)
    acc, all_results = control_evaluate(injection_data, teacher_eval=True, 
                                inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                max_input_length=1024, max_output_length=128, rng_key=new_rng, 
                                do_sample=False, num_beams=1, max_length=128)
    teacher_accuracies = acc
    print('accuracy:', acc)
    
    with open(os.path.join(data_out_path, 'teacher_accuracies.pkl'), 'wb') as f:
        pkl.dump(teacher_accuracies, f)
    
    # question generation

    print('generating questions ...')

    rng_key, new_rng = jax.random.split(rng_key)
    questions = tk_generate_questions(injection_data, inference=inference, mesh=mesh, 
                                    bsize=1, n_questions=1024, 
                                    max_input_length=1024, rng_key=new_rng, 
                                    do_sample=True, num_beams=1, max_length=128)
    
    with open(os.path.join(data_out_path, 'questions.pkl'), 'wb') as f:
        pkl.dump(questions, f)
    
    # distillation data generation
    
    print('generating distillation data ...')

    rng_key, new_rng = jax.random.split(rng_key)
    distill_data = generate_distillation_data(
        injection_data, questions, inference=inference, mesh=mesh, 
        bsize=1, n_per_question=1, max_input_length=1024, max_output_length=128, 
        rng_key=new_rng, compression_n_samples=None, postproc_f=None, do_sample=True, num_beams=1, 
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

    print('evaluating student before distillation ...')

    rng_key, new_rng = jax.random.split(rng_key)
    acc, all_results = control_evaluate(injection_data, teacher_eval=False, 
                                    inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                    max_input_length=1024, max_output_length=128, rng_key=new_rng, 
                                    do_sample=False, num_beams=1, max_length=128)
    pre_distill_student_accuracies = acc
    print('accuracy:', acc)
    
    with open(os.path.join(data_out_path, 'pre_distill_student_accuracies.pkl'), 'wb') as f:
        pkl.dump(pre_distill_student_accuracies, f)
    
    # distill student
    
    print('distilling ...')

    rng_key, new_rng = jax.random.split(rng_key)
    trainer = distill(distill_data, trainer, mesh, bsize=8, epochs=1, max_input_length=256, rng_key=new_rng, decompress_smoothing_epsilon=None)

    inference.update_params(trainer.params)
    
    if model_out_path is not None:
        model.save_pretrained(
            model_out_path, 
            params=jax.device_get(trainer.params), 
        )

    # eval student

    print('evaluating student after distillation ...')

    rng_key, new_rng = jax.random.split(rng_key)
    acc, all_results = control_evaluate(injection_data, teacher_eval=False, 
                                    inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                    max_input_length=1024, max_output_length=128, rng_key=new_rng, 
                                    do_sample=False, num_beams=1, max_length=128)
    post_distill_student_accuracies = acc
    print('accuracy:', acc)
    
    with open(os.path.join(data_out_path, 'post_distill_student_accuracies.pkl'), 'wb') as f:
        pkl.dump(post_distill_student_accuracies, f)
    
    print('summary:')
    print('teacher accuracy:', teacher_accuracies)
    print('pre-distill student accuracy:', pre_distill_student_accuracies)
    print('post-distill student accuracy:', post_distill_student_accuracies)

    del trainer
    del inference
    del model
    del mesh
    trainer, inference, model, mesh = None, None, None, None


