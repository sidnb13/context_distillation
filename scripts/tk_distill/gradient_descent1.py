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
from tk_inject import generate_tk_instance, tk_evaluate, format_input

from utils.randomness import RandomState, seed_context

if __name__ == "__main__":
    
    # setup
    
    # data_out_path = "../../outputs/task_assoc_dataset_data_test2/"
    data_out_path = "../../outputs/tk_grad_descent___task879_schema_guided_dstc8_classification/"
    model_out_path = "../../outputs/tk_grad_descent___task879_schema_guided_dstc8_classification/model/"
    # model_out_path = None
    model_checkpoint = 'outputs/T5_11B_random_nat_inst_finetune_test2/model/'

    # task = 'task879_schema_guided_dstc8_classification'
    task = 'task1391_winogrande_easy_answer_generation'
    
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
            checkpoint_path=model_checkpoint, 
            from_pretrained=True, 
            use_fp16=True, 
            gradient_checkpoint=True, 
        ), 
        optim=AdamWConfig(
            grad_accum_steps=1, 
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
            add_task_name=False, add_task_definition=False, num_pos_examples=2, 
            num_neg_examples=0, add_explanation=False, max_num_instances_per_task=100, 
            max_num_instances_per_eval_task=100, 
        )

        injection_data = generate_tk_instance(formatted_test_data[task])
        input_questions = [format_input(item['input']) for item in formatted_test_data[task][0]['prompt_positive_examples']]
        output_answers = [item['output'] for item in formatted_test_data[task][0]['prompt_positive_examples']]
    
    with open(os.path.join(data_out_path, 'injection_data.pkl'), 'wb') as f:
        pkl.dump({'injection_data': injection_data, 'input_questions': input_questions, 'output_answers': output_answers}, f)

    
    # eval model

    print('evaluating model before distillation ...')

    pre_distill_model_accuracies = {}
    
    print('task:', task)
    rng_key, new_rng = jax.random.split(rng_key)
    acc, all_results = tk_evaluate(injection_data, teacher_eval=False, 
                                    inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                    max_input_length=1024, rng_key=new_rng, 
                                    do_sample=False, num_beams=1, max_length=128)
    pre_distill_model_accuracies[task] = acc
    print('accuracy:', acc)
    
    with open(os.path.join(data_out_path, 'pre_distill_model_accuracies.pkl'), 'wb') as f:
        pkl.dump(pre_distill_model_accuracies, f)
    
    # train model

    for x in range(50):
        with mesh:
            for i in range(1):
                rng_key, new_key = jax.random.split(rng_key)
                loss = trainer.train_step_from_str(
                    in_strs=input_questions, 
                    out_strs=output_answers, 
                    max_input_length=1024, 
                    max_output_length=128, 
                    rng_key=new_key, 
                )
                print(f'step: {i+1} loss: {loss}')
        
        inference.update_params(trainer.params)
        
        # eval student

        print('evaluating model after distillation ...')

        post_distill_model_accuracies = {}
        
        print('task:', task)
        rng_key, new_rng = jax.random.split(rng_key)
        acc, all_results = tk_evaluate(injection_data, teacher_eval=False, 
                                        inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                        max_input_length=1024, rng_key=new_rng, 
                                        do_sample=False, num_beams=1, max_length=128)
        post_distill_model_accuracies[task] = acc
        print('accuracy:', acc)
        
        with open(os.path.join(data_out_path, 'post_distill_model_accuracies_%d.pkl' % (x)), 'wb') as f:
            pkl.dump(post_distill_model_accuracies, f)
        
        print('summary:')
        print('pre-train accuracy:', pre_distill_model_accuracies[task])
        print('post-train accuracy:', post_distill_model_accuracies[task])
