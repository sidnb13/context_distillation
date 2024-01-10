import random
from micro_config import MetaConfig
from base_configs import project_root, AdamWConfig
from gradient_descent import generate_gradient_descent_prompt_data
from spider_data import create_spider_injection_data, load_spider
from t5_config import T5ModelConfig
from core import TKInferenceConfig
import contextlib
from core import TKServerInference, TKTrainConfig
from nat_inst.ni_formatter import get_formatted_ni_data
from task_assoc import generate_task_association_data, get_binary_tasks
from injection_functions import distill, format_input, generate_distillation_data, random_token_questions, tk_generate_questions
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
    model_checkpoint = 'outputs/T5_11B_random_nat_inst_finetune_test2/model/'
    add_description = True

    prompt_split, _ = load_spider('../../data/spider/dev.json', 3, 0)
    injection_datas = create_spider_injection_data('../../data/spider/dev.json', 3, 0, add_description, grad_descent_eval_mode=False)
    gradient_descent_injection_datas = create_spider_injection_data('../../data/spider/dev.json', 3, 0, add_description, grad_descent_eval_mode=True)

    rng_key = jax.random.PRNGKey(0)

    bsize = 8
    max_epochs = 25
    
    for db, injection_data in injection_datas.items():
        print('RUNNING:')
        print('='*25)
        print(db)
        print('='*25)

        data_out_path = "../../outputs/injectionpromptvgradientdescent_spider/tk_gradient_descent_add_description_%s____%s/" % (str(add_description), db)
        
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

        grad_descent_in, grad_descent_out = list(zip(*prompt_split[db]))
        if add_description:
            description = injection_data.teacher_prompt[:injection_data.teacher_prompt.find('Positive Example 1')].strip()
            grad_descent_in = list(map(lambda x: description+format_input(x), grad_descent_in))
        else:
            grad_descent_in = list(map(lambda x: format_input(x), grad_descent_in))
        
        grad_descent_eval = gradient_descent_injection_datas[db]
        
        with open(os.path.join(data_out_path, 'grad_descent_eval.pkl'), 'wb') as f:
            pkl.dump(grad_descent_eval, f)

        with open(os.path.join(data_out_path, 'grad_descent_in_out.pkl'), 'wb') as f:
            pkl.dump({'grad_descent_in': grad_descent_in, 'grad_descent_out': grad_descent_out}, f)

        # eval teacher

        print('evaluating teacher ...')

        teacher_accuracies = {}

        print('db:', db)
        rng_key, new_rng = jax.random.split(rng_key)
        acc, all_results = tk_evaluate(injection_data, teacher_eval=True, 
                                    inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                    max_input_length=1024, rng_key=new_rng, 
                                    do_sample=False, num_beams=1, max_length=128)
        teacher_accuracies[db] = acc
        print('accuracy:', acc)
    
        with open(os.path.join(data_out_path, 'teacher_accuracies.pkl'), 'wb') as f:
            pkl.dump(teacher_accuracies, f)
        
        # eval student

        print('evaluating student before distillation ...')

        pre_distill_model_accuracies = {}
        
        print('db:', db)
        rng_key, new_rng = jax.random.split(rng_key)
        acc, all_results = tk_evaluate(grad_descent_eval, teacher_eval=True, 
                                       inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                       max_input_length=1024, rng_key=new_rng, 
                                       do_sample=False, num_beams=1, max_length=128)
        pre_distill_model_accuracies[db] = acc
        print('accuracy:', acc)
        
        with open(os.path.join(data_out_path, 'pre_distill_student_accuracies.pkl'), 'wb') as f:
            pkl.dump(pre_distill_model_accuracies, f)
        
        # train model

        all_post_distill_accuracies = []
        
        for x in range(max_epochs):
            with mesh:
                
                # shuffle training data
                with seed_context(random_state):
                    idxs = list(range(len(grad_descent_in)))
                    random.shuffle(idxs)
                    grad_descent_in = [grad_descent_in[i] for i in idxs]
                    grad_descent_out = [grad_descent_out[i] for i in idxs]

                # train for 1 epoch
                for i in range(0, len(grad_descent_in), bsize):
                    input_questions = grad_descent_in[i:(i+bsize)]
                    output_answers = grad_descent_out[i:(i+bsize)]

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
            
            print('db:', db)
            rng_key, new_rng = jax.random.split(rng_key)
            acc, all_results = tk_evaluate(grad_descent_eval, teacher_eval=True, 
                                           inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                           max_input_length=1024, rng_key=new_rng, 
                                           do_sample=False, num_beams=1, max_length=128)
            post_distill_model_accuracies[db] = acc
            print('accuracy:', acc)
            
            all_post_distill_accuracies.append(post_distill_model_accuracies)
            with open(os.path.join(data_out_path, 'post_distill_model_accuracies_epoch_%d.pkl' % (x)), 'wb') as f:
                pkl.dump(post_distill_model_accuracies, f)
            
        print('summary:')
        print('teacher accuracy:', teacher_accuracies)
        print('pre-train accuracy:', pre_distill_model_accuracies)
        print('post-train accuracy:', all_post_distill_accuracies)

        del trainer
        del inference
        del model
        del mesh
        trainer, inference, model, mesh = None, None, None, None
