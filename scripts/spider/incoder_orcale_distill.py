from collections import defaultdict
import random
from micro_config import MetaConfig
from base_configs import project_root, AdamWConfig
from gradient_descent import generate_gradient_descent_prompt_data
from incoder_config import IncoderModelConfig
from incoder_spider_data import create_spider_injection_data, load_spider
from incoder_core import IncoderInferenceConfig, IncoderTrainConfig
import contextlib
from nat_inst.ni_formatter import get_formatted_ni_data
from task_assoc import generate_task_association_data, get_binary_tasks
from injection_functions import distill, format_input, generate_distillation_data, oracle_distillation_data, random_token_questions, tk_generate_questions
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
    
    model_out_path = "../../outputs/spider_test2_model/model/"
    student_checkpoint = None
    teacher_checkpoint = None
    add_description = True
    n_per_prompt = 4
    n_prompts = 4

    prompt_split, _ = load_spider('../../data/spider/dev.json', n_per_prompt, n_prompts, 0)
    injection_datas = create_spider_injection_data('../../data/spider/dev.json', '../../data/spider/db_id2schema.pkl', n_per_prompt, n_prompts, 0, add_description, grad_descent_eval_mode=False)

    with open('../../data/spider/db_id2schema.pkl', 'rb') as f:
        prompt_schema = pkl.load(f)
    
    with open('../../data/spider/dev.json', 'r') as f:
        raw_data = json.load(f)

    rng_key = jax.random.PRNGKey(0)

    bsize = 8
    max_epochs = 25
    
    for db, injection_data_set in injection_datas.items():
        for curr_injection_idx, injection_data in enumerate(injection_data_set):
            if db != 'flight_2':
                continue

            all_idx_outputs = defaultdict(list)

            print('RUNNING:')
            print('='*25)
            print(db)
            print('='*25)

            data_out_path = "../../outputs/incoder_injectionpromptvgradientdescent_spider_test2/tk_injection_add_description_%s_%d_datapoints_%d_per_prompt____%s/" % (str(add_description), n_per_prompt*(curr_injection_idx+1), n_per_prompt, db)
            
            random_state = RandomState(0)
            rng_key = jax.random.PRNGKey(0)
            # rng_key, new_key = jax.random.split(rng_key)
            # random_state = RandomState(jax.random.randint(new_key, [], 0, 2**30).item())
            
            if model_out_path is not None:
                if not os.path.exists(os.path.dirname(model_out_path)):
                    os.makedirs(os.path.dirname(model_out_path))
            if not os.path.exists(os.path.dirname(data_out_path)):
                os.makedirs(os.path.dirname(data_out_path))
            
            # save config

            os.system(f'cp {__file__} {os.path.join(data_out_path, "config.py")}')

            # load teacher

            print('loading teacher ...')
        
            metaconfig = MetaConfig(
                project_root=project_root, 
                verbose=False, 
            )

            model_config = IncoderModelConfig(
                # model_str="google/t5-v1_1-xl", 
                # model_str="t5-3b", 
                # model_str="google/ul2", 
                model_str='facebook/incoder-6B', 
                # model_str="allenai/tk-instruct-11b-def-pos-neg-expl", 
                checkpoint_path=teacher_checkpoint, 
                from_pretrained=True, 
                use_fp16=True, 
                gradient_checkpoint=True, 
            )
            
            trainer_config = IncoderTrainConfig(
                model=model_config, 
                optim=AdamWConfig(
                    grad_accum_steps=4, 
                    lr=1e-4, 
                    weight_decay=0.00, 
                    beta1=0.9, 
                    beta2=0.999, 
                    eps=1e-6, 
                ), 
                pjit=True, 
                verbose=True, 
            )

            trainer, inference, model, mesh = trainer_config.unroll(metaconfig)

            _, _, tokenizer, _ = model_config.unroll(metaconfig)

            # eval teacher

            print('evaluating teacher ...')

            teacher_accuracies = defaultdict(list)

            print('db:', db)
            rng_key, new_rng = jax.random.split(rng_key)
            acc, all_results = tk_evaluate(injection_data, teacher_eval=True, 
                                            inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                            max_input_length=1024+512, rng_key=new_rng, 
                                            do_sample=False, num_beams=1, max_length=1024+512+512, 
                                            pad_token_id=tokenizer.pad_token_id, 
                                            eos_token_id=tokenizer.encode('\n\n')[0])
            teacher_accuracies[db].append(acc)
            print('accuracy:', acc)

            for z, (q, refs) in enumerate(injection_data.dataset_eval):
                ref = refs[0]
                for idx, item2 in enumerate(raw_data):
                    if q.strip() == item2['question'] and ref == item2['query']:
                        all_idx_outputs[f'teacher_{curr_injection_idx}'].append((idx, all_results[z]['generation'].strip()))
                        break
        
            with open(os.path.join(data_out_path, 'teacher_accuracies.pkl'), 'wb') as f:
                pkl.dump(teacher_accuracies, f)
            
            with open(os.path.join(data_out_path, 'all_idx_outputs.pkl'), 'wb') as f:
                pkl.dump(all_idx_outputs, f)
            
            # question generation

            print('generating questions ...')

            # print('db:', db)
            # rng_key, new_rng = jax.random.split(rng_key)
            # questions = tk_generate_questions(injection_data, inference=inference, mesh=mesh, 
            #                                   bsize=1, n_questions=1024, 
            #                                   max_input_length=1024+512, rng_key=new_rng, 
            #                                   do_sample=True, num_beams=1, max_length=1024+512+512, 
            #                                   pad_token_id=tokenizer.pad_token_id, 
            #                                   eos_token_id=tokenizer.encode('\n')[0], 
            #                                   temperature=0.2)
            # questions = list(map(lambda x: x[len(' Now complete the following example - Input: '):-len(' Output: ')], questions))
            # with open(os.path.join(data_out_path, 'questions.pkl'), 'wb') as f:
            #     pkl.dump(questions, f)
            # questions = [""]
            
            # distillation data generation
        
            print('generating distillation data ...')

            # print('db:', db)
            # rng_key, new_rng = jax.random.split(rng_key)
            # distill_data = generate_distillation_data(
            #     injection_data, questions, inference=inference, mesh=mesh, 
            #     bsize=1, n_per_question=1024, max_input_length=1024+512, max_output_length=512, 
            #     rng_key=new_rng, do_sample=True, num_beams=1, max_length=1024+512+512, 
            #     pad_token_id=tokenizer.pad_token_id, 
            #     # eos_token_id=tokenizer.encode('\n\n')[0], 
            # )

            print('db:', db)
            distill_data = oracle_distillation_data(
                injection_data, 
                questions=[q for q, _ in injection_data.dataset_eval], 
                answers=[f"{a[0]}\n\n" for _, a in injection_data.dataset_eval], 
                inference=inference, mesh=mesh, 
                bsize=1, max_input_length=1024, max_output_length=128, 
            )
            
            # load student

            if student_checkpoint != teacher_checkpoint:
                
                print('loading student ...')
                
                metaconfig = MetaConfig(
                    project_root=project_root, 
                    verbose=False, 
                )
                
                model_config = IncoderModelConfig(
                    # model_str="google/t5-v1_1-xl", 
                    # model_str="t5-3b", 
                    # model_str="google/ul2", 
                    model_str='facebook/incoder-6B', 
                    # model_str="allenai/tk-instruct-11b-def-pos-neg-expl", 
                    checkpoint_path=teacher_checkpoint, 
                    from_pretrained=True, 
                    use_fp16=True, 
                    gradient_checkpoint=True, 
                )
            
                trainer_config = IncoderTrainConfig(
                    model=model_config, 
                    optim=AdamWConfig(
                        grad_accum_steps=4, 
                        lr=1e-4, 
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

            pre_distill_model_accuracies = {}
            
            print('db:', db)
            rng_key, new_rng = jax.random.split(rng_key)
            acc, all_results = tk_evaluate(injection_data, teacher_eval=False, 
                                           inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                           max_input_length=1024+512, rng_key=new_rng, 
                                           do_sample=False, num_beams=1, max_length=1024+512+512, 
                                           pad_token_id=tokenizer.pad_token_id, 
                                           eos_token_id=tokenizer.encode('\n\n')[0])
            pre_distill_model_accuracies[db] = acc
            print('accuracy:', acc)
            
            with open(os.path.join(data_out_path, 'pre_distill_student_accuracies.pkl'), 'wb') as f:
                pkl.dump(pre_distill_model_accuracies, f)
            
            for z, (q, refs) in enumerate(injection_data.dataset_eval):
                ref = refs[0]
                for idx, item2 in enumerate(raw_data):
                    if q.strip() == item2['question'] and ref == item2['query']:
                        all_idx_outputs[f'raw_student_{curr_injection_idx}'].append((idx, all_results[z]['generation'].strip()))
                        break
            
            with open(os.path.join(data_out_path, 'all_idx_outputs.pkl'), 'wb') as f:
                pkl.dump(all_idx_outputs, f)
            
            # distill student
        
            print('distilling ...')

            rng_key, new_rng = jax.random.split(rng_key)
            trainer = distill(distill_data, trainer, mesh, bsize=4, epochs=8, max_input_length=1024+512, rng_key=new_rng)

            inference.update_params(trainer.params)
            
            if model_out_path is not None:
                model.save_pretrained(
                    model_out_path, 
                    params=jax.device_get(trainer.params), 
                )

            # eval student

            print('evaluating student after distillation ...')

            post_distill_student_accuracies = {}
            
            print('db:', db)
            rng_key, new_rng = jax.random.split(rng_key)
            acc, all_results = tk_evaluate(injection_data, teacher_eval=False, 
                                           inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                           max_input_length=1024+512, rng_key=new_rng, 
                                           do_sample=False, num_beams=1, max_length=1024+512+512, 
                                           pad_token_id=tokenizer.pad_token_id, 
                                           eos_token_id=tokenizer.encode('\n\n')[0])
            post_distill_student_accuracies[db] = acc
            print('accuracy:', acc)
            
            with open(os.path.join(data_out_path, 'post_distill_student_accuracies.pkl'), 'wb') as f:
                pkl.dump(post_distill_student_accuracies, f)
            
            for z, (q, refs) in enumerate(injection_data.dataset_eval):
                ref = refs[0]
                for idx, item2 in enumerate(raw_data):
                    if q.strip() == item2['question'] and ref == item2['query']:
                        all_idx_outputs[f'post_distill_student_{curr_injection_idx}'].append((idx, all_results[z]['generation'].strip()))
                        break
            
            with open(os.path.join(data_out_path, 'all_idx_outputs.pkl'), 'wb') as f:
                pkl.dump(all_idx_outputs, f)
                
            print('summary:')
            print('teacher accuracy:', teacher_accuracies)
            print('pre-distill accuracy:', pre_distill_model_accuracies)
            print('post-distill accuracy:', post_distill_student_accuracies)

            del trainer
            del inference
            del model
            del mesh
            trainer, inference, model, mesh = None, None, None, None

            student_checkpoint = model_out_path[model_out_path.find('outputs'):]

