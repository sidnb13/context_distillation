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
from injection_functions import distill, format_input, generate_distillation_data, random_token_questions, tk_generate_questions
import pickle as pkl
import jax
import json
import os
import numpy as np
from transformers import T5Tokenizer
from tk_inject import generate_tk_instance, tk_evaluate
from utils.randomness import RandomState, seed_context
import wandb

if __name__ == "__main__":
    
    # setup
    
    model_out_path = None
    add_description = True
    n_per_prompt = 8
    n_prompts = 2

    prompt_sets, dev_set = load_spider('../../data/spider/dev.json', n_per_prompt, n_prompts, 0)
    # prompt_sets = prompt_sets[:n_prompts]
    injection_datas = create_spider_injection_data(prompt_sets, dev_set, '../../data/spider/db_id2schema_2.pkl', add_description, grad_descent_eval_mode=False)

    dbs = [
        # 'concert_singer', 
        # 'orchestra', 
        # # 'battle_death', 
        # 'pets_1', 
        # 'car_1', 
        # 'poker_player', 
        # 'concert_singer', 
        # # 'real_estate_properties', 
        # # 'course_teach', 
        # # 'singer', 
        # 'cre_Doc_Template_Mgt', 
        'student_transcripts_tracking', 
        # 'tvshow', 
        # 'employee_hire_evaluation', 
        # # 'voter_1', 
        # 'flight_2', 
        'world_1', 
        # 'museum_visit', 
        # 'wta_1', 
        # 'network_1', 
        'dog_kennels', 
    ]

    # dbs = dbs[:(len(dbs) // 2)]

    # dbs = dbs[(len(dbs) // 2):]
    
    with open('../../data/spider/db_id2schema_2.pkl', 'rb') as f:
        prompt_schema = pkl.load(f)
    
    with open('../../data/spider/dev.json', 'r') as f:
        raw_data = json.load(f)

    rng_key = jax.random.PRNGKey(0)

    in_length = 1024+512+(512-208)
    out_length = 208
    model_str = 'facebook/incoder-6B'

    for db in dbs:

        injection_data_set = injection_datas[db]
        
        print('loading model ...')

        metaconfig = MetaConfig(
            project_root=project_root, 
            verbose=False, 
        )

        model_config = IncoderModelConfig(
            # model_str="google/t5-v1_1-xl", 
            # model_str="t5-3b", 
            # model_str="google/ul2", 
            model_str=model_str, 
            # model_str="allenai/tk-instruct-11b-def-pos-neg-expl", 
            checkpoint_path=None, 
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
        
        print('generating questions ...')

        print('db:', db)
        rng_key, new_rng = jax.random.split(rng_key)
        questions = tk_generate_questions(injection_data_set[0], inference=inference, mesh=mesh, 
                                        bsize=32, n_questions=4*1024*len(injection_data_set), 
                                        max_input_length=in_length, rng_key=new_rng, 
                                        do_sample=True, num_beams=1, max_length=in_length+out_length, 
                                        pad_token_id=tokenizer.pad_token_id, 
                                        eos_token_id=tokenizer.encode('\n')[0], 
                                        temperature=1.0)
        questions = list(map(lambda x: x[len(' Now complete the following example - Input: '):-len(' Output: ')], questions))
        
        # combined_distill_data = sum(distill_items)
        # injection_data = injection_data_set[0]
        distill_data = []
        
        for curr_injection_idx, injection_data in enumerate(injection_data_set):

            data_out_path = "../../outputs/incoder_inject_spider_data_combined_from_cache_test2/%s_%d/" % (db, curr_injection_idx)
            random_state = RandomState(0)
            rng_key = jax.random.PRNGKey(0)
            # rng_key, new_key = jax.random.split(rng_key)
            # random_state = RandomState(jax.random.randint(new_key, [], 0, 2**30).item())
            
            if model_out_path is not None:
                if not os.path.exists(os.path.dirname(model_out_path)):
                    os.makedirs(os.path.dirname(model_out_path))
            if not os.path.exists(os.path.dirname(data_out_path)):
                os.makedirs(os.path.dirname(data_out_path))

            # # simotaneous injection
            # distill_data = sum(distill_items[:(curr_injection_idx+1)], [])

            all_idx_outputs = defaultdict(list)

            print('loading model ...')

            metaconfig = MetaConfig(
                project_root=project_root, 
                verbose=False, 
            )

            model_config = IncoderModelConfig(
                # model_str="google/t5-v1_1-xl", 
                # model_str="t5-3b", 
                # model_str="google/ul2", 
                model_str=model_str, 
                # model_str="allenai/tk-instruct-11b-def-pos-neg-expl", 
                checkpoint_path=None, 
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
                                            max_input_length=in_length, rng_key=new_rng, 
                                            do_sample=False, num_beams=1, max_length=in_length+out_length, 
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
            
            # distillation data generation
        
            print('generating distillation data ...')

            print('db:', db)
            rng_key, new_rng = jax.random.split(rng_key)
            distill_data += generate_distillation_data(
                injection_data, questions[(curr_injection_idx)*(len(questions) // len(injection_data_set)):(curr_injection_idx+1)*(len(questions) // len(injection_data_set))], 
                inference=inference, mesh=mesh, 
                bsize=32, n_per_question=1, max_input_length=in_length, max_output_length=out_length, 
                rng_key=new_rng, compression_n_samples=100, 
                postproc_f=lambda x: x + '\n\n' if not x.endswith('\n\n') else x, 
                do_sample=True, num_beams=1, max_length=in_length+out_length, 
                pad_token_id=tokenizer.pad_token_id, 
                eos_token_id=tokenizer.encode('\n\n')[0], 
            )


            print('evaluating student before distillation ...')

            pre_distill_model_accuracies = {}
            
            print('db:', db)
            rng_key, new_rng = jax.random.split(rng_key)
            acc, all_results = tk_evaluate(injection_data, teacher_eval=False, 
                                            inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                            max_input_length=in_length, rng_key=new_rng, 
                                            do_sample=False, num_beams=1, max_length=in_length+out_length, 
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
        
            if jax.process_index() == 0:
                wandb_state = wandb.init(project="distill_spider", name="%s_%d" % (db, curr_injection_idx), reinit=True)

            for epoch in range(2):
                
                print('distilling ...')

                rng_key, new_rng = jax.random.split(rng_key)
                trainer = distill(distill_data, trainer, mesh, bsize=4, epochs=1, max_input_length=in_length, rng_key=new_rng, decompress_smoothing_epsilon=1e-7)

                inference.update_params(trainer.params)

                # eval student

                print('evaluating student after distillation ...')

                post_distill_student_accuracies = {}
                
                print('db:', db)
                rng_key, new_rng = jax.random.split(rng_key)
                acc, all_results = tk_evaluate(injection_data, teacher_eval=False, 
                                            inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                            max_input_length=in_length, rng_key=new_rng, 
                                            do_sample=False, num_beams=1, max_length=in_length+out_length, 
                                            pad_token_id=tokenizer.pad_token_id, 
                                            eos_token_id=tokenizer.encode('\n\n')[0])
                post_distill_student_accuracies[db] = acc
                print(acc)
                if jax.process_index() == 0:
                    wandb.log(acc)
                
                with open(os.path.join(data_out_path, f'post_distill_student_accuracies_epoch_{epoch}.pkl'), 'wb') as f:
                    pkl.dump(post_distill_student_accuracies, f)
                
                for z, (q, refs) in enumerate(injection_data.dataset_eval):
                    ref = refs[0]
                    for idx, item2 in enumerate(raw_data):
                        if q.strip() == item2['question'] and ref == item2['query']:
                            all_idx_outputs[f'post_distill_student_{curr_injection_idx}_epoch_{epoch}'].append((idx, all_results[z]['generation'].strip()))
                            break
            
            if jax.process_index() == 0:
                wandb.finish()
            
            with open(os.path.join(data_out_path, 'all_idx_outputs.pkl'), 'wb') as f:
                pkl.dump(all_idx_outputs, f)
            
            print('summary:')
            print('pre-distill accuracy:', pre_distill_model_accuracies)
            print('post-distill accuracy:', post_distill_student_accuracies)

            del trainer
            del inference
            del model
            del mesh
            trainer, inference, model, mesh, = None, None, None, None
