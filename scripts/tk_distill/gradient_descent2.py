import random
from micro_config import MetaConfig
from base_configs import project_root, AdamWConfig
from gradient_descent import generate_gradient_descent_prompt_data
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
    model_checkpoint = 'outputs/T5_11B_random_nat_inst_finetune_test2/model/'

    bsize = 8
    max_epochs = 25
    add_description = True

    tasks = [
        ('task1391_winogrande_easy_answer_generation', 1), 
        ('task1391_winogrande_easy_answer_generation', 2), 
        ('task1391_winogrande_easy_answer_generation', 4), 
        ('task1391_winogrande_easy_answer_generation', 8), 
        ('task1391_winogrande_easy_answer_generation', 16), 
        ('task1391_winogrande_easy_answer_generation', 32), 

        ('task1394_meta_woz_task_classification', 1), 
        ('task1394_meta_woz_task_classification', 2), 
        ('task1394_meta_woz_task_classification', 4), 
        ('task1394_meta_woz_task_classification', 8), 
        ('task1394_meta_woz_task_classification', 16), 
        ('task1394_meta_woz_task_classification', 32), 

        ('task202_mnli_contradiction_classification', 1), 
        ('task202_mnli_contradiction_classification', 2), 
        ('task202_mnli_contradiction_classification', 4), 
        ('task202_mnli_contradiction_classification', 8), 
        ('task202_mnli_contradiction_classification', 16), 
        ('task202_mnli_contradiction_classification', 32), 

        ('task1155_bard_analogical_reasoning_trash_or_treasure', 1), 
        ('task1155_bard_analogical_reasoning_trash_or_treasure', 2), 
        ('task1155_bard_analogical_reasoning_trash_or_treasure', 4), 
        ('task1155_bard_analogical_reasoning_trash_or_treasure', 8), 
        ('task1155_bard_analogical_reasoning_trash_or_treasure', 16), 
        ('task1155_bard_analogical_reasoning_trash_or_treasure', 32), 

        ('task935_defeasible_nli_atomic_classification', 1), 
        ('task935_defeasible_nli_atomic_classification', 2), 
        ('task935_defeasible_nli_atomic_classification', 4), 
        ('task935_defeasible_nli_atomic_classification', 8), 
        ('task935_defeasible_nli_atomic_classification', 16), 
        ('task935_defeasible_nli_atomic_classification', 32), 

        ('task1386_anli_r2_entailment', 1), 
        ('task1386_anli_r2_entailment', 2), 
        ('task1386_anli_r2_entailment', 4), 
        ('task1386_anli_r2_entailment', 8), 
        ('task1386_anli_r2_entailment', 16), 
        ('task1386_anli_r2_entailment', 32), 

        ('task828_copa_commonsense_cause_effect', 1), 
        ('task828_copa_commonsense_cause_effect', 2), 
        ('task828_copa_commonsense_cause_effect', 4), 
        ('task828_copa_commonsense_cause_effect', 8), 
        ('task828_copa_commonsense_cause_effect', 16), 
        ('task828_copa_commonsense_cause_effect', 32), 

        ('task1624_disfl_qa_question_yesno_classification', 1), 
        ('task1624_disfl_qa_question_yesno_classification', 2), 
        ('task1624_disfl_qa_question_yesno_classification', 4), 
        ('task1624_disfl_qa_question_yesno_classification', 8), 
        ('task1624_disfl_qa_question_yesno_classification', 16), 
        ('task1624_disfl_qa_question_yesno_classification', 32), 

        ('task232_iirc_link_number_classification', 1), 
        ('task232_iirc_link_number_classification', 2), 
        ('task232_iirc_link_number_classification', 4), 
        ('task232_iirc_link_number_classification', 8), 
        ('task232_iirc_link_number_classification', 16), 
        ('task232_iirc_link_number_classification', 32), 

        ('task738_perspectrum_classification', 1), 
        ('task738_perspectrum_classification', 2), 
        ('task738_perspectrum_classification', 4), 
        ('task738_perspectrum_classification', 8), 
        ('task738_perspectrum_classification', 16), 
        ('task738_perspectrum_classification', 32), 

        ('task362_spolin_yesand_prompt_response_sub_classification', 1), 
        ('task362_spolin_yesand_prompt_response_sub_classification', 2), 
        ('task362_spolin_yesand_prompt_response_sub_classification', 4), 
        ('task362_spolin_yesand_prompt_response_sub_classification', 8), 
        ('task362_spolin_yesand_prompt_response_sub_classification', 16), 
        ('task362_spolin_yesand_prompt_response_sub_classification', 32), 

        ('task1393_superglue_copa_text_completion', 1), 
        ('task1393_superglue_copa_text_completion', 2), 
        ('task1393_superglue_copa_text_completion', 4), 
        ('task1393_superglue_copa_text_completion', 8), 
        ('task1393_superglue_copa_text_completion', 16), 
        ('task1393_superglue_copa_text_completion', 32), 

        ('task1154_bard_analogical_reasoning_travel', 1), 
        ('task1154_bard_analogical_reasoning_travel', 2), 
        ('task1154_bard_analogical_reasoning_travel', 4), 
        ('task1154_bard_analogical_reasoning_travel', 8), 
        ('task1154_bard_analogical_reasoning_travel', 16), 
        ('task1154_bard_analogical_reasoning_travel', 32), 
    ]
    
    for task, n_training_data in tasks:
        print('RUNNING:')
        print('='*25)
        print(task, n_training_data)
        print('='*25)

        data_out_path = "../../outputs/injectionpromptvgradientdescent/tk_gradient_descent_%d_datapoints_add_description_%s____%s/" % (n_training_data, str(add_description), task)
        
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
                grad_accum_steps=2 if n_training_data >= 16 else 1, 
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
        
        injection_datas, grad_descent_eval, grad_descent_in, grad_descent_out = generate_gradient_descent_prompt_data(50, 50, 3, 0, False, add_description)

        injection_datas = injection_datas[task]
        grad_descent_eval = grad_descent_eval[task]
        grad_descent_in = grad_descent_in[task][:n_training_data]
        grad_descent_out = grad_descent_out[task][:n_training_data]
        
        with open(os.path.join(data_out_path, 'eval_data.pkl'), 'wb') as f:
            pkl.dump(grad_descent_eval, f)
        
        with open(os.path.join(data_out_path, 'grad_descent_in_out.pkl'), 'wb') as f:
            pkl.dump({'grad_descent_in': grad_descent_in, 'grad_descent_out': grad_descent_out}, f)

        # eval teacher
        
        print('evaluating teacher ...')

        all_teacher_accuracies = []
        
        for injection_data in injection_datas:

            teacher_accuracies = {}

            print('task:', task)
            rng_key, new_rng = jax.random.split(rng_key)
            acc, all_results = tk_evaluate(injection_data, teacher_eval=True, 
                                        inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                        max_input_length=1024, rng_key=new_rng, 
                                        do_sample=False, num_beams=1, max_length=128)
            teacher_accuracies[task] = acc
            print('accuracy:', acc)
        
            all_teacher_accuracies.append(teacher_accuracies)
    
        with open(os.path.join(data_out_path, 'all_teacher_accuracies.pkl'), 'wb') as f:
            pkl.dump(all_teacher_accuracies, f)
        
        # eval student

        print('evaluating student before distillation ...')

        pre_distill_model_accuracies = {}
        
        print('task:', task)
        rng_key, new_rng = jax.random.split(rng_key)
        acc, all_results = tk_evaluate(grad_descent_eval, teacher_eval=True, 
                                       inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                       max_input_length=1024, rng_key=new_rng, 
                                       do_sample=False, num_beams=1, max_length=128)
        pre_distill_model_accuracies[task] = acc
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
            
            print('task:', task)
            rng_key, new_rng = jax.random.split(rng_key)
            acc, all_results = tk_evaluate(grad_descent_eval, teacher_eval=True, 
                                           inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                           max_input_length=1024, rng_key=new_rng, 
                                           do_sample=False, num_beams=1, max_length=128)
            post_distill_model_accuracies[task] = acc
            print('accuracy:', acc)
            
            all_post_distill_accuracies.append(post_distill_model_accuracies)
            with open(os.path.join(data_out_path, 'post_distill_model_accuracies_epoch_%d.pkl' % (x)), 'wb') as f:
                pkl.dump(post_distill_model_accuracies, f)
            
        print('summary:')
        print('teacher accuracy:', all_teacher_accuracies)
        print('pre-train accuracy:', pre_distill_model_accuracies)
        print('post-train accuracy:', all_post_distill_accuracies)

        del trainer
        del inference
        del model
        del mesh
        trainer, inference, model, mesh = None, None, None, None
