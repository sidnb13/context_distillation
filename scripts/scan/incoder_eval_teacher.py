import random
from micro_config import MetaConfig
from base_configs import project_root, AdamWConfig
from incoder_config import IncoderModelConfig
from incoder_core import IncoderInferenceConfig
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
from incoder_scan_data import create_scan_cirriculum
from utils.randomness import RandomState, seed_context

if __name__ == "__main__":
    
    # setup
    
    teacher_checkpoint = None
    
    cirriculum, eval_data = create_scan_cirriculum(add_explanation=True, n_positive=2, seed=0)
    
    random_state = RandomState(0)
    rng_key = jax.random.PRNGKey(0)

    print(eval_data.teacher_prompt)

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
    
    inference_config = IncoderInferenceConfig(
        model=model_config, 
        pjit=True, 
        verbose=True, 
    )

    inference, model, mesh = inference_config.unroll(metaconfig)

    _, _, tokenizer, _ = model_config.unroll(metaconfig)
    
    # eval teacher

    for i, item in enumerate(cirriculum):

        if len(item.dataset_eval) > 0:
            print('evaluating teacher ...', i)

            rng_key, new_rng = jax.random.split(rng_key)
            acc, all_results = tk_evaluate(item, teacher_eval=True, 
                                        inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                        max_input_length=512, rng_key=new_rng, 
                                        do_sample=False, num_beams=1, max_length=1024, 
                                        pad_token_id=tokenizer.pad_token_id, 
                                        eos_token_id=tokenizer.encode('\n')[0])
            print('accuracy:', acc)
    
    print('evaluating teacher final ...')
    
    rng_key, new_rng = jax.random.split(rng_key)
    acc, all_results = tk_evaluate(eval_data, teacher_eval=True, 
                                   inference=inference, mesh=mesh, bsize=1, num_instances=None, 
                                   max_input_length=512, rng_key=new_rng, 
                                   do_sample=False, num_beams=1, max_length=1024, 
                                   pad_token_id=tokenizer.pad_token_id, 
                                   eos_token_id=tokenizer.encode('\n')[0])
    print('accuracy:', acc)
