import contextlib
from micro_config import MetaConfig
from base_configs import AdamWConfig, project_root
from fact_edit import counterfact_evaluate, counterfact_teacher_train, generate_counterfact_instance, shuffle_countefact_data
from t5_config import T5ModelConfig
from core import TKServerInference, TKInferenceConfig, TKTrainConfig
from nat_inst.ni_formatter import get_formatted_ni_data
from task_assoc import generate_task_association_data, get_binary_tasks
import pickle as pkl
import jax
import json
from tk_inject import generate_tk_instance, tk_evaluate
from utils.randomness import RandomState, seed_context
import random
import tree

if __name__ == "__main__":
    seed = 0
    random_state = RandomState(seed)
    rng_key = jax.random.PRNGKey(0)

    with seed_context(random_state):
        with open('../../data/counterfact/counterfact_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open('../../data/counterfact/new_counterfact_data.json', 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        eval_ids = set(map(lambda x: x['case_id'], eval_data))
        
        data = [item for item in data if item['case_id'] not in eval_ids]
        
        injection_data = []
        for i in range(len(data)):
            # shuffle_countefact_data(data[i])
            injection_data.append(generate_counterfact_instance(data[i], new_data=False))
    
    instances = []
    for item in injection_data:
        instances.extend([(item.teacher_prompt, inputs, corrects) for inputs, corrects, _, _ in item.dataset_eval])

    with seed_context(random_state):
        random.shuffle(instances)
    
    instances = instances[:10000]    
    
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
            checkpoint_path='outputs/T5_11B_random_nat_inst_finetune_test2/model/', 
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

    trainer = counterfact_teacher_train(instances, trainer=trainer, 
                                        mesh=mesh, bsize=8, 
                                        max_input_length=1024, max_output_length=128, 
                                        rng_key=rng_key)
    
    model.save_pretrained(
        save_directory=metaconfig.convert_path('outputs/trained_counterfact_teacher/model/'), 
        params=trainer.params, 
    )
