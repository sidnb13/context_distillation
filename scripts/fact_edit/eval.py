import contextlib
from micro_config import MetaConfig
from base_configs import project_root
from fact_edit import counterfact_evaluate, generate_counterfact_instance, shuffle_countefact_data
from t5_config import T5ModelConfig
from core import TKServerInference, TKInferenceConfig
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

    with seed_context(random_state):
        with open('../../data/counterfact/new_counterfact_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        injection_data = []
        for i in range(len(data)):
            # shuffle_countefact_data(data[i])
            injection_data.append(generate_counterfact_instance(data[i], new_data=True))
    
    metaconfig = MetaConfig(
        project_root=project_root, 
        verbose=False, 
    )
    
    inference_config = TKInferenceConfig(
        model=T5ModelConfig(
            # model_str="google/t5-v1_1-xl", 
            # model_str="t5-3b", 
            # model_str="google/ul2", 
            model_str="google/t5-xxl-lm-adapt", 
            # model_str="allenai/tk-instruct-11b-def-pos-neg-expl", 
            checkpoint_path='outputs/trained_counterfact_teacher/model/', 
            from_pretrained=True, 
            use_fp16=True, 
            gradient_checkpoint=False, 
        ), 
        pjit=True, 
        verbose=True, 
    )

    inference, _, mesh = inference_config.unroll(metaconfig)

    # inference = TKServerInference('http://34.133.90.23:8000/')
    # mesh = contextlib.nullcontext()
    
    all_items = []
    for item in injection_data:
        print('task:', item.student_prompt)
        acc, all_results = counterfact_evaluate(item, teacher_eval=False, 
                                                inference=inference, mesh=mesh, bsize=1, 
                                                max_input_length=1024, max_output_length=128)
        all_items.append(acc)
        print('accuracy:', acc)
        print('avg:', tree.map_structure(lambda *x: sum(x) / len(x), *all_items))
