import contextlib
from micro_config import MetaConfig
from base_configs import project_root
from t5_config import T5ModelConfig
from core import TKServerInference, TKInferenceConfig
from nat_inst.ni_formatter import get_formatted_ni_data
from task_assoc import generate_task_association_data, get_binary_tasks
import pickle as pkl
import jax
import json
from tk_inject import generate_tk_instance, tk_evaluate
from utils.randomness import RandomState, seed_context

if __name__ == "__main__":
    random_state = RandomState(0)
    rng_key = jax.random.PRNGKey(0)

    
    injection_data = {}

    with seed_context(random_state):
        formatted_train_data, formatted_test_data = get_formatted_ni_data(
            add_task_name=False, add_task_definition=True, num_pos_examples=2, 
            num_neg_examples=2, add_explanation=False, max_num_instances_per_task=100, 
            max_num_instances_per_eval_task=100, 
        )

        for task, data in formatted_test_data.items():
            injection_data[task] = generate_tk_instance(data)

