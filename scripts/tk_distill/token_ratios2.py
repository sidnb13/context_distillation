from transformers import T5Tokenizer
import jax
from utils.randomness import RandomState, seed_context
from copy import copy
from nat_inst.ni_formatter import get_formatted_ni_data
from explanations import generate_tk_instance2
from tk_inject import generate_tk_instance
from task_assoc import generate_task_association_data

tokenizer = T5Tokenizer.from_pretrained('google/t5-xxl-lm-adapt')

tasks = [
        ('task1631_openpi_answer_generation', True, True), 
        ('task039_qasc_find_overlapping_words', True, True), 
        ('task1157_bard_analogical_reasoning_rooms_for_containers', True, True), 
        ('task1158_bard_analogical_reasoning_manipulating_items', True, True), 
        ('task1516_imppres_naturallanguageinference', True, True), 
        
        ('task034_winogrande_question_modification_object', True, True), 
        ('task1540_parsed_pdfs_summarization', True, True), 
        ('task418_persent_title_generation', True, True), 
        ('task401_numeric_fused_head_reference', True, True), 
        ('task891_gap_coreference_resolution', True, True), 

        # ('task1624_disfl_qa_question_yesno_classification', True, False), 
        # ('task970_sherliic_causal_relationship', True, False), 
        # ('task1516_imppres_naturallanguageinference', True, False), 
        # ('task1195_disflqa_disfluent_to_fluent_conversion', True, False), 
        # ('task362_spolin_yesand_prompt_response_sub_classification', True, False), 

        # ('task1586_scifact_title_generation', True, False), 
        # ('task743_eurlex_summarization', True, False), 
        # ('task1155_bard_analogical_reasoning_trash_or_treasure', True, False), 
        # ('task033_winogrande_answer_generation', True, False), 
        # ('task937_defeasible_nli_social_classification', True, False), 
    ]


random_state = RandomState(0)
rng_key = jax.random.PRNGKey(0)

random_state_copy = copy(random_state)
        
with seed_context(random_state):
    formatted_train_data, formatted_test_data = get_formatted_ni_data(
        add_task_name=False, add_task_definition=True, 
        num_pos_examples=2, num_neg_examples=2, 
        add_explanation=True, 
        max_num_instances_per_task=100, 
        max_num_instances_per_eval_task=None, 
    )

with seed_context(random_state_copy):
    formatted_train_data_no_expl, formatted_test_data_no_expl = get_formatted_ni_data(
        add_task_name=False, add_task_definition=True, 
        num_pos_examples=2, num_neg_examples=2, 
        add_explanation=False, 
        max_num_instances_per_task=100, 
        max_num_instances_per_eval_task=None, 
    )

breakpoint()
pass
