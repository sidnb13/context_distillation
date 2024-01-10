from itertools import chain, permutations
from typing import Callable, List, Dict, Any, Optional, Tuple
import random
from core import TKInference
from injection_functions import format_input, InjectionData
import jax
from jax.experimental.maps import Mesh
from tqdm.auto import tqdm
from reasoning_distill import ReasoningData
from tk_jax.compute_metrics import compute_metrics
import re

def generate_drop_instance(task_examples: List[Dict[str, Any]], has_rationale: bool, extraction_f: Callable[[str], Optional[str]]) -> InjectionData:
    return ReasoningData(
        teacher_prompt="Lets think step by step.", 
        student_prompt="", 
        question_prompts=[""], 
        questions=[f" Context: {item['processed c']} Question: {item['processed q']}" for item in task_examples], 
        scratchpad_answers=[item['processed a'] for item in task_examples] if has_rationale else None, 
        direct_answers=[extraction_f(item['processed a']) for item in task_examples] if has_rationale else [item['processed a'] for item in task_examples], 
        extract_answer_function=extraction_f, 
    )

def extract_drop_answer(completion: str) -> Optional[str]:
    search_result = re.search(r'\#(.+)\#', completion)
    if search_result:
        s = search_result.group(0)
        for c in '# -':
            s = s.replace(c, '')
        return s
    return None
