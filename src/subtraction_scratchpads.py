from typing import List, Generator, Dict, Optional
import random
from injection_functions import format_input
from reasoning_distill import ReasoningData
from utils.randomness import RandomState, seed_context
import re

digits_to_str = lambda x: ' '.join(map(str, x))
int_to_digits = lambda x, pad: [0]*max(pad-len(str(x)), 0)+list(map(int, str(x)))
int_to_str = lambda x: digits_to_str(int_to_digits(x, 0))
n_digits = lambda x: len(int_to_digits(x, 0))
def str_to_int(s):
    s = s.replace(" ", "")
    if s.isdigit():
        return int(s)
    return None
def nested_digits_to_str(digits):
	return ' '.join(map(lambda x: str(x) if n_digits(x) <= 1 else f"( {nested_digits_to_str(int_to_digits(x, 0))} )", digits))

def generate_scratchpad(a: int, b: int) -> str:
	n = max(n_digits(a), n_digits(b))
	a_digits, b_digits = int_to_digits(a, n), int_to_digits(b, n)

	scratchpad = f"{digits_to_str(a_digits)} - {digits_to_str(b_digits)} => "

	num = a_digits
	denom = b_digits
	curr_digits = []
	
	for i in range(len(num)-1, -1, -1):
		curr_idx = len(num)-1
		if num[curr_idx] < denom[curr_idx]:
			num[curr_idx] += 10
			curr_idx -= 1
			while num[curr_idx] == 0:
				num[curr_idx] += 9
				curr_idx -= 1
			num[curr_idx] -= 1
		curr_digits = [num[i] - denom[i]]+curr_digits
		scratchpad += f"{nested_digits_to_str(num)} - {nested_digits_to_str(denom)} , {int_to_str(num[i])} - {int_to_str(denom[i])} , {digits_to_str(curr_digits)} => "
		num = num[:-1]
		denom = denom[:-1]
	scratchpad += f"{digits_to_str(curr_digits)}"

	return scratchpad

def sample_number_with_n_digits(n: int) -> int:
    min_val = 10**(n-1) if n > 1 else 0
    max_val = 10**n - 1
    return random.randint(min_val, max_val)

def extract_scratch_answer(completion: str) -> Optional[str]:
    search_result = re.search(r'\#(.+)\#', completion)
    if search_result:
        s = search_result.group(0)
        for c in '# -':
            s = s.replace(c, '')
        return s
    return None

def generate_scratchpad_dataset(digits: List[int], n_items: int, seed: int) -> ReasoningData:
    random_state = RandomState(seed)
    unique_questions = set()
    questions, scratchpad_answers, direct_answers = [], [], []

    with seed_context(random_state):
        while len(unique_questions) < n_items:
            num_digits = random.choice(digits)
            a = sample_number_with_n_digits(num_digits)
            b = sample_number_with_n_digits(num_digits)
            if a < b:
                continue
            question = f" Question: {int_to_str(a)} - {int_to_str(b)}"
            if question in unique_questions:
                continue
            unique_questions.add(question)

            scratch_output_str = f"{generate_scratchpad(a, b)} Final answer: #{int_to_str(a + b)}#"
            direct_output_str = f"{int_to_str(a - b)}"

            questions.append(question)
            scratchpad_answers.append(scratch_output_str)
            direct_answers.append(direct_output_str)   
    
    return ReasoningData(
        teacher_prompt="Let's think step by step.", 
        student_prompt="", 
        question_prompts=[""], 
        questions=questions, 
        scratchpad_answers=scratchpad_answers, 
        direct_answers=direct_answers, 
        extract_answer_function=extract_scratch_answer, 
    )


