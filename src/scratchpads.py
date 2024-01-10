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

def generate_scratchpad(a: int, b: int) -> str:
	n = max(n_digits(a), n_digits(b))
	a_digits, b_digits = int_to_digits(a, n), int_to_digits(b, n)

	scratchpad = f"{digits_to_str(a_digits)} + {digits_to_str(b_digits)} , C: 0 => "

	carry, *curr_digits = int_to_digits(a_digits[-1]+b_digits[-1], 2)
	for a_digit, b_digit in list(zip(a_digits, b_digits))[::-1][1:]:
		scratchpad += f"{a_digit} + {b_digit} , {digits_to_str(curr_digits)}  C: {carry} => "
		carry, new_digit = int_to_digits(a_digit+b_digit+carry, 2)
		curr_digits = [new_digit]+curr_digits

	scratchpad += f", {digits_to_str(curr_digits)}  C: {carry} => "
	
	curr_digits = [carry]+curr_digits
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

def extract_scratch_answer_tk(completion: str) -> Optional[str]:
    search_result = re.search(r'\# (.+) \#', completion)
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
            question = f" Question: {int_to_str(a)} + {int_to_str(b)}"
            if question in unique_questions:
                continue
            unique_questions.add(question)

            scratch_output_str = f"{generate_scratchpad(a, b)} Final answer: #{int_to_str(a + b)}#"
            direct_output_str = f"{int_to_str(a + b)}"

            questions.append(question)
            scratchpad_answers.append(scratch_output_str)
            direct_answers.append(direct_output_str)   
    
    return ReasoningData(
        teacher_prompt="Let's think step by step.", 
        student_prompt="", 
        question_prompts=["Generate an addition problem."], 
        questions=questions, 
        scratchpad_answers=scratchpad_answers, 
        direct_answers=direct_answers, 
        extract_answer_function=extract_scratch_answer, 
    )

def tk_generate_scratchpad_dataset(digits: List[int], n_items: int, seed: int, random_tk_ins: List[str], random_tk_outs: List[str]) -> ReasoningData:
    random_state = RandomState(seed)
    unique_questions = set()
    questions, scratchpad_answers, direct_answers = [], [], []

    with seed_context(random_state):
        while len(unique_questions) < n_items:
            num_digits = random.choice(digits)
            a = sample_number_with_n_digits(num_digits)
            b = sample_number_with_n_digits(num_digits)
            question = f"Description: In this task you will add two numbers and show your work step by step. Positive Example 1 – Input: 8 0 5 6 0 2 0 + 1 6 7 9 2 1 5 . Output: 8 0 5 6 0 2 0 + 1 6 7 9 2 1 5 , C: 0 => 2 + 1 , 5  C: 0 => 0 + 2 , 3 5  C: 0 => 6 + 9 , 2 3 5  C: 0 => 5 + 7 , 5 2 3 5  C: 1 => 0 + 6 , 3 5 2 3 5  C: 1 => 8 + 1 , 7 3 5 2 3 5  C: 0 => , 9 7 3 5 2 3 5  C: 0 => 0 9 7 3 5 2 3 5 Final answer: # 9 7 3 5 2 3 5 # . Positive Example 2 – Input: 7 7 0 1 3 + 7 3 6 9 1 . Output: 7 7 0 1 3 + 7 3 6 9 1 , C: 0 => 1 + 9 , 4  C: 0 => 0 + 6 , 0 4  C: 1 => 7 + 3 , 7 0 4  C: 0 => 7 + 7 , 0 7 0 4  C: 1 => , 5 0 7 0 4  C: 1 => 1 5 0 7 0 4 Final answer: # 1 5 0 7 0 4 # . Positive Example 3 – Input: 6 0 8 8 7 4 3 + 8 9 9 5 9 7 0 . Output: 6 0 8 8 7 4 3 + 8 9 9 5 9 7 0 , C: 0 => 4 + 7 , 3  C: 0 => 7 + 9 , 1 3  C: 1 => 8 + 5 , 7 1 3  C: 1 => 8 + 9 , 4 7 1 3  C: 1 => 0 + 9 , 8 4 7 1 3  C: 1 => 6 + 8 , 0 8 4 7 1 3  C: 1 => , 5 0 8 4 7 1 3  C: 1 => 1 5 0 8 4 7 1 3 Final answer: # 1 5 0 8 4 7 1 3 # . Now complete the following example - Input: {int_to_str(a)} + {int_to_str(b)} . Output: "
            if question in unique_questions:
                continue
            unique_questions.add(question)

            scratch_output_str = f"{generate_scratchpad(a, b)} Final answer: # {int_to_str(a + b)} #"

            questions.append(question)
            scratchpad_answers.append(scratch_output_str)
            direct_answers.append(f"{int_to_str(a + b)}")
    
    return ReasoningData(
        teacher_prompt="", 
        student_prompt="", 
        question_prompts=["Generate an addition problem."], 
        questions=questions+random_tk_ins, 
        scratchpad_answers=scratchpad_answers+random_tk_outs, 
        direct_answers=direct_answers+random_tk_outs, 
        extract_answer_function=extract_scratch_answer_tk, 
    )

def tk_generate_direct_dataset(digits: List[int], n_items: int, seed: int, random_tk_ins: List[str], random_tk_outs: List[str]) -> ReasoningData:
    random_state = RandomState(seed)
    unique_questions = set()
    questions, direct_answers = [], []

    with seed_context(random_state):
        while len(unique_questions) < n_items:
            num_digits = random.choice(digits)
            a = sample_number_with_n_digits(num_digits)
            b = sample_number_with_n_digits(num_digits)
            question = f"Description: In this task you will add two numbers. Positive Example 1 – Input: 8 0 5 6 0 2 0 + 1 6 7 9 2 1 5 . Output: 9 7 3 5 2 3 5 . Positive Example 2 – Input: 7 7 0 1 3 + 7 3 6 9 1 . Output: 1 5 0 7 0 4 . Now complete the following example - Input: {int_to_str(a)} + {int_to_str(b)} . Output: "
            if question in unique_questions:
                continue
            unique_questions.add(question)

            direct_output_str = f"{int_to_str(a + b)}"

            questions.append(question)
            direct_answers.append(direct_output_str)   
    
    return ReasoningData(
        teacher_prompt="", 
        student_prompt="", 
        question_prompts=["Generate an addition problem."], 
        questions=questions+random_tk_ins, 
        scratchpad_answers=['' for _ in range(len(direct_answers)+len(random_tk_outs))], 
        direct_answers=direct_answers+random_tk_outs, 
        extract_answer_function=extract_scratch_answer_tk, 
    )

def tk_generate_direct_dataset2(digits: List[int], n_items: int, seed: int, random_tk_ins: List[str], random_tk_outs: List[str]) -> ReasoningData:
    random_state = RandomState(seed)
    unique_questions = set()
    questions, direct_answers = [], []

    with seed_context(random_state):
        while len(unique_questions) < n_items:
            num_digits = random.choice(digits)
            a = sample_number_with_n_digits(num_digits)
            b = sample_number_with_n_digits(num_digits)
            question = f"Description: In this task you will add two numbers. Positive Example 1 – Input: 8 0 5 6 0 2 0 + 1 6 7 9 2 1 5 . Output: 8 0 5 6 0 2 0 + 1 6 7 9 2 1 5 = 9 7 3 5 2 3 5 . Positive Example 2 – Input: 7 7 0 1 3 + 7 3 6 9 1 . Output: 7 7 0 1 3 + 7 3 6 9 1 = 1 5 0 7 0 4 . Now complete the following example - Input: {int_to_str(a)} + {int_to_str(b)} . Output: "
            if question in unique_questions:
                continue
            unique_questions.add(question)

            direct_output_str = f"{int_to_str(a)} + {int_to_str(b)} = {int_to_str(a + b)}"

            questions.append(question)
            direct_answers.append(direct_output_str)   
    
    return ReasoningData(
        teacher_prompt="", 
        student_prompt="", 
        question_prompts=["Generate an addition problem."], 
        questions=questions+random_tk_ins, 
        scratchpad_answers=['' for _ in range(len(direct_answers)+len(random_tk_outs))], 
        direct_answers=direct_answers+random_tk_outs, 
        extract_answer_function=extract_scratch_answer_tk, 
    )

def tk_generate_distractor_direct_dataset(digits: List[int], n_items: int, seed: int, random_tk_ins: List[str], random_tk_outs: List[str], distractor_words: List[str]) -> ReasoningData:
    random_state = RandomState(seed)
    unique_questions = set()
    questions, direct_answers = [], []

    with seed_context(random_state):
        while len(unique_questions) < n_items:
            num_digits = random.choice(digits)
            a = sample_number_with_n_digits(num_digits)
            b = sample_number_with_n_digits(num_digits)
            distractor_word1 = random.choice(distractor_words)
            distractor_word2 = random.choice(distractor_words)
            distractor_word3 = random.choice(distractor_words)
            distractor_word_curr = random.choice(distractor_words)
            question = f"Description: In this task you will add two numbers of things. Positive Example 1 – Input: 8 0 5 6 0 2 0 {distractor_word1} + 1 6 7 9 2 1 5 {distractor_word1} . Output: 9 7 3 5 2 3 5 {distractor_word1} . Positive Example 2 – Input: 7 7 0 1 3 {distractor_word2} + 7 3 6 9 1 {distractor_word2} . Output: 1 5 0 7 0 4 {distractor_word2} . Now complete the following example - Input: {int_to_str(a)} {distractor_word_curr} + {int_to_str(b)} {distractor_word_curr} . Output: "
            if question in unique_questions:
                continue
            unique_questions.add(question)

            direct_output_str = f"{int_to_str(a + b)} {distractor_word_curr}"

            questions.append(question)
            direct_answers.append(direct_output_str)   
    
    return ReasoningData(
        teacher_prompt="", 
        student_prompt="", 
        question_prompts=["Generate an addition problem."], 
        questions=questions+random_tk_ins, 
        scratchpad_answers=['' for _ in range(len(direct_answers)+len(random_tk_outs))], 
        direct_answers=direct_answers+random_tk_outs, 
        extract_answer_function=extract_scratch_answer_tk, 
    )

def tk_generate_distractor_direct_dataset2(digits: List[int], n_items: int, seed: int, random_tk_ins: List[str], random_tk_outs: List[str], distractor_words: List[str]) -> ReasoningData:
    random_state = RandomState(seed)
    unique_questions = set()
    questions, direct_answers = [], []

    with seed_context(random_state):
        while len(unique_questions) < n_items:
            num_digits = random.choice(digits)
            a = sample_number_with_n_digits(num_digits)
            b = sample_number_with_n_digits(num_digits)
            distractor_word1 = random.choice(distractor_words)
            distractor_word2 = random.choice(distractor_words)
            distractor_word3 = random.choice(distractor_words)
            distractor_word_curr = random.choice(distractor_words)
            question = f"Description: In this task you will add two numbers of things. Positive Example 1 – Input: 8 0 5 6 0 2 0 {distractor_word1} + 1 6 7 9 2 1 5 {distractor_word1} . Output: 8 0 5 6 0 2 0 + 1 6 7 9 2 1 5 = 9 7 3 5 2 3 5 {distractor_word1} . Positive Example 2 – Input: 7 7 0 1 3 {distractor_word2} + 7 3 6 9 1 {distractor_word2} . Output: 7 7 0 1 3 + 7 3 6 9 1 = 1 5 0 7 0 4 {distractor_word2} . Now complete the following example - Input: {int_to_str(a)} {distractor_word_curr} + {int_to_str(b)} {distractor_word_curr} . Output: "
            if question in unique_questions:
                continue
            unique_questions.add(question)

            direct_output_str = f"{int_to_str(a)} + {int_to_str(b)} = {int_to_str(a + b)} {distractor_word_curr}"

            questions.append(question)
            direct_answers.append(direct_output_str)   
    
    return ReasoningData(
        teacher_prompt="", 
        student_prompt="", 
        question_prompts=["Generate an addition problem."], 
        questions=questions+random_tk_ins, 
        scratchpad_answers=['' for _ in range(len(direct_answers)+len(random_tk_outs))], 
        direct_answers=direct_answers+random_tk_outs, 
        extract_answer_function=extract_scratch_answer_tk, 
    )

def tk_generate_contextual_direct_dataset(digits: List[int], n_items: int, seed: int, random_tk_ins: List[str], random_tk_outs: List[str], distractor_words: List[str]) -> ReasoningData:
    random_state = RandomState(seed)
    unique_questions = set()
    questions, direct_answers = [], []

    with seed_context(random_state):
        while len(unique_questions) < n_items:
            num_digits = random.choice(digits)
            a = sample_number_with_n_digits(num_digits)
            b = sample_number_with_n_digits(num_digits)
            distractor_word1 = random.choice(distractor_words)
            distractor_word2 = random.choice(distractor_words)
            distractor_word3 = random.choice(distractor_words)
            distractor_word_curr = random.choice(distractor_words)
            ask_a = random.choice([True, False])
            question = f"Description: In this task you will answer questions about numbers of things. Positive Example 1 – Input: A has 8 0 5 6 0 2 0 {distractor_word1} . B has 1 6 7 9 2 1 5 {distractor_word1} . How many {distractor_word1} does B have ? Output: 1 6 7 9 2 1 5 {distractor_word1} . Positive Example 2 – Input: A has 7 7 0 1 3 {distractor_word2} . B has 7 3 6 9 1 {distractor_word2} . How many {distractor_word2} does A have ? Output: 7 7 0 1 3 {distractor_word2} . Now complete the following example - Input: A has {int_to_str(a)} {distractor_word_curr} . B has {int_to_str(b)} {distractor_word_curr} . How many {distractor_word_curr} does {'A' if ask_a else 'B'} have ? Output: "
            if question in unique_questions:
                continue
            unique_questions.add(question)

            direct_output_str = f"{int_to_str(a) if ask_a else int_to_str(b)} {distractor_word_curr}"

            questions.append(question)
            direct_answers.append(direct_output_str)   
    
    return ReasoningData(
        teacher_prompt="", 
        student_prompt="", 
        question_prompts=["Generate an addition problem."], 
        questions=questions+random_tk_ins, 
        scratchpad_answers=['' for _ in range(len(direct_answers)+len(random_tk_outs))], 
        direct_answers=direct_answers+random_tk_outs, 
        extract_answer_function=extract_scratch_answer_tk, 
    )


# if __name__ == "__main__":
#     d = tk_generate_scratchpad_dataset(list(range(1, 9)), 100, 0, [], [])
