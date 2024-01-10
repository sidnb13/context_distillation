from collections import defaultdict
from datasets import load_dataset
from utils.randomness import RandomState, seed_context
from injection_functions import InjectionData, format_input
import random
import json

def load_spider(data_path, n_per_prompt, seed):
    random_state = RandomState(seed)

    with open(data_path, 'r') as f:
        d = json.load(f)
    
    by_db_id = defaultdict(list)
    for item in d:
        by_db_id[item['db_id']].append((item['question'], item['query']))
    
    prompt_set = defaultdict(list)
    dev_set = defaultdict(list)
    
    with seed_context(random_state):
        for k in by_db_id.keys():
            if len(by_db_id[k]) == 0:
                continue
            query_matches = defaultdict(list)
            for item in by_db_id[k]:
                query_matches[item[1]].append(item[0])
            train_queries = random.sample(list(query_matches.keys()), n_per_prompt)
            for query in train_queries:
                question = random.choice(query_matches[query])
                prompt_set[k].append((question, query))
            dev_queries = sorted(set(query_matches.keys()).difference(set(train_queries)))
            for query in dev_queries:
                for question in query_matches[query]:
                    dev_set[k].append((question, query))
    
    return prompt_set, dev_set

def create_spider_injection_data(data_path, n_per_prompt, seed, add_description, grad_descent_eval_mode=False):
    
    prompt_set, dev_set = load_spider(data_path, n_per_prompt, seed)

    injection_datas = {}
    for k in prompt_set.keys():
        if add_description:
            teacher_prompt = f"Definition: In this semantic parsing task you will convert natural language questions into SQL queries. All SQL queries will be executed on the \"{' '.join(k.split('_'))}\" database."
        else:
            teacher_prompt = ""

        if not grad_descent_eval_mode:
            for i in range(len(prompt_set[k])):
                teacher_prompt += f" Positive Example {i+1} – Input: {prompt_set[k][i][0]} Output: {prompt_set[k][i][1]} ."

        injection_datas[k] = InjectionData(
            teacher_prompt=teacher_prompt, 
            student_prompt="", 
            question_generation_prompts=[], 
            dataset_questions=[format_input(question) for question, _ in dev_set[k]], 
            dataset_eval=[(format_input(question), [query]) for question, query in dev_set[k]], 
            meta=None, 
        )
    
    return injection_datas

if __name__ == "__main__":
    data = create_spider_injection_data('../data/spider/dev.json', 3, 0, True)
