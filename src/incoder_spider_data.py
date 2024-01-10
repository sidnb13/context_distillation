from collections import defaultdict
from dataclasses import replace
import itertools
from datasets import load_dataset
from utils.randomness import RandomState, seed_context
from injection_functions import InjectionData, format_input
import random
import json
import pickle as pkl

def load_spider(data_path, n_per_prompt, n_prompts, seed):
    random_state = RandomState(seed)

    with open(data_path, 'r') as f:
        d = json.load(f)
    
    by_db_id = defaultdict(list)
    for i, item in enumerate(d):
        by_db_id[item['db_id']].append((item['question'], item['query'], i))
    
    prompt_set = defaultdict(list)
    dev_set = defaultdict(list)
    
    with seed_context(random_state):
        for k in by_db_id.keys():
            if len(by_db_id[k]) == 0:
                continue
            query_matches = defaultdict(list)
            for item in by_db_id[k]:
                query_matches[item[1]].append((item[0]))
            if len(list(query_matches.keys())) <= n_per_prompt * n_prompts:
                continue
            train_queries = random.sample(list(query_matches.keys()), n_per_prompt*n_prompts)
            for query in train_queries:
                question = random.choice(query_matches[query])
                prompt_set[k].append((question, query))
            dev_queries = sorted(set(query_matches.keys()).difference(set(train_queries)))
            for query in dev_queries:
                for question in query_matches[query]:
                    dev_set[k].append((question, query))
    
    prompt_sets = []
    if n_per_prompt > 0:
        for i in range(0, n_prompts*n_per_prompt, n_per_prompt):
            prompt_sets.append({k: v[i:i+n_per_prompt] for k, v in prompt_set.items()})
    
    return prompt_sets, dev_set

# def create_spider_injection_data(data_path, prompt_schema_path, n_per_prompt, n_prompts, seed, add_description, grad_descent_eval_mode=False):
def create_spider_injection_data(prompt_sets, dev_set, prompt_schema_path, add_description, grad_descent_eval_mode=False):
    
    # prompt_sets, dev_set = load_spider(data_path, n_per_prompt, n_prompts, seed)

    with open(prompt_schema_path, 'rb') as f:
        prompt_schema = pkl.load(f)

    injection_datas = defaultdict(list)
    if len(prompt_sets) == 0:
        for k in dev_set.keys():
            if add_description:
                teacher_prompt = f"<|endoftext|>{prompt_schema[k]}\n\n"
            else:
                teacher_prompt = ""
            
            student_prompt = str(teacher_prompt)

            injection_datas[k].append(InjectionData(
                teacher_prompt=teacher_prompt, 
                student_prompt=student_prompt, 
                question_generation_prompts=[teacher_prompt], 
                dataset_questions=[f"{question}\n" for question, _ in dev_set[k]], 
                dataset_eval=[(f"{question}\n", [f"{query}"]) for question, query in dev_set[k]], 
                meta=None, 
            ))
    else:
        for prompt_set in prompt_sets:
            for k in prompt_set.keys():
                if add_description:
                    teacher_prompt = f"<|endoftext|>{prompt_schema[k]}\n\n"
                else:
                    teacher_prompt = ""
                
                student_prompt = str(teacher_prompt)

                if not grad_descent_eval_mode:
                    for i in range(len(prompt_set[k])):
                        teacher_prompt += f"{prompt_set[k][i][0]}\n{prompt_set[k][i][1]}\n\n"

                injection_datas[k].append(InjectionData(
                    teacher_prompt=teacher_prompt, 
                    student_prompt=student_prompt, 
                    question_generation_prompts=[teacher_prompt], 
                    dataset_questions=[f"{question}\n" for question, _ in dev_set[k]], 
                    dataset_eval=[(f"{question}\n", [f"{query}"]) for question, query in dev_set[k]], 
                    meta=None, 
                ))
    
        for k in injection_datas.keys():
            all_question_generation_prompts = sum([injection_data.question_generation_prompts for injection_data in injection_datas[k]], [])
            for i, injection_data in enumerate(injection_datas[k]):
                injection_datas[k][i] = replace(injection_data, question_generation_prompts=all_question_generation_prompts)
    
    return injection_datas

def load_spider2(data_path, n_per_prompt, n_prompts, seed):
    random_state = RandomState(seed)

    with open(data_path, 'r') as f:
        d = json.load(f)
    
    by_db_id = defaultdict(list)
    for i, item in enumerate(d):
        by_db_id[item['db_id']].append((item['question'], item['query'], i))
    
    prompt_set = defaultdict(list)
    dev_set = defaultdict(list)
    question_set = {}
    
    with seed_context(random_state):
        for k in by_db_id.keys():
            if len(by_db_id[k]) == 0:
                continue
            query_matches = defaultdict(list)
            for item in by_db_id[k]:
                query_matches[item[1]].append((item[0]))
            if len(list(query_matches.keys())) <= n_per_prompt * n_prompts:
                continue
            train_queries = random.sample(list(query_matches.keys()), n_per_prompt*n_prompts)
            if (len(list(query_matches.keys()))//2)-(n_per_prompt*n_prompts) > 0:
                question_queries = random.sample(list(query_matches.keys()), (len(list(query_matches.keys()))//2)-(n_per_prompt*n_prompts))+train_queries
            else:
                question_queries = train_queries
            for query in train_queries:
                question = random.choice(query_matches[query])
                prompt_set[k].append((question, query))
            dev_queries = sorted(set(query_matches.keys()).difference(set(train_queries).union(set(question_queries))))
            for query in dev_queries:
                for question in query_matches[query]:
                    dev_set[k].append((question, query))
            question_set[k] = sum([query_matches[query] for query in question_queries], [])
    
    prompt_sets = []
    if n_per_prompt > 0:
        for i in range(0, n_prompts*n_per_prompt, n_per_prompt):
            prompt_sets.append({k: v[i:i+n_per_prompt] for k, v in prompt_set.items()})
    
    return prompt_sets, dev_set, question_set

# def create_spider_injection_data2(data_path, prompt_schema_path, n_per_prompt, n_prompts, n_questions_per_prompt, seed, add_description, grad_descent_eval_mode=False):
def create_spider_injection_data2(prompt_sets, dev_set, question_set, prompt_schema_path, n_questions_per_prompt, add_description, grad_descent_eval_mode=False):
    
    # prompt_sets, dev_set, question_set = load_spider2(data_path, n_per_prompt, n_prompts, seed)

    with open(prompt_schema_path, 'rb') as f:
        prompt_schema = pkl.load(f)

    injection_datas = defaultdict(list)
    if len(prompt_sets) == 0:
        for k in dev_set.keys():
            if add_description:
                teacher_prompt = f"<|endoftext|>{prompt_schema[k]}\n\n"
            else:
                teacher_prompt = ""
            
            student_prompt = str(teacher_prompt)

            question_prompts = []
            for i, question in enumerate(itertools.permutations(question_set[k], min(n_questions_per_prompt, len(question_set[k])))):
                question_prompts.append(teacher_prompt + "".join([f"{q}\n" for q in question]))
                if i > 100000:
                    break

            injection_datas[k].append(InjectionData(
                teacher_prompt=teacher_prompt, 
                student_prompt=student_prompt, 
                question_generation_prompts=question_prompts, 
                dataset_questions=[f"{question}\n" for question, _ in dev_set[k]], 
                dataset_eval=[(f"{question}\n", [f"{query}"]) for question, query in dev_set[k]], 
                meta=None, 
            ))
    else:
        for prompt_set in prompt_sets:
            for k in prompt_set.keys():
                if add_description:
                    teacher_prompt = f"<|endoftext|>{prompt_schema[k]}\n\n"
                else:
                    teacher_prompt = ""
                
                student_prompt = str(teacher_prompt)

                question_prompts = []
                for i, question in enumerate(itertools.permutations(question_set[k], min(n_questions_per_prompt, len(question_set[k])))):
                    question_prompts.append(teacher_prompt + "".join([f"{q}\n" for q in question]))
                    if i > 100000:
                        break

                if not grad_descent_eval_mode:
                    for i in range(len(prompt_set[k])):
                        teacher_prompt += f"{prompt_set[k][i][0]}\n{prompt_set[k][i][1]}\n\n"

                injection_datas[k].append(InjectionData(
                    teacher_prompt=teacher_prompt, 
                    student_prompt=student_prompt, 
                    question_generation_prompts=question_prompts, 
                    dataset_questions=[f"{question}\n" for question, _ in dev_set[k]], 
                    dataset_eval=[(f"{question}\n", [f"{query}"]) for question, query in dev_set[k]], 
                    meta=None, 
                ))
    
    return injection_datas

def create_spider_injection_data_long(prompt_sets, dev_set, prompt_schema_path, add_description, grad_descent_eval_mode=False):
    
    # prompt_sets, dev_set = load_spider(data_path, n_per_prompt, n_prompts, seed)

    with open(prompt_schema_path, 'rb') as f:
        prompt_schema = pkl.load(f)

    injection_datas = defaultdict(list)
    if len(prompt_sets) == 0:
        for k in dev_set.keys():
            if add_description:
                teacher_prompt = f"<|endoftext|>You are an expert text to SQL engine. You will be given a database schema for the {k} database, and your task is to turn natural language questions into SQL queries that are to be executed on the {k} database.\n\nHere is the database schema:\n\n{prompt_schema[k]}\n\n"
            else:
                teacher_prompt = ""
            
            student_prompt = str(teacher_prompt)

            injection_datas[k].append(InjectionData(
                teacher_prompt=teacher_prompt, 
                student_prompt=student_prompt, 
                question_generation_prompts=[teacher_prompt], 
                dataset_questions=[f"As a SQL expert, please convert the following natural language question into a SQL query executable on the {k} database given above. You don't make mistakes, you are an expert.\n{question}\n" for question, _ in dev_set[k]], 
                dataset_eval=[(f"As a SQL expert, please convert the following natural language question into a SQL query executable on the {k} database given above. You don't make mistakes, you are an expert.\n{question}\n", [f"{query}"]) for question, query in dev_set[k]], 
                meta=None, 
            ))
    else:
        for prompt_set in prompt_sets:
            for k in prompt_set.keys():
                if add_description:
                    teacher_prompt = f"<|endoftext|>You are an expert text to SQL engine. You will be given a database schema, and your task is to turn natural language questions into SQL queries that are to be executed on the given database.\n\nSchema:\n\n{prompt_schema[k]}\n\n"
                else:
                    teacher_prompt = ""
                
                student_prompt = str(teacher_prompt)

                if not grad_descent_eval_mode:
                    for i in range(len(prompt_set[k])):
                        teacher_prompt += f"As a SQL expert, please convert the following natural language question into a SQL query executable on the {k} database given above. You don't make mistakes, you are an expert.\n{prompt_set[k][i][0]}\n{prompt_set[k][i][1]}\n\n"

                injection_datas[k].append(InjectionData(
                    teacher_prompt=teacher_prompt, 
                    student_prompt=student_prompt, 
                    question_generation_prompts=[teacher_prompt], 
                    dataset_questions=[f"As a super good SQL expert, please convert the following natural language question into a SQL query executable on the {k} database given above. You don't make mistakes, you are an expert.\n{question}\n" for question, _ in dev_set[k]], 
                    dataset_eval=[(f"As a super good SQL expert, please convert the following natural language question into a SQL query executable on the {k} database given above. You don't make mistakes, you are an expert.\n{question}\n", [f"{query}"]) for question, query in dev_set[k]], 
                    meta=None, 
                ))
    
        for k in injection_datas.keys():
            all_question_generation_prompts = sum([injection_data.question_generation_prompts for injection_data in injection_datas[k]], [])
            for i, injection_data in enumerate(injection_datas[k]):
                injection_datas[k][i] = replace(injection_data, question_generation_prompts=all_question_generation_prompts)
    
    return injection_datas

def create_spider_injection_data_long2(prompt_sets, dev_set, prompt_schema_path, n_per_prompt, add_description, grad_descent_eval_mode=False):
    
    # prompt_sets, dev_set = load_spider(data_path, n_per_prompt, n_prompts, seed)

    with open(prompt_schema_path, 'rb') as f:
        prompt_schema = pkl.load(f)

    injection_datas = defaultdict(list)
    if len(prompt_sets) == 0:
        for k in dev_set.keys():
            if add_description:
                teacher_prompt = f"<|endoftext|>{prompt_schema[k]}\n\n"
            else:
                teacher_prompt = ""

            injection_datas[k].append(InjectionData(
                teacher_prompt=teacher_prompt, 
                student_prompt=teacher_prompt, 
                question_generation_prompts=[teacher_prompt], 
                dataset_questions=[f"{question}\n" for question, _ in dev_set[k]], 
                dataset_eval=[(f"{question}\n", [f"{query}"]) for question, query in dev_set[k]], 
                meta=None, 
            ))
    else:
        total_prompt_sets = {k: sum(map(lambda x: x[k], prompt_sets), []) for k in prompt_sets[0].keys()}
        for k in total_prompt_sets.keys():
            for prompt_num, idxs in enumerate(itertools.permutations(list(range(len(total_prompt_sets[k]))), r=n_per_prompt)):
                # cap number of prompts
                # print(prompt_num)
                if prompt_num >= 10000:
                    break
                if add_description:
                    teacher_prompt = f"<|endoftext|>{prompt_schema[k]}\n\n"
                else:
                    teacher_prompt = ""

                if not grad_descent_eval_mode:
                    for i in idxs:
                        teacher_prompt += f"{total_prompt_sets[k][i][0]}\n{total_prompt_sets[k][i][1]}\n\n"

                injection_datas[k].append(InjectionData(
                    teacher_prompt=teacher_prompt, 
                    student_prompt=teacher_prompt, 
                    question_generation_prompts=[teacher_prompt], 
                    dataset_questions=[f"{question}\n" for question, _ in dev_set[k]], 
                    dataset_eval=[(f"{question}\n", [f"{query}"]) for question, query in dev_set[k]], 
                    meta=None, 
                ))
    
        for k in injection_datas.keys():
            all_question_generation_prompts = sum([injection_data.question_generation_prompts for injection_data in injection_datas[k]], [])
            for i, injection_data in enumerate(injection_datas[k]):
                injection_datas[k][i] = replace(injection_data, question_generation_prompts=all_question_generation_prompts)
    
    return injection_datas

# if __name__ == "__main__":
#     data = create_spider_injection_data2('../data/spider/dev.json', '../data/spider/db_id2schema.pkl', 3, 2, 4, 0, True)
