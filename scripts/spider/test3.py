from incoder_spider_data import create_spider_injection_data, create_spider_injection_data_long2, load_spider
from transformers import AutoTokenizer

if __name__ == "__main__":
    model_out_path = None
    add_description = True
    n_per_prompt = 8
    n_prompts = 1

    prompt_sets, dev_set = load_spider('../../data/spider/dev.json', n_per_prompt, n_prompts, 0)
    injection_datas = create_spider_injection_data(prompt_sets, dev_set, '../../data/spider/db_id2schema_2.pkl', add_description, grad_descent_eval_mode=False)

    tokenizer = AutoTokenizer.from_pretrained('facebook/incoder-6B')

    for k in injection_datas.keys():
        for item in injection_datas[k]:
            print(k, max(map(lambda x: len(tokenizer.encode(item.teacher_prompt+x[0])), item.dataset_eval)))