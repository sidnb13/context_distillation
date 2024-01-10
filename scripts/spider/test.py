from collections import defaultdict
import pickle as pkl
from tk_jax.compute_metrics import compute_metrics

if __name__ == "__main__":
    d0 = pkl.load(open('../../outputs/spider_pretrain_test2_0_per_prompt/total_results.pkl', 'rb'))
    d1 = pkl.load(open('../../outputs/spider_pretrain_test2_1_per_prompt/total_results.pkl', 'rb'))
    d2 = pkl.load(open('../../outputs/spider_pretrain_test2_2_per_prompt/total_results.pkl', 'rb'))
    d3 = pkl.load(open('../../outputs/spider_pretrain_test2_3_per_prompt/total_results.pkl', 'rb'))
    d4 = pkl.load(open('../../outputs/spider_pretrain_test2_4_per_prompt/total_results.pkl', 'rb'))

    aggregate = defaultdict(list)
    for k in set(d0.keys()).intersection(set(d1.keys())).intersection(set(d2.keys())).intersection(set(d3.keys())).intersection(set(d4.keys())):
        print(k)
        for d, prompt_str in [(d0, 'no examples'), (d1, '1 example'), (d2, '2 examples'), (d3, '3 examples'), (d4, '4 examples')]:
            max_em = float('-inf')
            for item in d[k]:
                predictions = list(map(lambda x: x['generation'], item))
                reference = list(map(lambda x: x['reference'], item))
                max_em = max(max_em, compute_metrics(predictions, reference)['exact_match'])
            print(f'{prompt_str}: {max_em}')
            aggregate[prompt_str].append(max_em)
    
    print('aggregate')
    for k, v in aggregate.items():
        print(f'{k}: {sum(v)/len(v)}')
