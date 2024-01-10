from dataclasses import replace
import pickle as pkl
from typing import List
from scratchpads import generate_scratchpad_dataset, tk_generate_contextual_direct_dataset, tk_generate_direct_dataset, tk_generate_direct_dataset2, tk_generate_distractor_direct_dataset, tk_generate_distractor_direct_dataset2
from transformers import T5Tokenizer
from t5_config import T5ModelConfig
from core import TKTrainConfig
from micro_config import MetaConfig
from base_configs import project_root, AdamWConfig
from reasoning_distill import mixed_train, scratchpads_train, scratchpads_eval, direct_eval, direct_train
import jax
from tk_jax.data import NatInstSeq2SeqConfig
from tk_jax.eval_inference import TKInstructEvaluationConfig, tk_instruct_evaluate
from utils.randomness import RandomState, seed_context
import random
import os
import wandb

def extract_answer_eval(answer: str):
    answer = answer.replace(' ', '')
    if '=' in answer:
        answer = answer.split('=')[1]
    return ''.join(filter(lambda x: x in set(map(str, range(10))), list(answer))).strip().lower()

def eval_all_results(all_results: List[str]):
    scores = []
    for result in all_results:
        scores.append(extract_answer_eval(result['generation']) == extract_answer_eval(result['reference']))
    return sum(scores) / len(scores)

if __name__ == "__main__":
    use_scratchpad = False

    # model_checkpoint_path = "outputs/T5_11B_random_nat_inst_finetune_test2/model/"
    model_checkpoint_path = "outputs/scratch_tk_student_test2/model/"
    # model_checkpoint_path = "outputs/scratch_tk_test1/model/"
    # model_checkpoint_path = None
    # model_out_path = None

    rng = jax.random.PRNGKey(0)
    random_state = RandomState(1)
    
    all_data = tk_generate_direct_dataset2(digits=list(range(1, 9)), n_items=555, seed=0, 
                                           random_tk_ins=[], random_tk_outs=[])

    all_data1 = tk_generate_direct_dataset(digits=list(range(1, 9)), n_items=555, seed=0, 
                                           random_tk_ins=[], random_tk_outs=[])
    all_distractor_data = tk_generate_distractor_direct_dataset(digits=list(range(1, 9)), n_items=2000, seed=2, random_tk_ins=[], 
                                                                random_tk_outs=[], distractor_words=['cars', 'trucks', 'apples', 'oranges', 'fruits', 'dogs', 'cats', 'turkies'])
    all_distractor_data2 = tk_generate_distractor_direct_dataset2(digits=list(range(1, 9)), n_items=2000, seed=2, random_tk_ins=[], 
                                                                random_tk_outs=[], distractor_words=['cars', 'trucks', 'apples', 'oranges', 'fruits', 'dogs', 'cats', 'turkies'])
    
    all_contextual_data = tk_generate_contextual_direct_dataset(digits=list(range(1, 9)), n_items=2000, seed=2, random_tk_ins=[], 
                                                                random_tk_outs=[], distractor_words=['cars', 'trucks', 'apples', 'oranges', 'fruits', 'dogs', 'cats', 'turkies'])

    train_frac = 0.9

    # train_instance = replace(
    #     all_data, 
    #     questions=all_data.questions[:int(len(all_data.questions)*train_frac)], 
    #     scratchpad_answers=all_data.scratchpad_answers[:int(len(all_data.scratchpad_answers)*train_frac)], 
    #     direct_answers=all_data.direct_answers[:int(len(all_data.direct_answers)*train_frac)], 
    # )

    tk_inputs, tk_outputs = [], []
    with open('../../data/nat_inst/text2text/defintion_pos_2/train.tsv', 'r') as f:
        for line in f:
            input_str, output_str = line[:-1].split("\t")
            tk_inputs.append(input_str)
            tk_outputs.append(output_str)
    
    with seed_context(random_state):
        idxs = list(range(len(tk_inputs)))
        random.shuffle(idxs)
        tk_inputs, tk_outputs = [tk_inputs[idx] for idx in idxs], [tk_outputs[idx] for idx in idxs]
    tk_inputs, tk_outputs = tk_inputs[:4096*4*4], tk_outputs[:4096*4*4]
    print(len(tk_inputs), len(tk_outputs))

    with open("../../outputs/scratch_tk_test1/reasoning_data10k.pkl", 'rb') as f:
        train_instance = pkl.load(f)
    
    train_instance = replace(
        all_data, 
        questions=all_data.questions+tk_inputs, 
        scratchpad_answers=all_data.scratchpad_answers+tk_outputs, 
        direct_answers=all_data.direct_answers+tk_outputs, 
    )
    
    eval_instance = replace(
        all_data, 
        questions=all_data.questions[int(len(all_data.questions)*train_frac):], 
        scratchpad_answers=all_data.scratchpad_answers[int(len(all_data.scratchpad_answers)*train_frac):], 
        direct_answers=all_data.direct_answers[int(len(all_data.direct_answers)*train_frac):], 
    )

    eval_instance1 = replace(
        all_data1, 
        questions=all_data1.questions[int(len(all_data1.questions)*train_frac):], 
        scratchpad_answers=all_data1.scratchpad_answers[int(len(all_data1.scratchpad_answers)*train_frac):], 
        direct_answers=all_data1.direct_answers[int(len(all_data1.direct_answers)*train_frac):], 
    )
    
    tokenizer = T5Tokenizer.from_pretrained('google/t5-small-lm-adapt')
    
    print('loading model ...')
    
    metaconfig = MetaConfig(
        project_root=project_root, 
        verbose=False, 
    )

    model_config = T5ModelConfig(
        # model_str="google/t5-v1_1-xl", 
        # model_str="t5-3b", 
        # model_str="google/ul2", 
        model_str="google/t5-xxl-lm-adapt", 
        # model_str="allenai/tk-instruct-11b-def-pos-neg-expl", 
        # model_str="allenai/tk-instruct-3b-def-pos", 
        # checkpoint_path=None, 
        # checkpoint_path='outputs/T5_11B_random_sharded/model_1/', 
        # checkpoint_path='outputs/scratch_from_scratchpad_direct_test2/model/', 
        checkpoint_path=model_checkpoint_path, 
        from_pretrained=True, 
        use_fp16=False, 
        gradient_checkpoint=False, 
    )
    
    trainer_config = TKTrainConfig(
        model=model_config, 
        optim=AdamWConfig(
            grad_accum_steps=1, 
            lr=1e-5, 
            weight_decay=0.00, 
            beta1=0.9, 
            beta2=0.999, 
            eps=1e-6, 
        ), 
        pjit=True, 
        verbose=True, 
    )

    trainer, inference, model, mesh = trainer_config.unroll(metaconfig)

    nat_inst_eval_dataset = NatInstSeq2SeqConfig(
        tsv_path='data/nat_inst/text2text/defintion_pos_2/test.tsv', 
        enc_len=1024, 
        dec_len=128, 
        add_ar_sentinal=False, 
        target_prepend_pad=True, 
        model_tokenizer=model_config, 
    )

    tk_eval_config = TKInstructEvaluationConfig(
        eval_dataset=nat_inst_eval_dataset, 
        inference=trainer_config, 
        reference_file='data/nat_inst/text2text/defintion_pos_2/test_examples.jsonl', 
        task_categories_file='data/nat_inst/task_category.json', 
        rng=2, 
        bsize=32, 
        eval_batches=None, 
        save_generations_path=None, 
        generation_kwargs={
            'max_length': 128, 
            'do_sample': False, 
            'num_beams': 1, 
        }, 
        verbose=True, 
    )

    tk_eval_kwargs = tk_eval_config.unroll(metaconfig)

    train_f = scratchpads_train if use_scratchpad else direct_train
    eval_f = scratchpads_eval if use_scratchpad else direct_eval
    # train_f = mixed_train
    # eval_f = direct_eval

    print('evaluating model ...')

    rng, new_rng = jax.random.split(rng)
    accuracy, all_results = eval_f(
        reasoning_data=eval_instance, inference=inference, mesh=mesh, bsize=32, 
        num_instances=None, max_input_length=1024, rng_key=new_rng, 
        do_sample=False, num_beams=1, max_length=256, 
    )
    accuracy_new = eval_all_results(all_results)

    rng, new_rng = jax.random.split(rng)
    accuracy1, all_results1 = eval_f(
        reasoning_data=eval_instance1, inference=inference, mesh=mesh, bsize=32, 
        num_instances=None, max_input_length=1024, rng_key=new_rng, 
        do_sample=False, num_beams=1, max_length=256, 
    )
    accuracy1_new = eval_all_results(all_results1)

    rng, new_rng = jax.random.split(rng)
    distractor_accuracy, distractor_all_results = eval_f(
        reasoning_data=all_distractor_data, inference=inference, mesh=mesh, bsize=32, 
        num_instances=None, max_input_length=1024, rng_key=new_rng, 
        do_sample=False, num_beams=1, max_length=256, 
    )
    distractor_accuracy_new = eval_all_results(distractor_all_results)

    rng, new_rng = jax.random.split(rng)
    distractor_accuracy2, distractor_all_results2 = eval_f(
        reasoning_data=all_distractor_data2, inference=inference, mesh=mesh, bsize=32, 
        num_instances=None, max_input_length=1024, rng_key=new_rng, 
        do_sample=False, num_beams=1, max_length=256, 
    )
    distractor_accuracy2_new = eval_all_results(distractor_all_results2)

    rng, new_rng = jax.random.split(rng)
    contextual_accuracy, contextual_all_results = eval_f(
        reasoning_data=all_contextual_data, inference=inference, mesh=mesh, bsize=32, 
        num_instances=None, max_input_length=1024, rng_key=new_rng, 
        do_sample=False, num_beams=1, max_length=256, 
    )
    true_contextual_accuracy = print(sum([' '.join([item for item in x['generation'].split(' ') if item in set(map(str, range(10)))]) == ' '.join([item for item in x['reference'].split(' ') if item in set(map(str, range(10)))]) for x in contextual_all_results]) / len(contextual_all_results))
    contextual_accuracy_new = eval_all_results(contextual_all_results)

    # tk_metrics = tk_instruct_evaluate(**{**tk_eval_kwargs, 'inference': inference})
    
    print({'normal': accuracy, 'normal1': accuracy1, 'distractor': distractor_accuracy, 'distractor2': distractor_accuracy2, 'adverse': contextual_accuracy, 'true_adverse': true_contextual_accuracy})
    # print({'tk_metrics': tk_metrics})

    print({'normal_new': accuracy_new, 'normal1_new': accuracy1_new, 'distractor_new': distractor_accuracy_new, 'distractor2_new': distractor_accuracy2_new, 'adverse_new': contextual_accuracy_new})

    breakpoint()
    pass
    
    # with open(os.path.join(data_out_path, 'all_results_%d.pkl' % (i)), 'wb') as f:
    #     pkl.dump({'normal': all_results, 'normal1': all_results1, 'distractor': distractor_all_results, 'distractor2': distractor_all_results2, 'adverse': contextual_all_results, 'tk_metrics': tk_metrics}, f)
