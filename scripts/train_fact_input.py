from micro_config import MetaConfig
from base_configs import AdaFactorConfig, AdamWConfig, project_root
from fact_edit_input_synth import FactEditSeq2SeqConfig
from nat_inst.random_input_data_generator import TKInstructInputDataSetting
from tk_jax.data import NatInstInputsSeq2SeqGeneratorConfig, NatInstSeq2SeqConfig, NatInstSeq2SeqGeneratorConfig
from t5_config import T5ModelConfig
from itertools import product
from core import TKInference, TKTrainConfig
from nat_inst.random_data_gen import TKInstructDataSetting
from tk_jax.finetune_loop import TrainLoopConfig, EvaluateLossConfig, evaluate_loss, train_model
from tk_jax.eval_inference import TKInstructEvaluationConfig, tk_instruct_evaluate
import os
import pickle as pkl
import json
from utils.randomness import RandomState, seed_context
import jax

model = T5ModelConfig(
    # model_str="google/t5-v1_1-xl", 
    # model_str="t5-3b", 
    # model_str="google/ul2", 
    model_str="google/t5-xxl-lm-adapt", 
    checkpoint_path='outputs/T5_11B_random_nat_inst_finetune_test2/model/', 
    from_pretrained=True, 
    use_fp16=True, 
    gradient_checkpoint=True, 
)

seed = 0
random_state = RandomState(seed)
rng_key = jax.random.PRNGKey(0)

with seed_context(random_state):
    with open('../data/counterfact/counterfact_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open('../data/counterfact/new_counterfact_data.json', 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    eval_ids = set(map(lambda x: x['case_id'], eval_data))
    
    data = [item for item in data if item['case_id'] not in eval_ids]

def train_examples():
    return data

def eval_examples():
    return eval_data

train_dataset = FactEditSeq2SeqConfig(
    examples=train_examples, 
    enc_len=1024, 
    dec_len=128, 
    add_ar_sentinal=False, 
    target_prepend_pad=True, 
    model_tokenizer=model, 
)

eval_dataset = FactEditSeq2SeqConfig(
    examples=eval_examples, 
    enc_len=1024, 
    dec_len=128, 
    add_ar_sentinal=False, 
    target_prepend_pad=True, 
    model_tokenizer=model, 
)

optim = AdamWConfig(
    grad_accum_steps=2, 
    lr=1e-5, 
    weight_decay=0.00, 
    beta1=0.9, 
    beta2=0.999, 
    eps=1e-6, 
)

# optim = AdaFactorConfig(
#     grad_accum_steps=8, 
#     lr=1e-5, 
#     multiply_by_parameter_scale=False, 
#     momentum_fp16=False,  
# )

trainer = TKTrainConfig(
    model=model, 
    optim=optim, 
    pjit=True, 
    verbose=True, 
)

evaluators = {
    "data": (EvaluateLossConfig(
        eval_dataset=eval_dataset, 
        inference=trainer, 
        rng=1, 
        bsize=32, 
        prefetch_batches=None, 
        eval_batches=32, 
        verbose=False, 
    ), evaluate_loss), 
}

def _get_evaluate_fn(metaconfig: MetaConfig):
    
    eval_kwargs = {}
    for k, (config, f) in evaluators.items():
        eval_kwargs[k] = (config.unroll(metaconfig), f)
    
    def _eval_fn(inference: TKInference):
        results = {}
        for k, (kwargs, f) in eval_kwargs.items():
            kwargs = {**kwargs, 'inference': inference}
            results[k] = f(**kwargs)
        return results['data']['loss'], results
    
    return _eval_fn

train_config = TrainLoopConfig(
    train_dataset=train_dataset, 
    trainer=trainer, 
    rng=3, 
    save_dir='outputs/T5_11B_fact_edit_input_generator2/', 
    max_checkpoints=None, 
    epochs=10, 
    max_steps=4096*4, 
    bsize=8, 
    prefetch_batches=None, 
    log_every=256, 
    eval_every=1024, 
    save_every=None, 
    save_only_at_end=False, 
    use_wandb=True, 
    wandb_project='fact_edit_input_generator', 
    wandb_run_name='T5_11B_fact_edit_input_generator', 
    verbose=True, 
)

if __name__ == "__main__":
    metaconfig = MetaConfig(
        project_root=project_root, 
        verbose=False, 
    )
    
    save_dir = metaconfig.convert_path(train_config.save_dir)
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
            pkl.dump(train_config, f)
    
    train_objects = train_config.unroll(metaconfig)

    evaluate_fn = _get_evaluate_fn(metaconfig)

    train_objects['evaluator'] = evaluate_fn
    train_objects['wandb_config']['evaluator'] = evaluators

    train_model(**train_objects)

