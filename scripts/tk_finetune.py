from micro_config import MetaConfig
from base_configs import AdaFactorConfig, AdamWConfig, project_root
from tk_jax.data import NatInstSeq2SeqConfig, NatInstSeq2SeqGeneratorConfig
from t5_config import T5ModelConfig
from itertools import product
from core import TKInference, TKTrainConfig
from nat_inst.random_data_gen import TKInstructDataSetting
from tk_jax.finetune_loop import TrainLoopConfig, EvaluateLossConfig, evaluate_loss, train_model
from tk_jax.eval_inference import TKInstructEvaluationConfig, tk_instruct_evaluate
import os
import pickle as pkl

model = T5ModelConfig(
    # model_str="google/t5-v1_1-xl", 
    # model_str="t5-3b", 
    # model_str="google/ul2", 
    model_str="google/t5-xxl-lm-adapt", 
    checkpoint_path=None, 
    from_pretrained=True, 
    use_fp16=True, 
    gradient_checkpoint=True, 
)

# get natural instructions settings

nat_inst_options = {'add_task_definition': [True], 'num_pos_examples': [0, 1, 2, 3], 
                    'num_neg_examples': [0], 'add_explanation': [True, False], 
                    'add_task_name': [False]}
nat_inst_settings = []
nat_inst_options_ks, nat_inst_options_vs = list(zip(*nat_inst_options.items()))
for items in product(*nat_inst_options_vs):
    nat_inst_settings.append(TKInstructDataSetting(**dict(zip(nat_inst_options_ks, items))))

train_dataset = NatInstSeq2SeqGeneratorConfig(
    data_path='data/nat_inst/splits/default/', 
    task_path='data/nat_inst/tasks/', 
    ni_dataset_script_path='src/nat_inst/ni_dataset.py', 
    max_num_instances_per_task=100, 
    max_num_instances_per_eval_task=100, 
    enc_len=1024, 
    dec_len=128, 
    split='train', 
    rng=0, 
    data_settings=nat_inst_settings, 
    add_ar_sentinal=False, 
    target_prepend_pad=True, 
    model_tokenizer=model, 
)

eval_dataset = NatInstSeq2SeqConfig(
    tsv_path='data/nat_inst/text2text/defintion_pos_2_neg_2_expl/test.tsv', 
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
    "inference": (TKInstructEvaluationConfig(
        eval_dataset=eval_dataset, 
        inference=trainer, 
        reference_file='data/nat_inst/text2text/defintion_pos_2_neg_2_expl/test_examples.jsonl', 
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
    ), tk_instruct_evaluate), 
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
    save_dir='outputs/T5_11B_tk_no_scan/', 
    max_checkpoints=None, 
    epochs=10, 
    max_steps=4864*2, 
    bsize=8, 
    prefetch_batches=None, 
    log_every=256, 
    eval_every=4864, 
    save_every=None, 
    save_only_at_end=True, 
    use_wandb=True, 
    wandb_project='natinst_finetune', 
    wandb_run_name='T5_11B_tk_no_scan', 
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
