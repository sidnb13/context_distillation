from micro_config import MetaConfig, ConfigScript
from base_configs import AdaFactorConfig, AdamWConfig, PretrainedHFPjitModelConfig, project_root
from data import Seq2SeqDataset
from incoder_config import IncoderModelConfig
from incoder_core import IncoderTrainConfig
from tk_jax.data import NatInstSeq2SeqConfig, NatInstSeq2SeqGeneratorConfig
from t5_config import T5ModelConfig
from itertools import product
from core import TKInference, TKTrainConfig, block_tokens
from nat_inst.random_data_gen import TKInstructDataSetting
from tk_jax.finetune_loop import TrainLoopConfig, EvaluateLossConfig, evaluate_loss, train_model
from tk_jax.eval_inference import TKInstructEvaluationConfig, tk_instruct_evaluate
import os
import pickle as pkl
from dataclasses import dataclass
from exact_match_eval import ExactMatchEvaluationConfig, exact_match_evaluate

@dataclass
class ScanPretrainConfig(ConfigScript):
    pkl_path: str
    split_idx: int
    enc_len: int
    dec_len: int
    model_tokenizer: PretrainedHFPjitModelConfig

    def unroll(self, metaconfig: MetaConfig) -> Seq2SeqDataset:
        _, _, tokenizer, _ = self.model_tokenizer.unroll(metaconfig)
        in_tokens, out_tokens = [], []
        with open(metaconfig.convert_path(self.pkl_path), 'rb') as f:
            d = pkl.load(f)[self.split_idx]
        for item in d:
            input_str, output_str = item['prompt'], item['completion']
            in_tokens.append(tokenizer(input_str)['input_ids'])
            out_tokens.append(tokenizer(output_str)['input_ids'])
        in_tokens = block_tokens(in_tokens, self.enc_len, tokenizer.pad_token_id)
        out_tokens = block_tokens(out_tokens, self.dec_len, tokenizer.pad_token_id)
        return Seq2SeqDataset(in_tokens, out_tokens, d)

model = IncoderModelConfig(
    # model_str="google/t5-v1_1-xl", 
    # model_str="t5-3b", 
    # model_str="google/ul2", 
    model_str='facebook/incoder-6B', 
    # model_str="allenai/tk-instruct-11b-def-pos-neg-expl", 
    checkpoint_path=None, 
    from_pretrained=True, 
    use_fp16=True, 
    gradient_checkpoint=True, 
)

# get natural instructions settings

train_dataset = ScanPretrainConfig(
    pkl_path='data/scan/pre_ft_train_test_prompt_completions_2.pkl', 
    split_idx=0, 
    enc_len=256*3, 
    dec_len=256, 
    model_tokenizer=model, 
)

eval_dataset = ScanPretrainConfig(
    pkl_path='data/scan/pre_ft_train_test_prompt_completions_2.pkl', 
    split_idx=1, 
    enc_len=256*3, 
    dec_len=256, 
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

trainer = IncoderTrainConfig(
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
    "inference": (ExactMatchEvaluationConfig(
        eval_dataset=eval_dataset, 
        inference=trainer, 
        rng=2, 
        bsize=32, 
        eval_batches=None, 
        save_generations_path='outputs/scan_pretrain_test1/generated_outputs.pkl', 
        generation_kwargs={
            'max_length': 256+256*3, 
            'do_sample': False, 
            'num_beams': 1, 
            'eos_token_id': 33, 
        }, 
        verbose=True, 
    ), exact_match_evaluate), 
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
    save_dir='outputs/scan_pretrain_test1/', 
    max_checkpoints=None, 
    epochs=1000, 
    max_steps=None, 
    bsize=8, 
    prefetch_batches=None, 
    log_every=256, 
    eval_every=1024, 
    save_every=None, 
    save_only_at_end=False, 
    use_wandb=True, 
    wandb_project='scan_pretrain', 
    wandb_run_name='scan_pretrain_test1', 
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
