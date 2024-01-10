from micro_config import ConfigScript, MetaConfig
from dataclasses import dataclass
from core import block_tokens, prepend_pad, prepend_ul2_autoregressive_sentenal
from base_configs import PretrainedHFPjitModelConfig
from data import Seq2SeqDataset, Seq2SeqIterableDataset
from typing import Union, Any, List, Tuple, Iterator, Optional, Callable, Iterable
from nat_inst.random_data_gen import TKInstructDataSetting, rand_data_gen
from nat_inst.random_input_data_generator import rand_input_data_gen, TKInstructInputDataSetting
import jax

@dataclass
class NatInstSeq2SeqConfig(ConfigScript):
    tsv_path: str
    enc_len: int
    dec_len: int
    add_ar_sentinal: bool
    target_prepend_pad: bool
    model_tokenizer: PretrainedHFPjitModelConfig

    def unroll(self, metaconfig: MetaConfig) -> Seq2SeqDataset:
        _, _, tokenizer, _ = self.model_tokenizer.unroll(metaconfig)
        in_tokens, out_tokens = [], []
        with open(metaconfig.convert_path(self.tsv_path), 'r') as f:
            for line in f:
                input_str, output_str = line[:-1].split("\t")
                if self.add_ar_sentinal:
                    input_str = prepend_ul2_autoregressive_sentenal(input_str)
                if self.target_prepend_pad:
                    output_str = prepend_pad(output_str)
                in_tokens.append(tokenizer(input_str)['input_ids'])
                out_tokens.append(tokenizer(output_str)['input_ids'])
        in_tokens = block_tokens(in_tokens, self.enc_len, tokenizer.pad_token_id)
        out_tokens = block_tokens(out_tokens, self.dec_len, tokenizer.pad_token_id)
        return Seq2SeqDataset(in_tokens, out_tokens, None)

@dataclass
class NatInstSeq2SeqGeneratorConfig(ConfigScript):
    data_path: str
    task_path: str
    ni_dataset_script_path: str
    max_num_instances_per_task: Optional[int]
    max_num_instances_per_eval_task: Optional[int]
    enc_len: int
    dec_len: int
    split: str
    rng: int
    data_settings: List[TKInstructDataSetting]
    add_ar_sentinal: bool
    target_prepend_pad: bool
    model_tokenizer: PretrainedHFPjitModelConfig

    def unroll(self, metaconfig: MetaConfig) -> Seq2SeqIterableDataset:
        _, _, tokenizer, _ = self.model_tokenizer.unroll(metaconfig)
        data = rand_data_gen(
            data_path=metaconfig.convert_path(self.data_path), 
            task_path=metaconfig.convert_path(self.task_path), 
            ni_dataset_script_path=metaconfig.convert_path(self.ni_dataset_script_path), 
            tokenizer=tokenizer, 
            max_num_instances_per_task=self.max_num_instances_per_task, 
            max_num_instances_per_eval_task=self.max_num_instances_per_eval_task, 
            max_source_length=self.enc_len, 
            max_target_length=self.dec_len, 
            split=self.split, 
            rng=jax.random.PRNGKey(self.rng), 
            settings=self.data_settings, 
        )

        def _iter():
            while True:
                input_str, output_str = next(data)
                if self.add_ar_sentinal:
                    input_str = prepend_ul2_autoregressive_sentenal(input_str)
                if self.target_prepend_pad:
                    output_str = prepend_pad(output_str)
                in_tokens = block_tokens([tokenizer(input_str)['input_ids']], self.enc_len, tokenizer.pad_token_id)[0]
                out_tokens = block_tokens([tokenizer(output_str)['input_ids']], self.dec_len, tokenizer.pad_token_id)[0]
                yield in_tokens, out_tokens, None
        
        return Seq2SeqIterableDataset(_iter())

@dataclass
class NatInstInputsSeq2SeqGeneratorConfig(ConfigScript):
    data_path: str
    task_path: str
    ni_dataset_script_path: str
    max_num_instances_per_task: Optional[int]
    max_num_instances_per_eval_task: Optional[int]
    enc_len: int
    dec_len: int
    split: str
    rng: int
    data_settings: List[TKInstructInputDataSetting]
    add_ar_sentinal: bool
    target_prepend_pad: bool
    model_tokenizer: PretrainedHFPjitModelConfig

    def unroll(self, metaconfig: MetaConfig) -> Seq2SeqIterableDataset:
        _, _, tokenizer, _ = self.model_tokenizer.unroll(metaconfig)
        data = rand_input_data_gen(
            data_path=metaconfig.convert_path(self.data_path), 
            task_path=metaconfig.convert_path(self.task_path), 
            ni_dataset_script_path=metaconfig.convert_path(self.ni_dataset_script_path), 
            tokenizer=tokenizer, 
            max_num_instances_per_task=self.max_num_instances_per_task, 
            max_num_instances_per_eval_task=self.max_num_instances_per_eval_task, 
            max_source_length=self.enc_len, 
            max_target_length=self.dec_len, 
            split=self.split, 
            rng=jax.random.PRNGKey(self.rng), 
            settings=self.data_settings, 
        )

        def _iter():
            while True:
                input_str, output_str = next(data)
                if self.add_ar_sentinal:
                    input_str = prepend_ul2_autoregressive_sentenal(input_str)
                if self.target_prepend_pad:
                    output_str = prepend_pad(output_str)
                in_tokens = block_tokens([tokenizer(input_str)['input_ids']], self.enc_len, tokenizer.pad_token_id)[0]
                out_tokens = block_tokens([tokenizer(output_str)['input_ids']], self.dec_len, tokenizer.pad_token_id)[0]
                yield in_tokens, out_tokens, None
        
        return Seq2SeqIterableDataset(_iter())
