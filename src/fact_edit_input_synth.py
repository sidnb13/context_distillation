from typing import Callable, Any
from micro_config import ConfigScript, MetaConfig
from dataclasses import dataclass
from core import block_tokens, prepend_pad, prepend_ul2_autoregressive_sentenal
from base_configs import PretrainedHFPjitModelConfig
from data import Seq2SeqDataset
from injection_functions import format_input

@dataclass
class FactEditSeq2SeqConfig(ConfigScript):
    examples: Callable[[], Any]
    enc_len: int
    dec_len: int
    add_ar_sentinal: bool
    target_prepend_pad: bool
    model_tokenizer: PretrainedHFPjitModelConfig

    def unroll(self, metaconfig: MetaConfig) -> Seq2SeqDataset:
        examples = self.examples()

        input_strs = []
        output_strs = []
        
        for example in examples:

            main_fact_prompt = example['requested_rewrite']['prompt'].replace('{}', example['requested_rewrite']['subject'])
            new_fact = f"{main_fact_prompt} {example['requested_rewrite']['target_new']['str']}"
            old_fact = f"{main_fact_prompt} {example['requested_rewrite']['target_true']['str']}"
            
            for prompt in example['generation_prompts']:
                teacher_prompt = f"Definition: In this task, you will generate diverse and high quality input queries for training a language model to edit the fact \"{old_fact}\" to \"{new_fact}\". Now complete the following example - Input: generation Output:"
                input_strs.append(teacher_prompt)
                output_strs.append(prompt)

            for prompt in example['neighborhood_prompts']:
                teacher_prompt = f"Definition: In this task, you will generate diverse and high quality input queries for training a language model to edit the fact \"{old_fact}\" to \"{new_fact}\". Now complete the following example - Input: neighborhood Output:"
                input_strs.append(teacher_prompt)
                output_strs.append(prompt)

            for prompt in example['attribute_prompts']:
                teacher_prompt = f"Definition: In this task, you will generate diverse and high quality input queries for training a language model to edit the fact \"{old_fact}\" to \"{new_fact}\". Now complete the following example - Input: attribute Output:"
                input_strs.append(teacher_prompt)
                output_strs.append(prompt)
            
            for prompt in example['paraphrase_prompts']:
                teacher_prompt = f"Definition: In this task, you will generate diverse and high quality input queries for training a language model to edit the fact \"{old_fact}\" to \"{new_fact}\". Now complete the following example - Input: paraphrase Output:"
                input_strs.append(teacher_prompt)
                output_strs.append(prompt)

        _, _, tokenizer, _ = self.model_tokenizer.unroll(metaconfig)
        in_tokens, out_tokens = [], []
        for input_str, output_str in zip(input_strs, output_strs):
            if self.add_ar_sentinal:
                input_str = prepend_ul2_autoregressive_sentenal(input_str)
            if self.target_prepend_pad:
                output_str = prepend_pad(output_str)
            in_tokens.append(tokenizer(input_str)['input_ids'])
            out_tokens.append(tokenizer(output_str)['input_ids'])
        in_tokens = block_tokens(in_tokens, self.enc_len, tokenizer.pad_token_id)
        out_tokens = block_tokens(out_tokens, self.dec_len, tokenizer.pad_token_id)
        return Seq2SeqDataset(in_tokens, out_tokens, None)
