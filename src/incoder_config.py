from functools import partial
from typing import Optional
import jax
import jax.numpy as jnp
# from transformers import FlaxT5ForConditionalGeneration, T5Config, AutoTokenizer, T5ForConditionalGeneration
from transformers_patch.xglm_config_remat import XGLMConfig
from transformers_patch.xglm_remat import FlaxXGLMForCausalLM
from micro_config import MetaConfig
from dataclasses import dataclass
from flax.core.frozen_dict import freeze
from jax.experimental import PartitionSpec as P
from transformers.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax
from base_configs import PretrainedHFPjitModelConfig, HFPjitModelResult
from utils.hf_utils import from_path
from transformers.tokenization_utils import PreTrainedTokenizer
import math
from transformers import AutoTokenizer

def patch_call(instance, func):
    class _(type(instance)):
        def __call__(self, *arg, **kwarg):
           return func(*arg, **kwarg)
    instance.__class__ = _

# PartitionSpec for incoder
# replicate the hidden dim and shard feed-forward and head dim
def _get_partition_rules_incoder():
    return [
        # embeddings
        (("model", "embed_tokens", "embedding"), P("mp", None)),
        # atention
        (("self_attn", "(k_proj|q_proj|v_proj)", "kernel"), P(None, "mp")),
        (("self_attn", "(k_proj|q_proj|v_proj)", "bias"), P("mp")),
        (("self_attn", "out_proj", "kernel"), P("mp", None)),
        (("self_attn", "out_proj", "bias"), None),
        # mlp
        (("fc1", "kernel"), P(None, "mp")),
        (("fc1", "bias"), P("mp")),
        (("fc2", "kernel"), P("mp", None)),
        (("fc2", "bias"), None),
        # layer norms
        (("model", "layer_norm", "bias"), None),
        (("model", "layer_norm", "scale"), None),
        (("self_attn_layer_norm", "bias"), None),
        (("self_attn_layer_norm", "scale"), None),
        (("final_layer_norm", "bias"), None),
        (("final_layer_norm", "scale"), None),
    ]

# Source: https://github.com/huggingface/transformers/tree/main/examples/research_projects/jax-projects/model_parallel
def load_incoder_from_pretrained(model_str, dtype, pad_token_id, n_tokens, gradient_checkpoint):
    model = FlaxXGLMForCausalLM.from_pretrained(model_str, _do_init=True, from_pt=True, dtype=dtype, pad_token_id=pad_token_id)
    params = model.params
    
    # pad embeddings
    emb = jnp.zeros((n_tokens, model.config.hidden_size))
    emb = emb.at[:50518, :].set(params["model"]["embed_tokens"]["embedding"])
    params["model"]["embed_tokens"]["embedding"] = emb
    
    config = XGLMConfig.from_pretrained(model_str, vocab_size=n_tokens, dtype=dtype, 
                                        pad_token_id=pad_token_id, 
                                        gradient_checkpoint=gradient_checkpoint)
    model = FlaxXGLMForCausalLM(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

def load_incoder_from_local_path(model_path, dtype, pad_token_id, n_tokens, gradient_checkpoint):
    params = from_path(FlaxXGLMForCausalLM, model_path)
    config = XGLMConfig.from_pretrained(model_path, vocab_size=n_tokens, dtype=dtype, pad_token_id=pad_token_id, gradient_checkpoint=gradient_checkpoint)
    model = FlaxXGLMForCausalLM(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

def load_incoder_from_random(model_str, dtype, pad_token_id, n_tokens, gradient_checkpoint, seed):
    config = XGLMConfig.from_pretrained(model_str, vocab_size=n_tokens, dtype=dtype, 
                                        pad_token_id=pad_token_id, 
                                        gradient_checkpoint=gradient_checkpoint)
    model = FlaxXGLMForCausalLM(config, _do_init=True, dtype=dtype, seed=seed)
    params = model.params
    model = FlaxXGLMForCausalLM(config, _do_init=False, dtype=dtype)
    return model, freeze(params)

@dataclass
class IncoderModelConfig(PretrainedHFPjitModelConfig):
    gradient_checkpoint: bool

    def unroll(self, metaconfig: MetaConfig):
        tokenizer = AutoTokenizer.from_pretrained(self.model_str)
        patch_call(tokenizer, partial(tokenizer.__call__, add_special_tokens=False))
        tokenizer.encode = partial(tokenizer.encode, add_special_tokens=False)
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

        with jax.default_device(jax.devices('cpu')[0]):
            dtype = self.get_dtype()

            n_tokens=int(2**math.ceil(math.log2(len(tokenizer))))

            if self.checkpoint_path is not None:
                model, params = load_incoder_from_local_path(self.checkpoint_path, dtype, 
                                                            tokenizer.pad_token_id, 
                                                            n_tokens, self.gradient_checkpoint)
            elif self.from_pretrained:
                model, params = load_incoder_from_pretrained(self.model_str, dtype, 
                                                            tokenizer.pad_token_id, 
                                                            n_tokens, self.gradient_checkpoint)
            else:
                model, params = load_incoder_from_random(self.model_str, dtype, 
                                                        tokenizer.pad_token_id, 
                                                        n_tokens, self.gradient_checkpoint, 0)

        params = model.to_fp32(params)
        shard_rules = _get_partition_rules_incoder()
        return HFPjitModelResult(model, params, tokenizer, shard_rules)

