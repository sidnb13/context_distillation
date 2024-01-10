from typing import Optional, Tuple, Union
from dataclasses import dataclass
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils import ModelOutput
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import jax
from flax.core.frozen_dict import freeze, unfreeze, FrozenDict
from jax import lax
from transformers.generation_flax_utils import FlaxGenerationMixin
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.configuration_utils import PretrainedConfig


@dataclass
class EnsembleGenerationOutput(ModelOutput):
    logits: jnp.ndarray = None
    past_key_values: Optional[Tuple[Tuple[Tuple[jnp.ndarray]]]] = None

# NOTE: this may not work with models that are not based on gpt2
class EnsembleGeneration(FlaxGenerationMixin):
    
    def __init__(
        self, 
        model: FlaxPreTrainedModel, 
    ):
        self.config = model.config
        self.model = model
    
    def __call__(
        self,
        input_ids: Optional[jnp.ndarray] = None, 
        attention_mask: Optional[jnp.ndarray] = None, 
        past_key_values: Optional[Tuple[Tuple[Tuple[jnp.ndarray]]]] = None, 
        output_hidden_states: Optional[bool] = True, 
        params: dict = None, 
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        model_outputs = self.model(input_ids, attention_mask=attention_mask, 
                                   past_key_values=past_key_values, **kwargs, 
                                   output_hidden_states=output_hidden_states, 
                                   params=params, train=False)
        logits = model_outputs.logits
        kvs = model_outputs.past_key_values

        ensemble_logits = jnp.log(jnp.mean(jax.nn.softmax(logits, axis=-1), axis=0, keepdims=True)+1e-7)

        return EnsembleGenerationOutput(logits=ensemble_logits, past_key_values=kvs)
    
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length), dtype=jnp.int32)
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.model.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )

        cache = unfreeze(init_variables["cache"])

        return cache
    
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jnp.DeviceArray] = None):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since GPT2 uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
