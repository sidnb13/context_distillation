import contextlib
import math
from typing import Any, Dict, Optional, Union
from micro_config import ConfigScript, MetaConfig
from dataclasses import asdict, dataclass
from core import TKInference, TKInferenceConfig, TKTrainConfig
from data import Seq2SeqDataset, Seq2SeqIterableDataset, dataloader, Dataset
import jax
import json
from tqdm.auto import tqdm
from incoder_core import IncoderTrainConfig
from tk_jax.compute_metrics import compute_grouped_metrics, compute_metrics
import os
from jax.random import KeyArray
from jax.experimental.maps import Mesh
import pickle as pkl

@dataclass
class ExactMatchEvaluationConfig(ConfigScript):
    eval_dataset: ConfigScript
    inference: Union[TKInferenceConfig, TKTrainConfig]
    rng: int
    bsize: int
    eval_batches: Optional[int]
    save_generations_path: Optional[str]
    generation_kwargs: Dict[str, Any]
    verbose: bool

    def unroll(self, metaconfig: MetaConfig) -> Any:
        if isinstance(self.inference, TKTrainConfig) or isinstance(self.inference, IncoderTrainConfig):
            _, inference, _, mesh = self.inference.unroll(metaconfig)
        else:
            inference, _, mesh = self.inference.unroll(metaconfig)
        return {
            'eval_dataset': self.eval_dataset.unroll(metaconfig), 
            'inference': inference, 
            'mesh': mesh, 
            'rng': jax.random.PRNGKey(self.rng), 
            'bsize': self.bsize, 
            'eval_batches': self.eval_batches, 
            'save_generations_path': metaconfig.convert_path(self.save_generations_path), 
            'generation_kwargs': self.generation_kwargs, 
            'verbose': self.verbose, 
        }

def exact_match_evaluate(*, eval_dataset: Union[Seq2SeqDataset, Seq2SeqIterableDataset], 
                         inference: TKInference, mesh: Optional[Mesh], rng: KeyArray, 
                         bsize: int, eval_batches: Optional[int], 
                         save_generations_path: Optional[str], generation_kwargs: Dict[str, Any], 
                         verbose: bool):
        
        if mesh is None:
            mesh = contextlib.nullcontext

        # eval on batches
        inputs = []
        references = []
        predictions = []
        raw = []
        steps_per_epoch = int(math.ceil(len(eval_dataset) / bsize)) if isinstance(eval_dataset, Dataset) else None

        with mesh:
            d = dataloader(None, eval_dataset, bsize, trunc=False)
            for i, (items, meta) in tqdm(enumerate(d), total=steps_per_epoch, disable=jax.process_index() > 0):
                
                # conditionally terminate early
                if eval_batches is not None and i >= eval_batches:
                    break

                # get eval logs
                rng, new_rng = jax.random.split(rng)
                model_outputs = inference.generate_from_tokens(items['input_ids'], new_rng, **generation_kwargs)
                inputs.extend(inference.tokenizer.batch_decode(items['input_ids'], skip_special_tokens=True))
                references.extend(inference.tokenizer.batch_decode(items['decoder_input_ids'], skip_special_tokens=True))
                predictions.extend(inference.tokenizer.batch_decode(model_outputs, skip_special_tokens=True))
                raw.extend(meta)
        
        summary_results = compute_metrics(predictions, list(map(lambda x: [x], references)), xlingual=False)

        all_results = {'inputs': inputs, 'references': references, 'predictions': predictions, 'raw_data': raw, 'metrics': summary_results}

        if verbose:
            print('Summary:')
            print(summary_results)

        if save_generations_path is not None:
            if not os.path.exists(os.path.dirname(save_generations_path)):
                os.makedirs(os.path.dirname(save_generations_path))
            with open(save_generations_path, 'wb') as f:
                pkl.dump(all_results, f)
        return summary_results
