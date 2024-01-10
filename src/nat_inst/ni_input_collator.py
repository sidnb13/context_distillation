# SOURCE: https://github.com/yizhongw/Tk-Instruct/blob/950445f453758868a509d9c1966c8f150a2a0f61/src/ni_collator.py

import logging
import random
import string
from transformers.data.data_collator import *

logger = logging.getLogger(__name__)


@dataclass
class DataInputCollatorForNI:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_task_definition: bool = True
    num_examples: int = 0
    text_only: bool=False

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []
        for instance in batch:
            add_task_name = self.add_task_name
            add_task_definition = self.add_task_definition
            num_examples = self.num_examples

            task_input = ""
            # add the input first.
            task_input += "Now complete the following example -\n"
            task_input += f"Input: "
            if not task_input[-1] in string.punctuation:
                task_input += "."
            task_input += "\n"
            task_input += "Output: "
            
            task_name = ""
            if add_task_name:
                task_name += instance["Task"] + ". "

            definition = ""
            if add_task_definition:
                if isinstance(instance["Definition"], list):
                    definition = "Definition: " + instance["Definition"][0].strip() # TODO: should we use <Definition>?
                else:
                    definition = "Definition: " + instance["Definition"].strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                definition += "\n\n"
            
            # try to add positive examples.
            pos_examples = []

            # modified to random sample and shuffle the examples instead of just taking the first ones.
            examples_list = []
            if (len(instance["Positive Examples"])+len(instance["Negative Examples"])) > num_examples:
                examples_list = random.sample(instance["Positive Examples"]+instance["Negative Examples"], num_examples)
            else:
                examples_list = instance["Positive Examples"]+instance["Negative Examples"]
            random.shuffle(examples_list)

            for idx, pos_example in enumerate(examples_list):
                pos_example_str = f" Positive Example {idx+1} -\n"
                pos_example_str += f"Input: "
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n"
                pos_example_str += f" Output: {pos_example['input'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n"
                pos_example_str += "\n"
                if len(self.tokenizer(definition + " ".join(pos_examples) + pos_example_str + task_input)["input_ids"]) <= self.max_source_length:
                    pos_examples.append(pos_example_str)
                else:
                    break
            
            source = task_name + definition + "".join(pos_examples) + task_input
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))

        if self.text_only:
            model_inputs = {"inputs": sources}
        else:
            model_inputs = self.tokenizer(
                sources, 
                max_length=self.max_source_length, 
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)

        if "input" in batch[0]["Instance"] and batch[0]["Instance"]["input"]:
            # Randomly select one reference if multiple are provided.
            labels = [ex["Instance"]["input"] for ex in batch]
            if self.text_only:
                model_inputs["labels"] = labels
            else:
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        labels,
                        max_length=self.max_target_length,
                        padding=self.padding,
                        return_tensors=self.return_tensors,
                        truncation=True,
                        pad_to_multiple_of=self.pad_to_multiple_of
                    )
                label_mask = labels["attention_mask"].bool()
                model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)
        else:
            model_inputs["labels"] = None

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels") and not self.text_only:
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
            model_inputs["decoder_input_ids"] = decoder_input_ids
            
        return model_inputs

