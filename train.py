import os
import sys
import copy
import json
import torch
from pdb import set_trace
from dataclasses import dataclass

import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig

import transformers
from transformers import LlamaConfig
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM

#data_file = './data/alpaca_data.json'
data_file = './data/arqmath-pairs.json'
load_model = True
en_param_offload = False

### Parse Arguments
@dataclass
class MyArguments:
    model_name_or_path: str

parser = transformers.HfArgumentParser(
    (transformers.TrainingArguments, MyArguments)
)
training_args, my_args = parser.parse_args_into_dataclasses()

model_path = my_args.model_name_or_path
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
model_path = os.path.expanduser(model_path)

### Zero Configuration
with open('ds_config_zero3.json', 'r') as fh:
    ds_config = json.load(fh)
## enable parameter cpu offload?
if en_param_offload:
    ds_config['zero_optimization']['offload_param'] = {
        "device": "cpu",
        "pin_memory": True
    }

# this has to be run before loading the model.from_pretrained()
ds_config_hf = HfDeepSpeedConfig(ds_config)

# Model and LoRa Adapter
if load_model:
    model = LlamaForCausalLM.from_pretrained(model_path,
        cache_dir='./data', torch_dtype=torch.float16)

    from peft import LoraConfig, get_peft_model
    TARGET_MODULES = [
        "q_proj",
        "v_proj",
    ]
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8, lora_dropout=0.05,
        lora_alpha=16, bias='none',
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
else:
    model = None

### Tokenizer
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    if load_model:
        model.resize_token_embeddings(len(tokenizer))

tokenizer = LlamaTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=model,
    )

### Dataset
from datasets import load_dataset

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def _tokenize_fn(strings, tokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources, targets, tokenizer):
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def train_tokenize_function(examples, tokenizer):
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    if 'input' in examples:
        sources = [
            prompt_input.format_map(dict(instruction=instruction, input=input)) if input != "" \
            else prompt_no_input.format_map(dict(instruction=instruction)) \
            for instruction, input in zip(examples['instruction'], examples['input'])
        ]
    else:
        sources = [
            prompt_no_input.format_map(dict(instruction=instruction)) \
            for instruction in examples['instruction']
        ]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples['output']]

    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict


raw_train_datasets = load_dataset('json', encoding='utf-8',
    data_files=data_file, split="train", cache_dir='./cache')

train_dataset = raw_train_datasets.map(
    train_tokenize_function,
    batched=True,
    batch_size=3000,
    num_proc=32,
    remove_columns=raw_train_datasets.column_names,
    load_from_cache_file=True,
    desc="Running tokenizer on train dataset",
    fn_kwargs={"tokenizer": tokenizer}
)
data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

if load_model:
    # Training
    from transformers import Trainer
    model.is_parallelizable = True
    model.model_parallel = True
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )
    trainer.train()
    trainer.save_state()
else:
    import pdb; pdb.set_trace()
