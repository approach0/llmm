import os
import sys
import copy
import json
import torch
from torch import autocast
from pdb import set_trace
from dataclasses import dataclass
from typing import Optional

import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig

import transformers
from transformers import LlamaConfig
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM

from transformers import BitsAndBytesConfig


### Parse Arguments
@dataclass
class MyArguments:
    model_name_or_path: str
    data_file: str
    dryrun: bool
    ctx_length: int
    datamap_nprocs: int
    use_flash_att2: bool
    load_8bit: bool
    cache_dir: Optional[str] = None
    specified_tokenizer: Optional[str] = None
    debug_single_layer: Optional[bool] = False


parser = transformers.HfArgumentParser(
    (transformers.TrainingArguments, MyArguments)
)
train_args, my_args = parser.parse_args_into_dataclasses()
ds_config = train_args.hf_deepspeed_config
ds_config_json = ds_config.config

### Pre-Process Arguments
model_path = my_args.model_name_or_path
model_path = os.path.expanduser(model_path)
if my_args.specified_tokenizer:
    tokenizer_path = os.path.expanduser(my_args.specified_tokenizer)
else:
    tokenizer_path = model_path

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

torch.cuda.set_device(local_rank)

if len(ds_config.mismatches) > 0:
    for mismatch in ds_config.mismatches:
        print(mismatch)
    quit(1)
elif my_args.load_8bit and my_args.use_flash_att2:
    print('Flash Attention may be incompatible with 8bit.')
elif my_args.load_8bit and world_size > 1:
    print('8-bit multi-gpu training is not supported.')
    quit(1)
else:
    print(my_args)
    print(train_args)
    print(json.dumps(ds_config_json, indent=2))

if my_args.use_flash_att2:
    from flash_attn_monkey_patch import (
        replace_llama_attn_with_flash_attn,
    )
    replace_llama_attn_with_flash_attn()

# Model and LoRa Adapter
if not my_args.dryrun:
    if my_args.debug_single_layer:
        from tools.tiny_llama import tiny_llama
        load_func = tiny_llama
    else:
        load_func = LlamaForCausalLM.from_pretrained

    if my_args.load_8bit:
        model = load_func(model_path, cache_dir=my_args.cache_dir,
            use_cache=(False if my_args.use_flash_att2 else True),
            load_in_8bit=True,  quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        )
    else:
        model = load_func(model_path, cache_dir=my_args.cache_dir,
            use_cache=(False if my_args.use_flash_att2 else True)
        )

    from peft import LoraConfig, get_peft_model
    TARGET_MODULES = [
        "q_proj",
        "v_proj",
    ]
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8, lora_dropout=0.05,
        lora_alpha=16, bias='none',
        target_modules=TARGET_MODULES
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
    if not my_args.dryrun:
        model.resize_token_embeddings(len(tokenizer))

tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
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
    ctx_length = my_args.ctx_length
    max_length = tokenizer.model_max_length if ctx_length == 'max' else ctx_length
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
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
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def train_tokenize_function(examples, tokenizer):
    prompt_input = PROMPT_DICT["prompt_input"]
    prompt_no_input = PROMPT_DICT["prompt_no_input"]
    if 'input' in examples:
        sources = [
            prompt_input.format_map(dict(instruction=instr, input=input)) if input != "" \
            else prompt_no_input.format_map(dict(instruction=instr)) \
            for instr, input in zip(examples['instruction'], examples['input'])
        ]
    else:
        sources = [
            prompt_no_input.format_map(dict(instruction=instr)) \
            for instr in examples['instruction']
        ]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples['output']]

    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict


raw_train_datasets = load_dataset('json', encoding='utf-8',
    data_files=my_args.data_file, split="train", cache_dir='./cache')

train_dataset = raw_train_datasets.map(
    train_tokenize_function,
    batched=True,
    batch_size=3_000,
    num_proc=my_args.datamap_nprocs,
    remove_columns=raw_train_datasets.column_names,
    load_from_cache_file=True,
    desc="Running tokenizer on train dataset",
    fn_kwargs={"tokenizer": tokenizer}
)
data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

### Training
if not my_args.dryrun:
    # Training
    from transformers import Trainer
    model.is_parallelizable = True
    model.model_parallel = True
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )
    with autocast(device_type="cuda"):
        trainer.train()
else:
    tokenizer.save_pretrained('output/dryrun_tokenizer')
    #import pdb; pdb.set_trace()
