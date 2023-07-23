import os
import sys
import json
import torch

import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig

from transformers import LlamaConfig
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM

#from auto_gptq import AutoGPTQForCausalLM

tokenizer_path = sys.argv[-2]
model_path = sys.argv[-1]

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

with open('ds_config_zero3.json') as fh:
    ds_config = json.load(fh)
ds_config['train_batch_size'] = world_size
del ds_config['gradient_accumulation_steps']
del ds_config['train_micro_batch_size_per_gpu']

# this has to be run before loading the model.from_pretrained()
ds_config_hf = HfDeepSpeedConfig(ds_config)

torch.cuda.set_device(local_rank)
model_path = os.path.expanduser(model_path)

tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
model = LlamaForCausalLM.from_pretrained(model_path,
    torch_dtype=torch.float16)
#model = AutoGPTQForCausalLM.from_quantized(model_path, device=local_rank)

ds_engine = deepspeed.initialize(model=model, config=ds_config)[0]
model = ds_engine.module
model.eval() # for inference

text_in = '''
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
Give three tips for staying healthy.

### Input:

### Response:
'''
#inputs = tokenizer(text_in, return_tensors="pt")
inputs = tokenizer.encode(text_in, return_tensors="pt")
inputs = inputs.to(device=local_rank)
with torch.no_grad():
    outputs = model.generate(inputs,
        synced_gpus=True, max_new_tokens=128, do_sample=True)
    #outputs = model.generate(**inputs)
text_out = tokenizer.decode(outputs[0],
    skip_special_tokens=True)

if local_rank == 0:
    print(f"{text_in} {text_out}")
