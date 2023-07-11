import os
import sys
import json
import torch

import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig

from transformers import LlamaConfig
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM

model_path = sys.argv[-1]
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

with open('ds_config_zero3.json') as fh:
    ds_config = json.load(fh)
ds_config['train_batch_size'] = world_size * 16
# this has to be run before loading the model.from_pretrained()
ds_config_hf = HfDeepSpeedConfig(ds_config)

torch.cuda.set_device(local_rank)
model_path = os.path.expanduser(model_path)

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path,
    torch_dtype=torch.float16)

ds_engine = deepspeed.initialize(model=model, config=ds_config)[0]
ds_engine.module.eval() # for inference

text_in = 'My name is Mariama, my favorite '
text_in = "When I was 6 my sister was half my age. Now I'm 70, my sister's age is "
inputs = tokenizer.encode(text_in, return_tensors="pt").to(device=local_rank)
with torch.no_grad():
    outputs = ds_engine.module.generate(inputs,
        synced_gpus=True, max_new_tokens=32, do_sample=True)
text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
if local_rank == 0:
    print(f"{text_in} {text_out}")
