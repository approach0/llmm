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

import sys
sys.path.insert(0, '../FastChat')
from flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)
use_flash_att2 = True
if use_flash_att2:
    replace_llama_attn_with_flash_attn()


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

default_prompt = '''
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
Give three tips for staying healthy.

### Input:

### Response:
'''
def inference(prompt=default_prompt):
    print('rank', local_rank, end='\n\n')
    #inputs = tokenizer(prompt, return_tensors="pt")
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    inputs = inputs.to(device=local_rank)
    use_cache = False if use_flash_att2 else True
    with torch.no_grad():
        outputs = model.generate(inputs, use_cache=use_cache,
            synced_gpus=True, max_new_tokens=128,
            do_sample=True)
        #outputs = model.generate(**inputs)
    text_out = tokenizer.decode(outputs[0],
        skip_special_tokens=True)

    if local_rank == 0:
        print(f"{prompt} {text_out}")
    return text_out

if local_rank != 0:
    while True:
        inference()
else:
    import gradio as gr
    iface = gr.Interface(
        fn=inference,
        inputs=gr.Textbox(default_prompt, lines=40),
        outputs=gr.Textbox(lines=40)
    )
    # Enabling the queue for inference times > 60 seconds:
    iface.queue().launch(server_port=8922,
        debug=True, share=True, inline=False
    )
