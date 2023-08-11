import os
import sys
import json
import torch
from torch import autocast
from dataclasses import dataclass
from typing import Optional

import deepspeed
import transformers
from transformers.deepspeed import HfDeepSpeedConfig

from transformers import LlamaConfig
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM

from transformers import BitsAndBytesConfig
#from auto_gptq import AutoGPTQForCausalLM

import sys

from flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

### Parse Arguments
@dataclass
class MyArguments:
    model_name_or_path: str
    ctx_length: int
    use_flash_att2: bool
    load_8bit: bool
    deepspeed: str
    interface_port: int
    infer_interface: str
    specified_tokenizer: Optional[str] = None
    adapter_path: Optional[str] = None

parser = transformers.HfArgumentParser(MyArguments)
my_args = parser.parse_args_into_dataclasses()[0]
assert my_args.infer_interface in ['test', 'gradio', 'flask']

with open(my_args.deepspeed) as fh:
    config = json.load(fh)
print(my_args)
print(config)

HfDeepSpeedConfig(config) # before loading model

if my_args.use_flash_att2:
    replace_llama_attn_with_flash_attn()

torch.cuda.set_device(local_rank)
model_path = os.path.expanduser(my_args.model_name_or_path)
if my_args.specified_tokenizer:
    tokenizer_path = os.path.expanduser(my_args.specified_tokenizer)
else:
    tokenizer_path = model_path
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

print('Loading model ...')
if my_args.load_8bit:
    model = LlamaForCausalLM.from_pretrained(model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,  quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    )
    #model = AutoGPTQForCausalLM.from_quantized(model_path, device=local_rank)
else:
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

if my_args.adapter_path:
    print('Loading adapter ...')
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, my_args.adapter_path,
        adapter_name='default', is_trainable=False)
    # now the model type is still PeftModel, for compatibility, convert it to LLaMA.
    model = model.merge_and_unload()

print('Initializing DeepSpeed ...')
ds_engine = deepspeed.initialize(model=model, config=config)[0]
model = ds_engine.module
model.eval() # for inference


default_prompt = r'''
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
Answer a math question in the input.
The input is also followed by some potentially relevant passages to assist you.

If you find any passage(s) very helpful, feel free to tell me, and utilize them to guide your answer as much as possible.

Remember to indicate your final answer in boxed LaTeX. For example, if you think the final answer is \sqrt{3}, write it as \boxed{\sqrt{3}} (in boxed LaTeX) at the very end of your output.

### Input:

### Response:
'''
def inference(prompt):
    print('inference rank', local_rank, end='\n\n')
    #inputs = tokenizer(prompt, return_tensors="pt")
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    inputs = inputs.to(device=local_rank)
    use_cache = False if my_args.use_flash_att2 else True
    with torch.no_grad():
        outputs = model.generate(inputs, use_cache=use_cache,
            synced_gpus=True, max_new_tokens=1024,
            do_sample=True)
    text_out = tokenizer.decode(outputs[0],
        skip_special_tokens=True)

    if local_rank == 0:
        print(f"{prompt} {text_out}")
    return text_out


if local_rank == 0:
    if my_args.infer_interface == 'test':
        inference('count from 1 to 10.')

    elif my_args.infer_interface == 'gradio':
        import gradio as gr
        iface = gr.Interface(
            fn=inference,
            inputs=gr.Textbox(default_prompt, lines=40),
            outputs=gr.Textbox(lines=40)
        )
        # Enabling the queue for inference times > 60 seconds:
        iface.queue().launch(
            server_port=my_args.interface_port,
            debug=True, share=True, inline=False
        )

    elif my_args.infer_interface == 'flask':
        from flask import Flask, request, jsonify
        app = Flask('ds_test API')

        @app.route('/generate', methods=['GET', 'POST'])
        def server_handler():
            j = request.json
            return inference(j['prompt'])

        app.run(
            port=my_args.interface_port, host="0.0.0.0")
else:
    while True: inference('noop.')
