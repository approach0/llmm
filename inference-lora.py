import os
import json
import torch

from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM

from peft import PeftModel

import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig


def convert(origin_model_path, adapter_path, output_path='./tmp'):
    model = LlamaForCausalLM.from_pretrained(origin_model_path,
        cache_dir='./data', torch_dtype=torch.float16)
    model_and_lora = PeftModel.from_pretrained(model, adapter_path,
        'default')
    model = model_and_lora.merge_and_unload()
    print(model)
    model.save_pretrained(output_path)


default_prompt = '''
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
Give three tips for staying healthy.

### Input:

### Response:
'''


def infer(tokenizer_path, model_path,
    direct_inference=True, device='cuda:0'):

    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    model = LlamaForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    def inference(prompt=default_prompt):
        print('prompt:', prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                max_new_tokens=128,
                do_sample=False
            )
        output = tokenizer.decode(generation_output[0])
        print('output:', output)
        return output

    if direct_inference:
        inference()
    else:
        iface = gr.Interface(fn=inference,
            inputs="text", outputs="text")
        # Enabling the queue for inference times > 60 seconds:
        iface.queue().launch(
            debug=True, share=True, inline=False
        )


if __name__ == '__main__':
    import fire
    fire.Fire({
        'convert': convert,
        'infer': infer,
    })
