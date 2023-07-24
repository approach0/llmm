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
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

    model.eval()
    model.is_parallelizable = True
    model.model_parallel = True

    ds_engine = deepspeed.init_inference(model, mp_size=world_size,
        dtype=torch.half, checkpoint=None, replace_with_kernel_inject=True)
    model = ds_engine.module

    def inference(prompt='hello!'):
        print('prompt:', prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        device = torch.cuda.current_device()
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                max_new_tokens=1024,
                do_sample=False
            )
        output = tokenizer.decode(generation_output[0])
        if local_rank == 0:
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

    torch.distributed.barrier()


def quantize(tokenizer_path, model_path, quantized_model_path):
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    quantize_config = BaseQuantizeConfig(
        bits=4, group_size=128, desc_act=False
    )

    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    examples = [
        tokenizer(
            "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
        )
    ]
    model = AutoGPTQForCausalLM.from_pretrained(model_path,
        quantize_config)
    model.to('cuda')
    model.quantize(examples)
    model.save_quantized(quantized_model_path)


if __name__ == '__main__':
    import fire
    fire.Fire({
        'convert': convert,
        'quantize': quantize,
        'infer': infer,
    })
