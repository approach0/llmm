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


def infer(tokenizer_path, model_path, direct_inference=True):
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)

    ## Zero Configuration
    with open('ds_config_zero3.json', 'r') as fh:
        ds_config = json.load(fh)
    ds_config_hf = HfDeepSpeedConfig(ds_config)

    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    model = LlamaForCausalLM.from_pretrained(model_path)

    ds_engine = deepspeed.init_inference(model,
        mp_size=world_size, dtype=torch.half,
        checkpoint=None, replace_with_kernel_inject=True)
    model = ds_engine.module

    def inference(prompt='My name is Mariama, my favorite '):
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
        print(f'rank#{local_rank} output:', output)
        return output

    if local_rank == 0:
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


if __name__ == '__main__':
    import fire
    fire.Fire({
        'convert': convert,
        'infer': infer,
    })
