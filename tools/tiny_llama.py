import torch
from transformers import LlamaConfig
from transformers import LlamaForCausalLM


def tiny_llama(target_model_path, **kargs):
    config = LlamaConfig.from_pretrained(target_model_path, **kargs)
    config.num_hidden_layers = 1
    model = LlamaForCausalLM._from_config(config, torch_dtype=torch.float16)
    return model


if __name__ == '__main__':
    model = tiny_llama('lmsys/vicuna-13b-v1.5-16k')
    print('saving...')
    model.save_pretrained('./output/tiny_llama')
    print('saved.')
