import sys
sys.path.insert(0, '../toolformer-pytorch')

import torch
from transformers import LlamaForCausalLM

model_path = './output/7B-lora-trained'
model_path = './output/tiny_llama'

model = LlamaForCausalLM.from_pretrained(model_path,
    torch_dtype=torch.float16)
print('model loaded')
