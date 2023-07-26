import sys
sys.path.insert(0, '../toolformer-pytorch')

import torch
from transformers import LlamaForCausalLM

model_path = './output/7B-lora-trained'
model_path = './output/tiny_llama'

dataset_name = 'dmayhem93/toolformer-v0-postprocessed'

model = LlamaForCausalLM.from_pretrained(model_path,
    torch_dtype=torch.float16)
print('model loaded')

from datasets import load_dataset
raw_data = load_dataset(dataset_name, cache_dir='./cache')

import pdb; pdb.set_trace()
