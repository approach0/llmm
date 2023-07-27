import sys
sys.path.insert(0, '../toolformer-pytorch')

from toolformer_pytorch import Toolformer
from functools import partial

import torch
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer

tokenizer_path = '/home/w32zhong/llama-models/7B-hgf-new'
model_path = './output/7B-lora-trained'
#model_path = './output/tiny_llama'

tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
model = LlamaForCausalLM.from_pretrained(model_path,
    torch_dtype=torch.float16)
model.to('cuda:7')
print('model loaded')

#dataset_name = 'dmayhem93/toolformer-v0-postprocessed'
#from datasets import load_dataset
#raw_data = load_dataset(dataset_name, cache_dir='./cache')

prompt = f"""
Your task is to add calls to a Calendar API to a piece of text.
The API calls should help you get information required to complete the text.
You can call the API by writing "[Calendar()]"
Here are some examples of API calls:
Input: Today is the first Friday of the year.
Output: Today is the first [Calendar()] Friday of the year.
Input: The president of the United States is Joe Biden.
Output: The president of the United States is [Calendar()] Joe Biden.
Input: [input]
Output:
"""

data = [
    "The store is never open on the weekend, so today it is closed.",
    "The number of days from now until Christmas is 30",
    "The current day of the week is Wednesday."
]

def Calendar():
    import datetime
    from calendar import day_name, month_name
    now = datetime.datetime.now()
    return f'Today is {day_name[now.weekday()]}, {month_name[now.month]} {now.day}, {now.year}.'

toolformer = Toolformer(
    model = model,
    model_seq_len = 256,
    teach_tool_prompt = prompt,
    tool_id = 'Calendar',
    tool = Calendar,
    finetune = True,
    tokenizer_encode = partial(tokenizer.encode, add_special_tokens=False),
    tokenizer_decode = tokenizer.decode
)

filtered_stats = toolformer(data)
