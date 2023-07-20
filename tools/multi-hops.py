import sys
sys.path.insert(0, './pya0')

import json
import os
from rich import print

from test_chatgpt import OAI_API
from test_gpt4 import gpt4_complete
from test_vicuna import api_init as vicuna_api_init, api as vicuna_api

#api_init = lambda x: pass
#api = OAI_API().get_completion
#api = gpt4_complete
api_init = vicuna_api_init
api = vicuna_api

from pya0.index_manager import from_prebuilt_index
from pya0.replace_post_tex import replace_dollar_tex, replace_display_tex, replace_inline_tex
from pya0.transformer_eval import psg_encoder__dpr_default, searcher__docid_vec_flat_faiss
from pya0.visualize import output_html

MATH_path = '../datasets/MATH/test/precalculus'
topic = os.path.basename(MATH_path)
default_tokenizer = 'approach0/dpr-cocomae-220'
single_vec_model = 'approach0/dpr-cocomae-220'
prebuilt_index = 'arqmath-task1-dpr-cocomae-220-hnsw'

def search_init():
    index_path = from_prebuilt_index(prebuilt_index)
    encoder, enc_utils = psg_encoder__dpr_default(default_tokenizer, single_vec_model, 0, 0, 'cpu')
    searcher, _ = searcher__docid_vec_flat_faiss(index_path, None, enc_utils, 'cpu')
    return encoder, searcher

def search(encoder, query, topk=3):
    query = replace_dollar_tex(query)
    query = replace_display_tex(query)
    query = replace_inline_tex(query)
    results = searcher(query, encoder, topk=topk, debug=False)
    imath_results, dollar_results = [], []
    for i, res in enumerate(results):
        d = res[2][1]
        imath_results.append(d)
        d = d.replace(r'[imath]', '$')
        d = d.replace(r'[/imath]', '$')
        dollar_results.append(d)
    return imath_results, dollar_results


template_prompt_init = r'''You are a mathematician, here is a math problem that I need you to solve:

--- PROBLEM BEGIN ---
{Q}
--- PROBLEM END   ---

To assist you, I have found some potentially relevant passages about this problem:

--- PASSAGE BEGIN ---
{P1}
--- PASSAGE END   ---

--- PASSAGE BEGIN ---
{P2}
--- PASSAGE END   ---

--- PASSAGE BEGIN ---
{P3}
--- PASSAGE END   ---

First, try to understand above passages and tell me which one(s) is/are useful to this problem.

Then, please try to utilize above passages as hints, think step by step, and derive the final answer.

Remember to indicate your final answer in boxed LaTeX. For example, if you think the final answer is \sqrt{3}, write it as \boxed{\sqrt{3}} at the very end of your output.

Keep your answer concise, you only have 500 tokens to generate!
'''


print('Initializing LLM...')
api_args = api_init()

print('Loading model...')
encoder, searcher = search_init()

for filename in os.listdir(MATH_path):
    json_path = os.path.join(MATH_path, filename)
    with open(json_path, 'r') as fh:
        j = json.load(fh)
    query = j['problem']
    solution = j['solution']
    prompt = template_prompt_init
    prompt = prompt.replace('{Q}', query)

    imath_results, dollar_results = search(encoder, query)
    prompt = prompt.replace('{P1}', dollar_results[0])
    prompt = prompt.replace('{P2}', dollar_results[1])
    prompt = prompt.replace('{P3}', dollar_results[2])

    print(f'[blue]Prompt{len(prompt)}[/blue][grey70]{prompt}[/grey70]', end='\n\n')
    out = api(prompt, args=api_args, debug=True)
    print(out, end='\n\n')
    print(f'[yellow]Solution[/yellow]: [green]{solution}[/green]')
    input('[red]Press Enter for the next question...[/red]')
