import os
import json
import random
import torch
import numpy as np
from rich import print as rich_print

from test_chatgpt import OAI_API
from test_gpt4 import gpt4_complete
from test_vicuna import api_init as vicuna_api_init, api as vicuna_api

import sys
sys.path.insert(0, './pya0')
sys.path.insert(0, '../Progressive-Hint')

from pya0.index_manager import from_prebuilt_index
from pya0.replace_post_tex import replace_dollar_tex, replace_display_tex, replace_inline_tex
from pya0.transformer_eval import psg_encoder__dpr_default, searcher__docid_vec_flat_faiss
from pya0.visualize import output_html

from main_clean import extract_math_answer

#api_init = lambda *args: None
#api = OAI_API().get_completion

#api_init = lambda *args: None
#api = gpt4_complete

api_init = vicuna_api_init
api = vicuna_api


MATH_path = '../datasets/MATH/test/precalculus'
topic = os.path.basename(MATH_path)
default_tokenizer = 'approach0/dpr-cocomae-220'
single_vec_model = 'approach0/dpr-cocomae-220'
prebuilt_index = 'arqmath-task1-dpr-cocomae-220-hnsw'


def reproducible(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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

Keep your answer concise, you have 2048 tokens to finish answering it!
'''

reproducible()

print('Initializing LLM...')
api_args = api_init()

print('Loading model...')
encoder, searcher = search_init()

correct_cnt, total_cnt = 0, 0
limit_tests = None

filenames = os.listdir(MATH_path)
filenames = filenames[:limit_tests] if isinstance(limit_tests, int) else filenames

for filename in filenames:
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

    def print_title(title):
        print('\n', '=' * 30, title, '=' * 30, end='\n\n')

    print_title('Problem')
    print(json_path)

    print_title(f'Prompt (len = {len(prompt)})')
    print(prompt)

    answer = api(prompt, args=api_args, debug=False)
    print_title(f'Answer (len = {len(answer)})')
    print(answer, end='\n\n')

    print_title('Ground Truth')
    print(solution)

    boxed_answer = extract_math_answer(answer)
    boxed_solution = extract_math_answer(solution)

    print_title('Marking')
    print('agent answer:', boxed_answer)
    print('ground truth:', boxed_solution)
    if boxed_answer == boxed_solution:
        rich_print('[green] correct [/green]')
        correct_cnt += 1
    else:
        rich_print('[red] wrong [/red]')
    total_cnt += 1

    #input('Press Enter for the next question...')

accuracy_percentage = correct_cnt / total_cnt * 100
print(f'Accuracy: {correct_cnt} / {total_cnt} = {accuracy_percentage:.2f}%')
