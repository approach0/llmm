import os
import json
import random
import time
import torch
import numpy as np
from rich import print as rich_print

from test_chatgpt import OAI_API
from test_gpt4 import gpt4_complete
from test_vicuna import api_init as vicuna_api_init, api as vicuna_api

import sys
sys.path.insert(0, './pya0')
sys.path.insert(0, '../Progressive-Hint')
sys.path.insert(0, '../math/modeling')

from pya0.index_manager import from_prebuilt_index
from pya0.replace_post_tex import replace_dollar_tex, replace_display_tex, replace_inline_tex
from pya0.transformer_eval import psg_encoder__dpr_default, searcher__docid_vec_flat_faiss
from pya0.visualize import output_html

from main_clean import extract_math_answer
from math_equivalence import is_equiv
from prompt_factory import *


dataset_prefix = 'datasets'
MATH_path = f'../{dataset_prefix}/MATH/test/number_theory'
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


def search(encoder, searcher, query, topk=3):
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


def map_query_log_path(inpath, run_name):
    inpath = inpath.rstrip('/')
    inpath = inpath.split('/')
    fname = inpath[-1] + '.log'
    start, end = inpath.index(dataset_prefix), -1
    outpath = '/'.join(inpath[start:end] + [run_name])
    return os.path.join('./output', outpath, fname)


def main(logname=None, run_pass=None, debug=False, args=None, prompt_mode=None):
    assert logname is not None
    assert prompt_mode in ['cot', 'ia', 'mhop', 'direct']

    if run_pass is None:
        print('please specify run_pass')
        quit(1)

    elif run_pass == 'vicuna':
        api_init = vicuna_api_init
        api = vicuna_api
        args = ['cuda', *args]

    elif run_pass == 'chatgpt':
        api_init = lambda *args: None
        api = OAI_API().get_completion

    elif run_pass == 'gpt4':
        api_init = lambda *args: None
        api = gpt4_complete

    else:
        raise NotImplemented

    reproducible()

    print('Initializing LLM...')
    api_args = api_init(*args)

    print('Loading model...')
    encoder, searcher = search_init()

    correct_cnt, total_cnt = 0, 0
    limit_tests = None

    filenames = os.listdir(MATH_path)
    filenames = filenames[:limit_tests] if isinstance(limit_tests, int) else filenames

    for filename in filenames:
        json_path = os.path.join(MATH_path, filename)
        logpath = map_query_log_path(json_path, f'logname__{logname}')
        os.makedirs(os.path.dirname(logpath), exist_ok=True)
        if os.path.exists(logpath):
            print(f'log exists: {logpath}')
            #time.sleep(1)
            continue

        with open(json_path, 'r') as fh:
            j = json.load(fh)
        query = j['problem']
        solution = j['solution']

        if prompt_mode == 'direct':
            prompt = direct1(query)

        elif prompt_mode == 'cot':
            prompt = cot1(query)

        elif prompt_mode == 'ia':
            imath_results, dollar_results = search(encoder, searcher, query)
            prompt = ia1(query, *dollar_results)

        else:
            raise NotImplemented

        def print_title(title):
            print('\n', '=' * 30, title, '=' * 30, end='\n\n')

        print_title('Problem')
        print(json_path)

        print_title(f'Prompt (len = {len(prompt)})')
        print(prompt)

        answer = api(prompt, args=api_args, debug=debug)
        if answer is None: # content_filter policy triggered
            continue

        print_title(f'Answer (len = {len(answer)})')
        print(answer, end='\n\n')

        print_title('Ground Truth')
        print(solution)

        boxed_answer = extract_math_answer(answer)
        boxed_solution = extract_math_answer(solution)

        print_title('Marking')
        print('agent answer:', boxed_answer)
        print('ground truth:', boxed_solution)
        equiv = is_equiv(boxed_answer, boxed_solution)
        if equiv:
            rich_print('[green] correct [/green]')
            correct_cnt += 1
        else:
            rich_print('[red] wrong [/red]')
        total_cnt += 1

        #input('Press Enter for the next question...')
        log_json = {
            'problem': json_path,
            'prompt': prompt,
            'answer': answer,
            'solution': solution,
            'agent_answer': boxed_answer,
            'ground_truth': boxed_solution,
            'is_equiv': equiv,
            'args': json.dumps(args)
        }
        with open(logpath, 'w') as fh:
            json.dump(log_json, fh, indent=2)
            print('Written log:', logpath)

        accuracy_percentage = correct_cnt / total_cnt * 100
        print(f'Accuracy: {correct_cnt} / {total_cnt} = {accuracy_percentage:.2f}%')


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(main)
