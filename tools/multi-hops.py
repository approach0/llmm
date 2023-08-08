import os
import json
import random
import time
import torch
import numpy as np
from rich import print as rich_print

from test_chatgpt import OAI_API, agent as chatgpt_agent
from test_gpt4 import gpt4_complete
from test_vicuna import api_init as vicuna_api_init, api as vicuna_api
from test_sympy import compute as sympy_compute

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

from functools import partial
import requests


dataset_prefix = 'MATH'
default_tokenizer = 'approach0/dpr-cocomae-220'
single_vec_model = 'approach0/dpr-cocomae-220'
prebuilt_index = 'arqmath-task1-dpr-cocomae-220-hnsw'
sota_searchd_url = 'http://tuna.cs.uwaterloo.ca:8080/search'


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


def unwrap_dollars_for_text(test_str):
    import string
    allowed = set(
        string.ascii_letters +
        string.whitespace + '.' + '$')
    if set(test_str) <= allowed:
        return test_str.strip().strip('$')
    else:
        return test_str


def sota_search(question, keywords, full_topk=30, topk=3, gt=None):
    print('requesting:', sota_searchd_url)
    json = {
        'topk': full_topk
    }
    if gt is not None and 'manual_query' in gt:
        question = None # for now...
        if len(gt['manual_query']) > 0:
            keywords = gt['manual_query']
            keywords = list(map(lambda x: f'${x}$', keywords))
            print('use ground truth.')
    if question is not None:
        json['question'] = question
        print('query question:', question)
    if keywords is not None:
        keywords = list(map(unwrap_dollars_for_text, keywords))
        json['keywords'] = keywords
        print('query keywords:', keywords)

    res = requests.post(sota_searchd_url, json=json)
    if res.ok:
        res = res.json()
        res = res[:topk]
        def mapper(item):
            content, post_id, _ = item
            url = ('https://math.stackexchange.com/' +
                f'questions/{post_id}')
            return f'URL: {url}\n\n' + content
        return list(map(mapper, res))
    else:
        return []


def call_sympy(args):
    print('computing:', args)
    if isinstance(args, dict):
        return 'Error: passed in a dictionary. Use array!'
    return sympy_compute(*args)


def capture(string, par):
    stack = []
    started = False
    begin, end = 0, 0
    for i, c in enumerate(string):
        if c == par[0]:
            stack.append(c)
            if not started:
                begin = i
                started = True
        elif c == par[1]:
            stack.pop()
            if len(stack) == 0:
                end = i
                break
    return begin, end


def has_api_call(answer, api_map):
    if has_result(answer, api_map):
        return True
    for key in api_map.keys():
        if key in answer:
            idx = answer.find(key)
            begin, end = capture(answer[idx:], ('[', ']'))
            if begin < end:
                return True
    return False


def has_result(answer, api_map):
    if r'<|im_end|>' in answer:
        return True
    key = r'\boxed'
    if key in answer:
        idx = answer.find(key)
        begin, end = capture(answer[idx:], ('{', '}'))
        if begin < end:
            return True
    return False


def has_any_captured(answer, api_map):
    return has_result(answer, api_map) or has_api_call(answer, api_map)


def inject_result(answer, api_map):
    for api_name in api_map:
        idx = answer.find(api_name)
        if idx == -1: continue

        idx += len(api_name)

        begin, end = capture(answer[idx:], ('[', ']'))
        begin, end = begin + idx, end + idx
        if begin >= end:
            return answer[:idx] + '\n\n' + multihop_err1()
        else:
            injected = answer[:end+1] + '\n\n'

        api_args = answer[begin:end+1]
        try:
            api_args = json.loads(api_args)
            results = api_map[api_name](api_args)
            injected += multihop_results1(results)
        except json.decoder.JSONDecodeError:
            injected += multihop_err1('JSON decode error!\n' +
            'Double check your API call and try it again!')
        return injected
    return None


def map_query_log_path(inpath, run_name):
    inpath = inpath.rstrip('/')
    inpath = inpath.split('/')
    fname = inpath[-1] + '.log'
    start, end = inpath.index(dataset_prefix), -1
    outpath = '/'.join(inpath[start:end] + [run_name])
    return os.path.join('./output', outpath, fname)


def main(logname=None, run_pass=None, debug=False, max_ctx=8000, args=None,
    prompt_mode=None, topic=None, fname_filter=None, skip_existing=True,
    begin=0, end=None, metric='pass@1', ground_truth_dir=None):

    metric_name, k = metric.split('@')
    k = int(k)
    assert logname is not None
    assert (prompt_mode in ['cot', 'ia', 'direct', 'mh', 'manual', 'askkey']
        or prompt_mode.startswith('example'))
    assert metric_name in ['pass', 'maj']
    assert k > 0

    MATH_path = f'../{dataset_prefix}/test/{topic}'
    print('dataset path:', MATH_path)
    filenames = os.listdir(MATH_path)
    print('number of tests:', len(filenames))
    time.sleep(1)

    if run_pass is None:
        print('please specify run_pass')
        quit(1)

    elif run_pass == 'vicuna':
        llm_init = vicuna_api_init
        llm_api = vicuna_api
        llm_rst = lambda *args: None
        llm_args = ['cuda', *args]

    elif run_pass == 'chatgpt':
        llm_init = lambda *args: None

        #llm_api = OAI_API().get_completion
        #llm_rst = lambda *args: None
        llm_api = chatgpt_agent.complete
        llm_rst = chatgpt_agent.reset

        llm_args = []

    elif run_pass == 'gpt4':
        llm_init = lambda *args: None
        llm_api = gpt4_complete
        llm_rst = lambda *args: None
        llm_args = []

    else:
        raise NotImplemented

    reproducible()

    def print_title(title):
        print('\n', '=' * 15, title, '=' * 15, end='\n\n')


    print('Initializing LLM ...', llm_args)
    llm_api_args = llm_init(*llm_args)

    manual_query = []
    tot_okcnt, tot_cnt = 0, 0
    filenames = filenames[begin:end]

    for filename in filenames:
        json_path = os.path.join(MATH_path, filename)
        logpath = map_query_log_path(json_path, f'run__{logname}')

        if fname_filter is not None:
            if os.path.basename(json_path) != fname_filter:
                continue

        if ground_truth_dir:
            path = os.path.join(ground_truth_dir, filename + '.log')
            if os.path.exists(path):
                with open(path, 'r') as fh:
                    gt = json.load(fh)
            else:
                continue
        else:
            gt = None

        if os.path.exists(logpath) and skip_existing:
            print(f'log exists: {logpath}')
            continue
        else:
            os.makedirs(os.path.dirname(logpath), exist_ok=True)

        # read question
        with open(json_path, 'r') as fh:
            j = json.load(fh)
        query = j['problem']
        solution = j['solution']
        boxed_solution = extract_math_answer(solution)

        # set up judge buffer
        judge_buffer = []
        k_count = 0

        # setup API map
        api_map = {
            'SEARCH': partial(sota_search, query, gt=gt),
            'COMPUTE': call_sympy
        }
        llm_api_kargs = {}

        prompt = ''
        pas_okcnt = 0
        while k_count < k:
            llm_rst()

            print_title('Problem')
            print(json_path)

            # manual prompt?
            if prompt_mode == 'manual':
                while True:
                    print('current manual query:', manual_query)
                    cmd = input('Enter formula, "save", "reset", or "skip":\n')
                    if cmd.strip() == '':
                        k_count = 0
                        break
                    elif cmd == 'reset':
                        manual_query = []
                    elif cmd == 'skip':
                        manual_query = []
                        k_count = k
                        break
                    elif cmd == 'save':
                        k_count = k
                        break
                    else:
                        manual_query.append(cmd)

                metric_name = 'maj' # pass@k would break the loop.
                if k_count == k: continue
            else:
                k_count += 1

            # determine the prompt
            if prompt_mode == 'direct':
                prompt = direct2(query)

            elif prompt_mode.startswith('example'):
                prompt = example(prompt_mode)

            elif prompt_mode == 'cot':
                prompt = cot2(query)

            elif prompt_mode == 'ia':
                results = sota_search(query, None, topk=4, gt=gt)
                if len(results) > 0:
                    prompt = ia2(query, *results)
                else:
                    prompt = cot2(query)

            elif prompt_mode == 'mh':
                prompt = multihop1(query)

            elif prompt_mode == 'manual':
                if len(manual_query) == 0:
                    prompt = cot2(query)
                else:
                    kws = list(map(lambda x: '$' + x + '$', manual_query))
                    results = sota_search(None, kws, topk=4)
                    prompt = ia2(query, *results)

            elif prompt_mode == 'askkey':
                prompt = ask_identity_formula(query)
                k_count = k
                llm_api_kargs = {'stop': '\n\n'}

            else:
                raise NotImplemented

            print_title(f'Prompt (len = {len(prompt)})')
            print(prompt)
            #import pdb; pdb.set_trace()

            # answering
            answer = ''
            while len(prompt) < max_ctx:
                print_title(f'Answer (total prompt len: {len(prompt)})')
                answer = llm_api(prompt,
                    args=llm_api_args, debug=debug, api_map=api_map,
                    abort_criteria=has_any_captured, **llm_api_kargs
                )

                if answer is None: # e.g., content_filter policy triggered
                    continue

                if prompt_mode == 'mh':
                    injected = inject_result(answer, api_map)

                    if injected is not None:
                        print_title(f'Injected')
                        print(injected)
                        answer = injected

                    if has_result(answer, api_map):
                        break

                    prompt += answer
                else:
                    print(answer)
                    break

            boxed_answer = extract_math_answer(answer)

            # marking
            print_title('Ground Truth')
            print(solution)

            print_title(f'Marking ({filename} pass#{k_count}/{k})')
            print('agent answer:', boxed_answer)
            print('ground truth:', boxed_solution)
            equiv = is_equiv(boxed_answer, boxed_solution)
            if equiv:
                rich_print('[green]correct[/green]')
                tot_okcnt += 1
                pas_okcnt += 1
            else:
                rich_print('[red]wrong[/red]')
            tot_cnt += 1
            pas_accuracy = pas_okcnt / k * 100
            tot_accuracy = tot_okcnt / tot_cnt * 100
            print(f'kPass: {pas_okcnt} / {k} = {pas_accuracy:.2f}%')
            print(f'Total: {tot_okcnt} / {tot_cnt} = {tot_accuracy:.2f}%')

            judge_buffer.append({
                'answer': answer,
                'boxed_answer': boxed_answer,
                'is_equiv': equiv
            })

            # abort upon correct answer in pass@k
            if metric_name == 'pass' and equiv:
                break

        if prompt_mode.startswith('example'):
            input('Press Enter to save this example output...')

        log_json = {
            'problem': json_path,
            'prompt': prompt,
            'solution': solution,
            'ground_truth': boxed_solution,
            'judge_buffer': judge_buffer,
            'manual_query': manual_query if gt is None else gt['manual_query'],
            'args': json.dumps(args)
        }
        with open(logpath, 'w') as fh:
            json.dump(log_json, fh, indent=2)
            print('Written log:', logpath)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(main)
