import os
import json
import random
import time
import torch
import math
import numpy as np
from functools import partial
from rich import print as rich_print
from colorama import Fore, Style

from test_chatgpt import OAI_API, agent as chatgpt_agent
from test_gpt4 import gpt4_complete
from test_vicuna import api_init as vicuna_api_init, api as vicuna_api
from test_sympy import compute as sympy_compute

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '../Progressive-Hint')
sys.path.insert(0, '../math/modeling')

from main_clean import extract_math_answer
from math_equivalence import is_equiv
from prompt_factory import *
from inspect_output import _output_html


dataset_prefix = 'MATH'
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


def utils_llm_init(*raw_args):
    sys.path.insert(0, '.')
    import utils2
    args = []
    kargs = {}
    for arg in raw_args:
        if isinstance(arg, dict):
            kargs.update(arg)
        else:
            args.append(arg)
    mode = args[0]
    ctx_len = args[1]
    return mode, ctx_len, *utils2.load_model(*args[2:], **kargs)


def utils_llm_gen(prompt, args, **kargs):
    sys.path.insert(0, '.')
    import utils2
    mode, ctx_len, tokenizer, model = args
    return utils2.generate(mode=mode,
        tokenizer=tokenizer, model=model,
        prompt=prompt, debug=False, context_len=ctx_len)


def call_sympy(query, args):
    print('computing:', args)
    if not isinstance(args, list):
        return 'Passed argument format error. Use array!'
    elif len(args) < 2:
        return 'Error: at least two arguments needed!'
    return sympy_compute(*args)


def has_math(s):
    if '+' in s:
        return True
    elif '\\' in s:
        return True
    elif '^' in s:
        return True
    elif '!' in s:
        return True
    elif '*' in s:
        return True
    elif '(' in s:
        return True
    elif ')' in s:
        return True
    elif '[' in s:
        return True
    elif ']' in s:
        return True
    elif '|' in s:
        return True
    elif '<' in s:
        return True
    elif '>' in s:
        return True
    elif '{' in s:
        return True
    elif '}' in s:
        return True
    elif '=' in s:
        return True
    elif ':' in s:
        return True
    else:
        return False


def smart_correct(kw):
    import string
    onlytext = set(
        string.ascii_letters +
        string.whitespace + '.' + ',' + '$')
    kw = kw.strip()
    if set(kw) <= onlytext:
        return kw.strip().strip('$')
    else:
        if not kw.startswith('$') and has_math(kw):
            kw = '$' + kw
        if not kw.endswith('$') and has_math(kw):
            kw = kw + '$'
        return kw


def search_wrapper(tool, question, keywords, gt=None, **kargs):
    if gt is not None and 'manual_query' in gt:
        if len(gt['manual_query']) > 0:
            keywords = gt['manual_query']
            keywords = list(map(lambda x: f'${x}$', keywords))

    if keywords:
        keywords = list(map(smart_correct, keywords))

    print(Fore.CYAN)
    print('requesting:', tool)
    print('query question:', question)
    print('query keywords:', keywords)
    print(Style.RESET_ALL)

    if tool == 'mabowdor':
        from tools.test_mabowdor import search
        return search('mabowdor', question, keywords)

    elif tool == 'a0':
        from tools.test_mabowdor import search
        return search('mabowdor', None, keywords)

    elif tool == 'MATH':
        from tools.test_mabowdor import search
        return search('MATH', question, None)

    elif tool == 'online':
        from tools.test_a0xyz_search import sleepy_search_api
        return sleepy_search_api(keywords=keywords)

    else:
        raise NotImplemented


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


def inject_result(query, answer, api_map):
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
            results = api_map[api_name](query, api_args)
            injected += multihop_results1(results)
        except json.decoder.JSONDecodeError:
            injected += multihop_err1('JSON decode error!\n' +
            'Double check your API call and try it again!')
        return injected
    return None


def map_query_log_path(inpath, run_name, suffix='.log'):
    inpath = inpath.rstrip('/')
    inpath = inpath.split('/')
    fname = inpath[-1] + suffix
    start, end = inpath.index(dataset_prefix), -1
    outpath = '/'.join(inpath[start:end] + [run_name])
    return os.path.join('./output', outpath, fname)


def main(logname=None, run_pass=None, debug=False, topic=None,
    search_tool='a0', max_ctx=10_000, args=None, prompt_mode=None,
    fname_filter=None, skip_existing=True, begin=0, end=None,
    metric='pass@1', ground_truth_dir=None, output_marking=True):

    assert logname is not None

    metric_name, k = metric.split('@')
    k = int(k)
    assert k > 0
    assert metric_name in ['pass', 'maj']

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
        llm_api = partial(chatgpt_agent.complete, engine='gpt-35-turbo')
        llm_rst = chatgpt_agent.reset

        llm_args = []

    elif run_pass == 'td003':
        llm_init = lambda *args: None
        llm_api = partial(chatgpt_agent.complete, engine='text-davinci-003', logprobs=None)
        llm_rst = chatgpt_agent.reset

        llm_args = []

    elif run_pass == 'gpt4':
        llm_init = lambda *args: None
        llm_api = gpt4_complete
        llm_rst = lambda *args: None
        llm_args = []

    elif run_pass == 'ds':
        from tools import test_ds_infer
        llm_init = lambda *_: args
        llm_api = test_ds_infer.test
        llm_rst = lambda *args: None
        llm_args = []

    elif run_pass == 'utils':
        llm_init = utils_llm_init
        llm_api = utils_llm_gen
        llm_rst = lambda *_: None
        llm_args = args

    else:
        raise NotImplemented

    # setup API map
    api_map = {
        'SEARCH': partial(search_wrapper, search_tool),
        'COMPUTE': call_sympy
    }

    reproducible()

    def print_title(title):
        print('\n', '=' * 15, title, '=' * 15, end='\n\n')


    print('Initializing LLM ...', llm_args)
    llm_api_args = llm_init(*llm_args)

    manual_query = None
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

            if 'manual_query' in gt:
                manual_query = gt['manual_query']
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
        llm_api_kargs = {}

        prompt = ''
        pas_okcnt = 0
        while k_count < k:
            llm_rst()

            print_title('Problem')
            print(json_path)

            print_title('Question')
            print(query)

            # determine the prompt
            if prompt_mode == 'direct':
                prompt = direct2(query)

            elif prompt_mode.startswith('example'):
                prompt = example(prompt_mode)

            elif prompt_mode == 'cot':
                prompt = cot2(query)

            elif prompt_mode == 'cot_wizard':
                prompt = cot_wizard(query)

            elif prompt_mode == 'cot_mytrain':
                prompt = cot_mytrain(query)

            elif prompt_mode == 'ia':
                results = api_map['SEARCH'](query, None, gt=gt)
                if len(results) > 0:
                    prompt = ia2(query, *results)
                else:
                    prompt = cot2(query)

            elif prompt_mode == 'mh':
                prompt = multihop1(query)

            elif prompt_mode == 'manual':
                while True:
                    print('current manual query:', manual_query)
                    cmd = input('keyword, "save", "clear", "none", or "skip":\n')
                    if cmd.strip() == '':
                        k_count = 0
                        break
                    elif cmd == 'save':
                        k_count = k
                        break
                    elif cmd == 'clear':
                        manual_query = []

                    elif cmd == 'none':
                        manual_query = None
                        break
                    elif cmd == 'skip':
                        manual_query = None
                        k_count = k
                        break
                    else:
                        if manual_query is None:
                            manual_query = []
                        manual_query.append(cmd)

                metric_name = 'maj' # pass@k would break the loop.
                if k_count == k: continue

                if manual_query is None:
                    prompt = cot_mytrain(query)

                elif len(manual_query) == 0:
                    results = api_map['SEARCH'](query, None)
                    prompt = ia_mytrain(query, [], *results)
                else:
                    results = api_map['SEARCH'](query, manual_query)
                    prompt = ia_mytrain(query, manual_query, *results)

            elif prompt_mode == 'askkey':
                if run_pass == 'td003':
                    prompt = ask_identity_formula(query)
                    #prompt = ask_identity_formula_logits(query)
                else:
                    prompt = ask_identity_formula(query)

                k_count = k
                llm_api_kargs = {'stop': '\n\n', 'stream': False}

            elif prompt_mode == 'manual-picky':
                while True:
                    while True:
                        print('current manual query:', manual_query)
                        cmd = input('formula, "clear", or "skip":\n')
                        if cmd.strip() == '':
                            k_count = k
                            break

                        elif cmd == 'clear':
                            manual_query = []

                        elif cmd == 'skip':
                            manual_query = []
                            k_count = 0
                            break

                        else:
                            if manual_query is None:
                                manual_query = []
                            manual_query.append(cmd)

                    if k_count == 0:
                        results, rates = [], []
                        break

                    if len(manual_query) == 0:
                        results = api_map['SEARCH'](query, None)

                    else:
                        results = api_map['SEARCH'](query, manual_query)

                    prompt = ia_mytrain(query, manual_query, *results)
                    save_log(**{
                        'json_path': json_path,
                        'query': query,
                        'prompt': prompt,
                        'solution': solution,
                        'ground_truth': boxed_solution,
                        'logpath': 'output/MATH/preview.json'
                    })

                    rates = []
                    for res in results:
                        rate = 0
                        print(res, end='\n\n')
                        inp = input('Is the result relevant? [yes/NO/redo]')
                        if inp == 'yes':
                            inp = input('Can you extract the final answer? [yes/NO/redo]')
                            if inp == 'yes':
                                rate = 2
                            elif inp == 'redo':
                                break
                            else:
                                rate = 1
                        elif inp == 'redo':
                            break
                        rates.append(rate)
                    else:
                        # next question
                        break

                    # redo
                    print_title('Problem')
                    print(json_path)

                    print_title('Question')
                    print(query)

                for i, (res, rate) in enumerate(zip(results, rates)):
                    prompt = ia_mytrain(query, manual_query, res)

                    print_title(f'Search Result')
                    print(res)
                    print('manual_rating:', rate)

                    part_logpath = map_query_log_path(json_path,
                        f'run__{logname}', suffix=f'.{search_tool}-{i}.log')
                    save_log(**{
                        'json_path': json_path,
                        'query': query,
                        'prompt': prompt,
                        'solution': solution,
                        'ground_truth': boxed_solution,
                        'manual_rating': rate,
                        'logpath': part_logpath
                    })
                else:
                    json_path = None
                    query = None
                    prompt = None
                    solution = None
                    boxed_solution = None
                    manual_query = []

                break # next question

            else:
                raise NotImplemented

            k_count += 1

            print_title(f'Prompt (len = {len(prompt)})')
            print(prompt)

            # answering
            answer = ''
            save_ans = ''
            while len(prompt) < max_ctx:
                print_title(f'Answer (total prompt len: {len(prompt)})')
                answer = llm_api(prompt,
                    args=llm_api_args, debug=debug, api_map=api_map,
                    abort_criteria=has_any_captured, **llm_api_kargs
                )

                if answer is None: # e.g., content_filter policy triggered
                    continue

                if prompt_mode == 'mh':
                    injected = inject_result(query, answer, api_map)

                    if injected is not None:
                        answer = injected

                    print(answer)
                    save_ans += answer
                    if has_result(answer, api_map):
                        break

                    prompt += answer
                else:
                    print(answer)
                    save_ans = answer
                    break

            #if not isinstance(answer, str):
            #    likelihood = -1
            #    for i, _ in enumerate(answer):
            #        token = answer[i][0]
            #        if (token.strip() in ['yes', 'no'] and
            #            '[' in answer[i-1][0] and
            #            ']' in answer[i+1][0]):
            #            logprob = dict(answer[i][1])[token]
            #            likelihood = math.exp(logprob)
            #            if token.strip() == 'no':
            #                likelihood = 1.0 - likelihood
            #    answer = f'{likelihood:.3f}'
            #    print(answer)

            boxed_answer = extract_math_answer(answer)

            # marking
            if output_marking:
                #print_title('Ground Truth')
                #print(solution)

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
                'answer': save_ans,
                'boxed_answer': boxed_answer,
                'is_equiv': equiv
            })

            # abort upon correct answer in pass@k
            if metric_name == 'pass' and equiv:
                break

        # after the metric@k break, before the next question... 
        if prompt_mode.startswith('example'):
            input('Press Enter to save this example output...')

        if k_count == k:
            save_log(**locals())


def save_log(**kwargs):
        log_json = {
            'problem': kwargs.get('json_path'),
            'query': kwargs.get('query'),
            'prompt': kwargs.get('prompt'),
            'solution': kwargs.get('solution'),
            'ground_truth': kwargs.get('boxed_solution'),
            'judge_buffer': kwargs.get('judge_buffer'),
            'manual_query': kwargs.get('manual_query'),
            'manual_rating': kwargs.get('manual_rating'),
            'args': json.dumps(kwargs.get('args'))
        }
        logpath = kwargs.get('logpath')
        with open(logpath, 'w') as fh:
            json.dump(log_json, fh, indent=2)
            print('Written log:', logpath)
        _output_html(logpath)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(main)
