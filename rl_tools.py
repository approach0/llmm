import json
import string
from colorama import Fore, Style

from test_sympy import compute as sympy_compute


def calculator(query, args):
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


def search(tool, question, keywords, gt=None, **kargs):
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


def has_call(answer, api_map):
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
    return has_result(answer, api_map) or has_call(answer, api_map)


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
