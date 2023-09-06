import json
import string
from colorama import Fore, Style
from functools import partial


class ToolError():
    def __init__(self, errstr):
        self.errstr = errstr
    def __repr__(self):
        return self.errstr


def sympy_solver(args):
    from tools.test_sympy import compute as sympy_compute
    print('computing API:', args)
    if not isinstance(args, list):
        return ToolError('Passed argument format error. Use array!')
    elif len(args) < 2:
        return ToolError('Error: at least two arguments needed!')
    else:
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


def search_mux(api_name, question, keywords, docid=None):
    if keywords:
        keywords = list(map(smart_correct, keywords))

    print(Fore.CYAN)
    print('search API:', api_name)
    print('docID:', docid)
    print('query question:', question)
    print('query keywords:', keywords)
    print(Style.RESET_ALL)

    if api_name == 'mabowdor':
        from tools.test_mabowdor import search
        return search('mabowdor', question, keywords)

    elif api_name == 'a0':
        from tools.test_mabowdor import search
        return search('mabowdor', None, keywords)

    elif api_name == 'MATH':
        from tools.test_mabowdor import search
        return search('MATH', question, None)

    elif api_name == 'dups':
        from tools.test_mabowdor import search
        return search('dups', None, keywords, docid, no_mapper=True)

    elif api_name == 'online':
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


def has_call(answer, tool_map):
    if has_result(answer, tool_map):
        return True
    for key in tool_map.keys():
        if key in answer:
            idx = answer.find(key)
            begin, end = capture(answer[idx:], ('[', ']'))
            if begin < end:
                return True
    return False


def has_result(answer, tool_map):
    if r'<|im_end|>' in answer:
        return True
    key = r'\boxed'
    if key in answer:
        idx = answer.find(key)
        begin, end = capture(answer[idx:], ('{', '}'))
        if begin < end:
            return True
    return False


def has_any_captured(answer, tool_map):
    return has_result(answer, tool_map) or has_call(answer, tool_map)


def tool_invoke(response, tool_map):
    for tool_name in tool_map:
        idx = response.find(tool_name)
        if idx == -1: continue
        idx += len(tool_name)
        begin, end = capture(response[idx:], ('[', ']'))
        begin, end = begin + idx, end + idx
        if begin >= end:
            pre_invoke = response[:idx].strip()
            return pre_invoke, ToolError(
                'Wrong JSON array format!\n' +
                'Forget to add square brackets?')

        pre_invoke = response[:end+1].strip()
        tool_args = response[begin:end+1]
        try:
            tool_args = json.loads(tool_args)
            tool_result = tool_map[tool_name](tool_args)
            return pre_invoke, tool_result

        except json.decoder.JSONDecodeError:
            return pre_invoke, ToolError(
                    'JSON array decode error!\n' +
                    'Check your calling format!')

    return response, ToolError('No tool being invoked!')


if __name__ == '__main__':
    tool_map = {
        'COMPUTE': sympy_solver,
        'SEARCH': partial(search_mux, 'dups', None, docid=10490)
    }

    response = r'''I can invoke the search API here...

    SEARCH["1^\\infty"]

    should it return some results here?'''

    if has_any_captured(response, tool_map):
        pre_invoke, tool_res = tool_invoke(response, tool_map)
        print(pre_invoke)
        print(tool_res)
        print(isinstance(tool_res, ToolError))
    else:
        print('nothing to capture!')

    response = r'''I can invoke the search API here...

    COMPUTE["solve x", "x^2 = 4"]

    should it return some results here?'''

    if has_any_captured(response, tool_map):
        pre_invoke, tool_res = tool_invoke(response, tool_map)
        print(pre_invoke)
        print(tool_res)
        print(isinstance(tool_res, ToolError))
    else:
        print('nothing to capture!')
