import os
import json
from collections import defaultdict


def is_correct_v1(j, detail, logpath):
    agent_answer = j['agent_answer']
    ground_truth = j['ground_truth']
    is_equiv = j['is_equiv']
    if detail > 1: print(logpath, '\t', agent_answer, '\t', ground_truth, '\t', is_equiv)
    elif detail > 0: print(logpath, is_equiv)
    return is_equiv


def is_correct_v2(j, detail, logpath, metric):
    import sys
    sys.path.insert(0, '../math/modeling')
    from math_equivalence import is_equiv

    judge_buffer = j['judge_buffer']
    ground_truth = j['ground_truth']

    vote_dict = defaultdict(int)
    for judge in judge_buffer:
        boxed_answer = judge['boxed_answer'].strip()
        vote_dict[boxed_answer] += 1
    votes = vote_dict.items()
    max_answer, max_votes = max(votes, key=lambda x: x[1])

    if metric == 'pass':
        good = any([is_equiv(ans, ground_truth) for ans in vote_dict.keys()])
    elif metric == 'maj':
        good = is_equiv(max_answer, ground_truth)
    else:
        raise NotImplemented

    if detail > 1: print(logpath, '\t', votes, '\t', ground_truth, '\t', good)
    elif detail > 0: print(logpath, good)
    return good


def get_topic_stats(logdir, detail=0, metric='pass'):
    correct_cnt, total_cnt = 0, 0
    for fname in os.listdir(logdir):
        logpath = os.path.join(logdir, fname)
        if os.path.isdir(logpath):
            c, t = get_topic_stats(logpath)
            correct_cnt += c
            total_cnt += t
        if logpath.split('.')[-1] != 'log':
            continue
        with open(logpath, 'r') as fh:
            j = json.load(fh)
        if 'agent_answer' in j:
            if is_correct_v1(j, detail, logpath):
                correct_cnt += 1
        else:
            if is_correct_v2(j, detail, logpath, metric):
                correct_cnt += 1
        total_cnt += 1

    if total_cnt == 0:
        total_cnt = -1
    accuracy_percentage = correct_cnt / total_cnt * 100
    print(f'{logdir}: {correct_cnt} / {total_cnt} = {accuracy_percentage:.2f}%')
    return correct_cnt, total_cnt


def textify_v1(items):
    return [f'<b>{key}</b>: {val}' for key, val in j.items()]


def textify_v2(items):
    text_list = []
    for key, val in items:
        # for each log item ...
        if key == 'judge_buffer':
            if len(val) == 0: continue
            for i, j in enumerate(val + [val[0]]):
                answer = j['answer']
                boxed_answer = j['boxed_answer']
                if j['is_equiv'] or i >= len(val):
                    text_list.append(f'<b>Answer</b>: {answer}')
                    text_list.append(f'<b>Boxed answer</b>: {boxed_answer}')
                    break
        elif key in ['args', 'manual_query']:
            continue
        else:
            text_list.append(f'<b>{key}</b>: {val}')
    return text_list


def _output_html(logpath):
    import sys
    sys.path.insert(0, './pya0')
    from pya0.visualize import output_html as output
    with open(logpath, 'r') as fh:
        j = json.load(fh)
        if 'agent_answer' in j:
            results = textify_v1(j.items())
        else:
            results = textify_v2(j.items())

        def html(fh, query, hit, page, idx):
            hit = hit.replace('\n', '<br/>')
            fh.write(f'<p>{hit}</p>\n\n')

        logdir = os.path.dirname(logpath)
        logbase = os.path.basename(logpath)
        output(logdir, logbase, '_', j['problem'], results,
            None, False, 100, html, create_parent_dir=False)


def output_html(logdir):
    for fname in os.listdir(logdir):
        logpath = os.path.join(logdir, fname)
        if os.path.isdir(logpath):
            output_html(logpath)
        if logpath.split('.')[-1] != 'log':
            continue
        print(logpath)
        _output_html(logpath)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'get_stats': get_topic_stats,
        'output_html': _output_html,
        'output_htmls': output_html
    })
