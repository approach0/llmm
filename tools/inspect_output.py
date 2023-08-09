import os
import re
import json
from collections import defaultdict


def get_stats_v1(j, detail, logpath):
    agent_answer = j['agent_answer']
    ground_truth = j['ground_truth']
    is_equiv = j['is_equiv']
    if detail > 1: print(logpath, '\t', agent_answer, '\t', ground_truth, '\t', is_equiv)
    elif detail > 0: print(logpath, is_equiv)
    return 1 if is_equiv else 0


def get_stats_v2(j, detail, logpath, metric):
    import sys
    sys.path.insert(0, '../math/modeling')
    from math_equivalence import is_equiv

    judge_buffer = j['judge_buffer']
    ground_truth = j['ground_truth']

    gt = 1 if ("manual_query" in j and len(j["manual_query"]) > 0) else 0

    vote_dict = defaultdict(int)
    for judge in judge_buffer:
        boxed_answer = judge['boxed_answer'].strip()
        vote_dict[boxed_answer] += 1
    votes = vote_dict.items()
    okays = [is_equiv(j['boxed_answer'], ground_truth) for j in judge_buffer]

    if metric == 'pass':
        good = any(okays)
    elif metric == 'all':
        good = all(okays)
    elif metric == 'maj':
        max_answer, max_votes = max(votes, key=lambda x: x[1])
        tie_votes = filter(lambda x: x[1] == max_votes, votes)
        good = any([is_equiv(x[0], ground_truth) for x in tie_votes])
    else:
        raise NotImplemented

    if detail > 2: print(logpath, '\t', votes, '\t', ground_truth, '\t', good)
    elif detail > 1: print(logpath, '\t', f'{gt}\t{sum(okays)}/{len(okays)}', '\t', good)
    elif detail > 0: print(logpath, good)

    cnt = 1 if good else 0
    return cnt, gt


def get_topic_stats(logdir, detail=0, metric='pass'):
    correct_cnt, total_cnt, gt_cnt = 0, 0, 0
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
            cnt = get_stats_v1(j, detail, logpath)
            correct_cnt += cnt
        else:
            cnt, gt = get_stats_v2(j, detail, logpath, metric)
            correct_cnt += cnt
            gt_cnt += gt
        total_cnt += 1

    if total_cnt == 0:
        total_cnt = -1
    if gt_cnt > 0:
        label_percentage = gt_cnt / total_cnt * 100
        print(f'Total manual labels: {gt_cnt} / {total_cnt} = {label_percentage:.2f}%')
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
                correct = j['is_equiv']
                if correct or i >= len(val):
                    text_list.append(f'<b>Answer</b>: {answer}')
                    text_list.append(f'<b>Boxed answer</b>: {boxed_answer}')
                    text_list.append(f'<b>Correct</b>: {correct}')
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


# https://github.com/lz1oceani/verify_cot/raw/main/results/chatgpt3.5/natural_program/MATH_np.json
def test_np(fpath='data/MATH_np.json'):
    with open(fpath, 'r') as fh:
        j = json.load(fh)
    print(j[0])
    d = defaultdict(int)
    cnt = 0
    for item in j:
        topic = item['type']
        okays = item['per_sample_correct']
        d[topic] += 1
        if topic == 'Precalculus':
            if any(okays):
                cnt += 1
    print(cnt, d)


def get_class_hist(logdir, mode):
    assert mode in ['gpt3.5', 'td003']
    data = []
    for fname in os.listdir(logdir):
        logpath = os.path.join(logdir, fname)
        if logpath.split('.')[-1] != 'log':
            continue
        with open(logpath, 'r') as fh:
            j = json.load(fh)
        answer = j['judge_buffer'][0]['answer']
        if mode == 'gpt3.5':
            m = re.search(r'prob\[(\d+)\]', answer)
            if m:
                confidence = int(m.group(1))
                data.append(confidence)
            else:
                print('wrong format:', answer)
        else:
            try:
                data.append(float(answer))
            except ValueError:
                print('wrong format:', answer)
    logdir = os.path.normpath(logdir)
    basename = os.path.basename(logdir)
    save_hist(f'{basename}.png', data)


def save_hist(path, x):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.hist(x)
    #plt.show()
    print('Saving:', path)
    plt.savefig(path)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'get_stats': get_topic_stats,
        'output_html': _output_html,
        'output_htmls': output_html,
        'get_class_hist': get_class_hist,
        'test_np': test_np,
    })
