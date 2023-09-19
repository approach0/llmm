import os
import re
import json
from collections import defaultdict

import sys
sys.path.insert(0, '../Progressive-Hint')
sys.path.insert(0, '../math/modeling')
from main_clean import extract_math_answer
from math_equivalence import is_equiv


def get_stats_v1(j, detail, logpath):
    agent_answer = j['agent_answer']
    ground_truth = j['ground_truth']
    is_equiv = j['is_equiv']
    if detail > 1: print(logpath, '\t', agent_answer, '\t', ground_truth, '\t', is_equiv)
    elif detail > 0: print(logpath, is_equiv)
    return 1 if is_equiv else 0


def get_stats_v2(j, detail, logpath, metric):
    judge_buffer = j['judge_buffer']
    ground_truth = j['ground_truth'] if 'ground_truth' in j else None

    if ('manual_query' in j and
        isinstance(j['manual_query'], list) and
        len(j["manual_query"]) > 0):
        gt = 1
    else:
        gt = 0

    vote_dict = defaultdict(int)
    for judge in judge_buffer:
        boxed_answer = judge['boxed_answer'].strip()
        vote_dict[boxed_answer] += 1
    votes = vote_dict.items()
    if ground_truth is None:
        okays = [jj['is_equiv'] for jj in judge_buffer]
    else:
        okays = [is_equiv(jj['boxed_answer'], ground_truth) for jj in judge_buffer]

    if '@' in metric:
        metric, k = metric.split('@')
        okays = okays[:int(k)]

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


def compare_differences(logdir1, logdir2, metric='pass'):
    win1, win2 = 0, 0
    for fname in os.listdir(logdir1):
        logpath1 = os.path.join(logdir1, fname)
        logpath2 = os.path.join(logdir2, fname)
        if not os.path.exists(logpath1) or logpath1.split('.')[-1] != 'log':
            continue
        if not os.path.exists(logpath2) or logpath2.split('.')[-1] != 'log':
            continue
        with open(logpath1, 'r') as fh:
            j1 = json.load(fh)
            assert not 'agent_answer' in j1
        with open(logpath2, 'r') as fh:
            j2 = json.load(fh)
            assert not 'agent_answer' in j2
        good1, _ = get_stats_v2(j1, 0, None, metric)
        good2, _ = get_stats_v2(j2, 0, None, metric)
        if good1 != good2:
            if good1:
                win1 += 1
            else:
                win2 += 1
            print(fname, good1, good2)
    def name(path):
        return os.path.basename(os.path.normpath(path))
    return name(logdir1), win1, name(logdir2), win2


def find_correct_samples(logdir):
    for fname in os.listdir(logdir):
        logpath = os.path.join(logdir, fname)
        if logpath.split('.')[-1] != 'log':
            continue
        with open(logpath, 'r') as fh:
            j = json.load(fh)
        judge_buffer = j['judge_buffer']
        for judge in judge_buffer:
            if not judge['is_equiv']: continue
            #print(j['problem'])
            #print(j['query'])
            #print(judge['answer'])
            #print(judge['boxed_answer'])
            _output_html(logpath)
            #input('Enter to continue...')


def textify_v1(items):
    return [f'<b>{key}</b>: {val}' for key, val in j.items()]


def textify_v2(j_dict):
    order = ['problem', 'query', 'ground_truth', 'prompt',
        'manual_rating', 'judge_buffer', 'solution']
    text_list = []
    for key in order:
        if key not in j_dict: continue
        val = j_dict[key]
        if val is None: continue
        # for each log item ...
        if key == 'judge_buffer':
            if len(val) == 0: continue
            for i, j in enumerate(val + [val[0]]):
                answer = j['answer']
                boxed_answer = j['boxed_answer']
                correct = j['is_equiv']
                if correct or i >= len(val):
                    text_list.append(f'<h3>answer</h3>{answer}<hr>')
                    text_list.append(f'<h3>boxed_answer</h3>{boxed_answer}<hr>')
                    text_list.append(f'<h3>correct</h3>{correct}<hr>')
                    break
        else:
            text_list.append(f'<h3>{key}</h3>{val}<hr>')
    return text_list


def textify_flip(j_dict):
    order = ['direct_prompt', 'direct_answer', 
        'agument_prompt', 'agument_answer', 'solution']
    text_list = []
    for key in order:
        if key not in j_dict: continue
        val = j_dict[key]
        if val is None: continue
        text_list.append(f'<h3>{key}</h3>{val}<hr>')
    return text_list


def _output_html(logpath, verbose=True, query_key='query'):
    import sys
    sys.path.insert(0, './pya0')
    from pya0.visualize import output_html as output
    with open(logpath, 'r') as fh:
        j = json.load(fh)
        if 'agent_answer' in j:
            results = textify_v1(j.items())
        elif 'direct_answer' in j:
            results = textify_flip(j)
        else:
            results = textify_v2(j)

        def html(fh, query, hit, page, idx):
            hit = re.sub("#+ (.+)\n", r"<h4>\1</h4>", hit)
            hit = re.sub("URL: (.+)\n+", r"<h4>\1</h4>", hit)
            if verbose: print(hit)
            hit = hit.replace('\n', '<br/>\n')
            fh.write(f'<p>{hit}</p>\n\n')

        logdir = os.path.dirname(logpath)
        logbase = os.path.basename(logpath)
        head = j[query_key] if query_key in j else j['question']
        output(logdir, logbase, '_', head, results,
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


def get_auto_judge_stats(logdir, suffix='.log',
    partial_relevance=False):
    correct_cnt, invalid_cnt, total_cnt = 0, 0, 0
    auto_scores, man_scores = [], []
    overlap_scores = []
    for fname in os.listdir(logdir):
        logpath = os.path.join(logdir, fname)
        if not logpath.endswith(suffix):
            continue
        with open(logpath, 'r') as fh:
            j = json.load(fh)
        judge_buffer = j['judge_buffer']
        jj = judge_buffer[0]
        answer = jj['answer']
        m_rate = j['manual_rating']
        man_scores.append(m_rate)

        if m := re.search(r'rate\[(\d+)\]', answer):
            rate = int(m.group(1))
            auto_scores.append(rate)

            if partial_relevance:
                correct = (bool(m_rate) == bool(rate))
            else:
                correct = (m_rate == rate)
            if correct:
                correct_cnt += 1
                overlap_scores.append(rate)
        else:
            #print('wrong format:', answer)
            invalid_cnt += 1
        total_cnt += 1
    accuracy = correct_cnt / total_cnt * 100
    print(f'invalid coutn: {invalid_cnt}')
    print(f'{correct_cnt} / {total_cnt} = {accuracy:.2f}%')

    save_hist('manu_judge_stats', man_scores, verbose=False, th=1)
    save_hist('auto_judge_stats', auto_scores, verbose=False, th=1)
    save_hist('overlap_judge_stats', overlap_scores, verbose=False, th=1)


def get_class_hist(logdir, suffix, mode='gpt3.5', verbose=False):
    assert mode in ['gpt3.5', 'td003']
    data = dict()
    logdir = os.path.normpath(logdir)
    names = logdir.split('/')[-2:]
    for fname in os.listdir(logdir):
        logpath = os.path.join(logdir, fname)
        if os.path.isdir(logpath):
            res = get_class_hist(logpath, suffix, mode=mode)
            data.update(res)
        if logpath.split('.')[-1] != 'log':
            continue
        elif not names[-1].endswith(suffix):
            continue
        with open(logpath, 'r') as fh:
            j = json.load(fh)
        answer = j['judge_buffer'][0]['answer']
        if mode == 'gpt3.5':
            m = re.search(r'prob\[(\d+)\]', answer)
            if m:
                confidence = int(m.group(1))
                data[logpath] = confidence
            else:
                #print('wrong format:', answer)
                pass
        else:
            try:
                data[logpath] = float(answer)
            except ValueError:
                #print('wrong format:', answer)
                pass

    if names[0] in ['MATH', 'test']:
        if names[0] == 'MATH': names[1] = 'overall'
        name = suffix + '__' + names[1]
        if verbose: print(name, len(data))
        save_hist(name, data.values(), verbose=verbose)

    return data


def save_hist(name, x, verbose=False, th=5):
    import matplotlib.pyplot as plt
    import numpy as np
    if verbose: print('Saving:', name)

    fig, axis = plt.subplots(1, 2)

    axis[0].hist(x)
    axis[0].set_title(name)

    x = list(map(lambda x: 1 if x >= th else 0, x))
    axis[1].hist(x)
    axis[1].set_title('yes=1, no=0')

    plt.savefig(name + '.png')


def get_json_val(logpath, key='prompt'):
    with open(logpath, 'r') as fh:
        j = json.load(fh)
    print(j[key])


def test_lr_query(logdir):
    import matplotlib.pyplot as plt
    import numpy as np
    stats = {}
    all_keys = [
        'ppo/std_scores',
        'ppo/mean_scores',
        'ppo/mean_non_score_reward',
        'ppo/loss/total',
    ]
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    logdir = os.path.normpath(logdir)
    run_name = os.path.basename(logdir)
    for fname in os.listdir(logdir):
        logpath = os.path.join(logdir, fname)
        if not logpath.endswith('.log'):
            continue
        print(logpath)
        m = re.search(r'step([0-9]+)_', logpath)
        step = int(m.group(1))
        with open(logpath, 'r') as fh:
            j = json.load(fh)
            #print(j.keys()); quit()
            stats_item = {}
            for key in j.keys():
                if key in all_keys:
                    stats_item[key] = j[key]
            stats[step] = stats_item
    fig, axes = plt.subplots(len(all_keys), 1)
    xpoints = np.array(sorted(stats.keys()))
    for i, key in enumerate(all_keys):
        ypoints = np.array([stats[x][key] for x in xpoints])
        avg_ypoints = np.convolve(ypoints, kernel, mode='same')
        fig.axes[i].plot(
            xpoints[kernel_size:-kernel_size],
            avg_ypoints[kernel_size:-kernel_size]
        )
        fig.axes[i].set_title(key)
    fig.tight_layout()
    plt.savefig(f'output-{run_name}.png')


def traverse_tree(tree, path, answers):
    node_type = tree['node_type']
    state = tree['state']
    prompt = tree['prompt']
    children = tree['children']

    path_types = [n['node_type'] for n in path]
    path_types = ''.join(path_types)

    if len(children) > 0:
        for child in children:
            path.append(tree)
            traverse_tree(child, path, answers)
            path.pop()
    elif path_types in ['Q', 'QKR']:
            answers[path_types].append((state, prompt))


def get_flips_in_trees(logdir):
    n_flips, n_total = 0, 0
    for fname in os.listdir(logdir):
        n_total += 1
        logpath = os.path.join(logdir, fname)
        if logpath.split('.')[-1] != 'log':
            continue
        with open(logpath, 'r') as fh:
            j = json.load(fh)
        path = j['path']
        sol = j['solution']
        tree = j['json']
        answers = defaultdict(list)
        traverse_tree(tree, [], answers)

        def mark(x):
            ans, prompt = x
            boxed_ans = extract_math_answer(ans)
            boxed_sol = extract_math_answer(sol)
            equiv = is_equiv(boxed_sol, boxed_ans)
            return equiv

        Q_mark = list(map(mark, answers['Q']))
        R_mark = list(map(mark, answers['QKR']))
        ratio = len(R_mark) // len(Q_mark)
        Q_x = []
        Q_mark_expand = []
        for mark, x in zip(Q_mark, answers['Q']):
            Q_x += [x] * ratio
            Q_mark_expand += [mark] * ratio

        n_sub_flips = 0
        for q_mark, r_mark, q_x, r_x in zip(
            Q_mark_expand, R_mark, Q_x, answers['QKR']):
            if not q_mark and r_mark:
                n_sub_flips += 1
                q_ans, q_prompt = q_x
                r_ans, r_prompt = r_x

                j = {}
                j['direct_prompt'] = q_prompt
                j['direct_answer'] = q_ans
                j['agument_prompt'] = r_prompt
                j['agument_answer'] = r_ans
                j['solution'] = sol
                j['path'] = path
                with open(logpath + f'.{n_sub_flips}.flip', 'w') as fh:
                    json.dump(j, fh)

        if n_sub_flips > 1:
            n_flips += 1
        print(path, n_sub_flips)

    return n_flips, n_total, n_flips / n_total


def output_flips(logdir):
    for fname in os.listdir(logdir):
        logpath = os.path.join(logdir, fname)
        if logpath.split('.')[-1] != 'flip':
            continue
        print(logpath)
        _output_html(logpath, query_key='path')



if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'get_stats': get_topic_stats,
        'diff': compare_differences,
        'find': find_correct_samples,
        'get': get_json_val,
        'output_html': _output_html,
        'output_htmls': output_html,
        'output_flips': output_flips,
        'get_class_hist': get_class_hist,
        'auto_judge_stats': get_auto_judge_stats,
        'test_np': test_np,
        'test_lr_query': test_lr_query,
        'get_flips': get_flips_in_trees,
    })
