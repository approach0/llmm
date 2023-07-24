import os
import json


def get_topic_stats(logdir):
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
        agent_answer = j['agent_answer']
        ground_truth = j['ground_truth']
        is_equiv = j['is_equiv']
        #print(logpath, '\t', agent_answer, '\t', ground_truth, '\t', is_equiv)
        #print(logpath, is_equiv)
        if is_equiv:
            correct_cnt += 1
        total_cnt += 1

    if total_cnt == 0:
        total_cnt = -1
    accuracy_percentage = correct_cnt / total_cnt * 100
    print(f'{logdir}: {correct_cnt} / {total_cnt} = {accuracy_percentage:.2f}%')
    return correct_cnt, total_cnt


def _output_html(logpath):
    import sys
    sys.path.insert(0, './pya0')
    from pya0.visualize import output_html as output
    with open(logpath, 'r') as fh:
        j = json.load(fh)
        results = [f'<b>{key}</b>: {val}' for key, val in j.items()]

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
