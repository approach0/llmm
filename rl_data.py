import re
import os
import json
from datasets import Dataset


def data_generator(json_file):
    with open(json_file, 'r') as fh:
        j = json.load(fh)

    for item in j:
        if item['problem'] is None:
            continue
        yield item


def collate_prompts(batch_tok_fn, batch_data):
    prompts = [d['prompt'] for d in batch_data]
    return batch_tok_fn(prompts), batch_data


def collate_ask_relevance(batch_tok_fn, batch_data):
    from tools.prompt_factory import ask_relevance
    prompts = [
        ask_relevance(
            d['prompt']
            .split('### Input:\n')[1]
            .replace('### Response:\n', '')
            + '\n\n' + '### Response:'
        )
        for d in batch_data
    ]
    return batch_tok_fn(prompts), batch_data


def reward_ask_relevance(config, inp, out):
    rewards = []
    for raw, inp_str, out_str in zip(
        inp[1], inp[0]['texts'], out):

        log = raw.copy()
        log['judge_buffer'] = [{
            'answer': out_str,
            'boxed_answer': log['manual_rating'],
            'is_equiv': None
        }]
        rewards.append(log)
    return rewards


def step_ask_relevance(cfg, inp, out, trainer, rewards):
    run_name = cfg.name
    output_dir = os.path.join(cfg.get('output_dir'), run_name)
    os.makedirs(output_dir, exist_ok=True)
    for log in rewards:
        log_name = os.path.basename(log['problem']) + '.log'
        log['logpath'] = os.path.join(
            output_dir, log_name,
        )
        save_log(**log)


def save_log(**kwargs):
    from tools.inspect_output import _output_html
    log_json = {
        'problem': kwargs.get('problem'),
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
    json_file = './output/merged_test.json'
    ds = Dataset.from_generator(data_generator,
        gen_kwargs={'json_file': json_file})
    print(ds[0])
    ds.push_to_hub("approach0/MATH-picky-test")
