import re
import os
import copy
import json
from datasets import Dataset

IGNORE_INDEX = -100


def data_generator(json_file):
    with open(json_file, 'r') as fh:
        j = json.load(fh)

    for item in j:
        if 'problem' in item and item['problem'] is None:
            continue
        yield item


def collate_prompts(batch_tok_fn, batch_data):
    prompts = [d['prompt'] for d in batch_data]
    return batch_tok_fn(prompts), batch_data


def collate_finetune_phase1(batch_tok_fn, batch_data):
    template = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        )
    sources = [
        template.format_map(dict(instruction=d['instruction'], input=d['input']))
        for d in batch_data
    ]
    targets = [d['output'] for d in batch_data]
    examples = [s + t for s, t in zip(sources, targets)]

    sources_tokenized = batch_tok_fn(sources)
    targets_tokenized = batch_tok_fn(targets)
    examples_tokenized = batch_tok_fn(examples)
    labels = examples_tokenized["input_ids"].clone()

    for label, src_len, exm_len in zip(labels,
        sources_tokenized["attention_mask"].sum(-1).tolist(),
        examples_tokenized["attention_mask"].sum(-1).tolist()):
        label[:src_len] = IGNORE_INDEX
        label[exm_len:] = IGNORE_INDEX

    examples_tokenized.update({
        'labels': labels
    })

    return examples_tokenized


def collate_prompts_finetune(batch_tok_fn, batch_data):
    def rate2outputs(rate):
        if rate == 0:
            return 'This search result is not helpful.'
        elif rate == 1:
            return 'This search result might be useful.'
        else:
            return 'This search result looks very relevant!'

    sources = [d['prompt'] + '\n' for d in batch_data]
    targets = [rate2outputs(d['manual_rating']) for d in batch_data]
    examples = [s + t for s, t in zip(sources, targets)]

    sources_tokenized = batch_tok_fn(sources)
    targets_tokenized = batch_tok_fn(targets)
    examples_tokenized = batch_tok_fn(examples)
    labels = examples_tokenized["input_ids"].clone()

    for label, src_len, exm_len in zip(labels,
        sources_tokenized["attention_mask"].sum(-1).tolist(),
        examples_tokenized["attention_mask"].sum(-1).tolist()):
        label[:src_len] = IGNORE_INDEX
        label[exm_len:] = IGNORE_INDEX

    examples_tokenized.update({
        'labels': labels
    })

    return examples_tokenized


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


def reward_ask_relevance(config, inp, out, model):
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


def step_ask_relevance(cfg, step, trainer, rewards):
    run_name = cfg.name
    output_dir = os.path.join(cfg.get('output_dir'), run_name)
    os.makedirs(output_dir, exist_ok=True)
    for b, log in enumerate(rewards):
        log_name = log['problem'].strip('.').replace('/', '_')
        log_name = log_name + f'.{step}_{b}.log'
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
    json_file = './data/finetune-pairs.json'
    ds = Dataset.from_generator(data_generator,
        gen_kwargs={'json_file': json_file})
    print(ds[0])
    ds.push_to_hub("approach0/mathy-phase1")
