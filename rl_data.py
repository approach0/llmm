import re
import os
import sys
import copy
import json
from datasets import Dataset, concatenate_datasets
from datasets.dataset_dict import DatasetDict

IGNORE_INDEX = -100


###############
# utilities
###############
def data_generator(json_file):
    with open(json_file, 'r') as fh:
        j = json.load(fh)

    for item in j:
        if 'problem' in item and item['problem'] is None:
            continue
        yield item


def decode_pr(batch_tok_fn, ids):
    ids = ids.clone()
    ids[ids == -100] = 0
    text = batch_tok_fn(ids, decode=True)
    print(text, len(ids))


def collate_pr(config, batch_tok_fn, sources, targets):
    eos = config.getboolean('collate_add_eos', True)
    debug = config.getboolean('collate_debug', False)

    examples = [s + t for s, t in zip(sources, targets)]
    sources_tokenized = batch_tok_fn(sources, eos=False)
    examples_tokenized = batch_tok_fn(examples, eos=eos)
    labels = examples_tokenized["input_ids"].clone()

    for label, src_len, exm_len in zip(labels,
        sources_tokenized["attention_mask"].sum(-1).tolist(),
        examples_tokenized["attention_mask"].sum(-1).tolist()):
        label[:src_len] = IGNORE_INDEX
        label[exm_len:] = IGNORE_INDEX

    examples_tokenized.update({
        'labels': labels
    })

    if debug:
        decode_pr(batch_tok_fn, examples_tokenized['input_ids'][0])
        decode_pr(batch_tok_fn, examples_tokenized['labels'][0])

    return examples_tokenized


###############
# datamap
###############
def datamap_good_rating(config, dataset):
    picky_dataset1 = dataset['train'].filter(lambda x: x['manual_rating'] > 1)
    picky_dataset2 = dataset['test'].filter(lambda x: x['manual_rating'] > 1)

    dataset = DatasetDict({
        'train': concatenate_datasets([picky_dataset1, picky_dataset2])
    })
    return dataset


###############
# collate func
###############
def collate_prompt(config, batch_tok_fn, batch_data):
    prompts = [d['prompt'] for d in batch_data]
    return batch_tok_fn(prompts), batch_data


def collate_finetune_phase1(config, batch_tok_fn, batch_data):
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
    return collate_pr(config, batch_tok_fn, sources, targets)


def collate_finetune_phase2(config, batch_tok_fn, batch_data):
    def rate2outputs(rate):
        if rate == 0:
            return 'This search result is not helpful.'
        elif rate == 1:
            return 'This search result might be useful.'
        else:
            return 'This search result looks very relevant!'

    sources = [d['prompt'] + '\n' for d in batch_data]
    targets = [rate2outputs(d['manual_rating']) for d in batch_data]
    return collate_pr(config, batch_tok_fn, sources, targets)


def collate_ask_relevance(config, batch_tok_fn, batch_data):
    from tools.prompt_factory import ask_relevance
    for data in batch_data:
        data['prompt'] = ask_relevance(
            data['prompt']
            .split('### Input:\n')[1]
            .replace('### Response:\n', '')
            + '\n\n' + '### Response:'
        )
    prompts = [d['prompt'] for d in batch_data]
    return batch_tok_fn(prompts), batch_data


def collate_phase2_learn_query(config, batch_tok_fn, batch_data):
    from tools.prompt_factory import tool_prompt1
    for data in batch_data:
        query = data['query']
        example = data['prompt']
        prompt = tool_prompt1(query)
        response_sect = '### Response:\n'
        response = example.split(response_sect)[1]
        srch_query = response.split('\n\n')[0]
        new_prompt = prompt + '\n\n' + response_sect
        data['prompt'] = new_prompt
        data['output'] = srch_query

    sources = [d['prompt'] + '\n' for d in batch_data]
    targets = [d['output'] + '\n' for d in batch_data]
    return collate_pr(config, batch_tok_fn, sources, targets)


###############
# reward func
###############
def reward_by_answer(config, inp, out, model):
    from main_clean import extract_math_answer
    from math_equivalence import is_equiv
    from rich import print as rich_print

    rewards = []
    for raw, inp_str, out_str in zip(
        inp[1], inp[0]['texts'], out):

        ground_truth = extract_math_answer(raw['solution'])
        out_boxed = extract_math_answer(out_str)
        equiv = is_equiv(ground_truth, out_boxed)

        if 'judge_buffer' not in raw or raw['judge_buffer'] is None:
            raw['judge_buffer'] = []
        raw['judge_buffer'].append({
            'answer': out_str,
            'boxed_answer': out_boxed,
            'is_equiv': equiv
        })
        rewards.append(1. if equiv else 0.)

        rich_print('[blue]ground truth:[/blue]', ground_truth)
        if equiv:
            rich_print('[green]correct[/green]')
        else:
            rich_print('[red]wrong[/red]', out_boxed)

    return rewards


###############
# log func
###############
def default_log(config, values):
    verbose = config.getboolean('log_verbose', False)
    run_name = config.name
    output_dir = os.path.join(config.get('output_dir'), run_name)
    os.makedirs(output_dir, exist_ok=True)
    step = values['step']
    for b, (inp, out) in enumerate(
        zip(values['batch_in'][1], values['batch_out'])
    ):
        log = copy.deepcopy(inp)
        log_name = log['problem'].strip('.').replace('/', '_')
        log_name = log_name + f'.step{step}_batch{b}.log'
        logpath = os.path.join(output_dir, log_name)
        save_log(logpath, verbose=verbose, **log)


def save_log(logpath, verbose=False, **kwargs):
    from tools.inspect_output import _output_html
    with open(logpath, 'w') as fh:
        json.dump(kwargs, fh, indent=2)
    _output_html(logpath, verbose=False)
    if verbose: print(kwargs.keys())
    if verbose: print('Written log:', logpath)


if __name__ == '__main__':
    json_file = './output/finetune-pairs.json'
    ds_all = Dataset.from_generator(data_generator,
        gen_kwargs={'json_file': json_file})
    ds_train = ds_all.filter(lambda x: 'train' in x['problem'])
    ds_test = ds_all.filter(lambda x: 'test' in x['problem'])

    dataset = DatasetDict({'train': ds_train, 'test': ds_test})
    dataset.push_to_hub("approach0/mathy-phase2")
