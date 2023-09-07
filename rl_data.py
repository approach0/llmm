import re
import os
import sys
import copy
import json
import torch
from functools import partial
from datasets import Dataset, concatenate_datasets
from datasets.dataset_dict import DatasetDict

IGNORE_INDEX = -100


###############
# utilities
###############
def data_generator_json(json_file):
    with open(json_file, 'r') as fh:
        j = json.load(fh)

    for item in j:
        if 'problem' in item and item['problem'] is None:
            continue
        yield item


def data_generator_jsonl(jsonl_file):
    with open(jsonl_file, 'r') as fh:
        for line in fh:
            item = json.loads(line)
            yield item


def decode_show(batch_tok_fn, ids):
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
        decode_show(batch_tok_fn, examples_tokenized['input_ids'][0])
        decode_show(batch_tok_fn, examples_tokenized['labels'][0])

    return examples_tokenized


def limit_length(limit, string):
    if len(string) > limit:
        return string[:limit] + '...'
    else:
        return string


###############
# datamap
###############
def datamap_merge_train_and_test(config, dataset):
    dataset1 = dataset['train']
    dataset2 = dataset['test']
    dataset = DatasetDict({
        'train': concatenate_datasets([dataset1, dataset2])
    })
    return dataset


def datamap_debug(config, dataset):
    dataset['train'] = dataset['train'].filter(lambda x: x['qid'] == '101157')
    return dataset


def datamap_good_rating(config, dataset):
    picky_dataset1 = dataset['train'].filter(lambda x: x['manual_rating'] > 0)
    picky_dataset2 = dataset['test'].filter(lambda x: x['manual_rating'] > 0)

    dataset = DatasetDict({
        'train': concatenate_datasets([picky_dataset1, picky_dataset2])
    })
    return dataset


def datamap_perfect_rating(config, dataset):
    picky_dataset1 = dataset['train'].filter(lambda x: x['manual_rating'] > 1)
    picky_dataset2 = dataset['test'].filter(lambda x: x['manual_rating'] > 1)

    dataset = DatasetDict({
        'train': concatenate_datasets([picky_dataset1, picky_dataset2])
    })
    return dataset


###############
# mock model
###############
class MockModel():
    pass

class MockModelForQueryLM(MockModel):
    def __init__(self):
        self.cnt = 0

    def generate(self):
        if self.cnt == 0:
            q = r'SEARCH["a\\in\\mathbb{R}"]'
        else:
            q = r'SEARCH["\\lim_{n\\to\\infty}\\frac{a^n}{n!}=0"]'
        self.cnt += 1
        return [q + '\n\n']

###############
# collate func
###############
def collate_prompt(config, batch_tok_fn, batch_data):
    prompts = [d['prompt'] for d in batch_data]
    eos = config.getboolean('collate_add_eos', True)
    return batch_tok_fn(prompts, eos=eos), batch_data


def collate_query_cot(config, batch_tok_fn, batch_data):
    from tools.prompt_factory import cot_mytrain
    prompts = [
        cot_mytrain(d['query'])
        for d in batch_data
    ]
    eos = config.getboolean('collate_add_eos', True)
    return batch_tok_fn(prompts, eos=eos), batch_data


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


def collate_retrieve_the_dup(config, batch_tok_fn, batch_data):
    from tools.prompt_factory import tool_prompt1
    eos = config.getboolean('collate_add_eos', True)
    response_sect = '### Response:\n'
    inputs = [
        limit_length(
            config.getint('context_length'),
            tool_prompt1(
                data['Q_dup']
                .replace(r'[imath]', '$')
                .replace(r'[/imath]', '$')
            )
        )
        + '\n\n' + response_sect
        for data in batch_data
    ]
    inputs_tokenized = batch_tok_fn(inputs, eos=eos, as_list=True)
    return inputs_tokenized, batch_data


###############
# stop func
###############
def stop_on_common_stop_tokens(config, tokenizer, response):
    if tokenizer.eos_token in response:
        return True
    elif '##' in response:
        return True
    return False


###############
# reward func
###############
def reward_by_answer(config, inp, out, models):
    from main_clean import extract_math_answer
    from math_equivalence import is_equiv
    from rich import print as rich_print

    rewards = []
    for raw, out_str in zip(inp[1], out):
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
            rich_print('[green]correct![/green]')
        else:
            rich_print('[red]wrong:[/red]', out_boxed)

    return rewards


def reward_by_retriever_score(config, batch_in, batch_out, models):
    from rl_tools import (
        search_mux,
        has_any_captured,
        tool_invoke,
        ToolError
    )
    from colorama import Fore, Style

    tokenizer, model, ref_model = models
    rewards = []
    for raw, out in zip(batch_in[1], batch_out):
        if not isinstance(out, str):
            out_str = tokenizer.decode(out)
        else:
            out_str = out
        target_docid = int(raw['qid'])
        tool_map = {
            'SEARCH': partial(search_mux, 'dups', None, docid=target_docid)
        }
        if not has_any_captured(out_str, tool_map):
            rewards.append(0.)
        else:
            pre_invoke, tool_res = tool_invoke(out_str, tool_map)

            if isinstance(tool_res, ToolError):
                rewards.append(0.)
            elif len(tool_res) == 0:
                rewards.append(0.)
            else:
                docid = int(tool_res[0][1])
                if docid == target_docid:
                    score = tool_res[0][2]
                    rewards.append(score)
                else:
                    rewards.append(0.)

            print(Fore.YELLOW)
            print(f'tool res: {tool_res}')
            print(Style.RESET_ALL)

        print(Fore.MAGENTA)
        print(f'reward: {rewards[-1]}')
        print(Style.RESET_ALL)

    return rewards


###############
# step func
###############
def rl_step_default(trainer, batch_in, batch_out, rewards):
    list_batch, batch_raw = batch_in
    inps = [d['input_ids'][0] for d in list_batch]
    outs = [ids for ids in batch_out]
    rewards = list(map(torch.tensor, rewards))
    stats = trainer.step(inps, outs, rewards)
    trainer.log_stats(stats, {'response': None}, rewards)
    return stats


###############
# log func
###############
def log_problem(config, values):
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
        save_answer_log(logpath, verbose=verbose, **log)


def save_answer_log(logpath, verbose=False, **kwargs):
    from tools.inspect_output import _output_html
    with open(logpath, 'w') as fh:
        json.dump(kwargs, fh, indent=2)
    _output_html(logpath, verbose=False)
    if verbose: print(kwargs.keys())
    if verbose: print('Written log:', logpath)


def log_rl_io(config, values):
    run_name = config.name
    output_dir = os.path.join(config.get('output_dir'), run_name)
    os.makedirs(output_dir, exist_ok=True)

    step = values['step']
    k = values['k']
    log_name = f'step{step}_k{k}.log'
    log_path = os.path.join(output_dir, log_name)

    logs = []
    for inp, rwd, out in zip(
        values['batch_in'][1],
        values['rewards'],
        values['batch_outstr']):
        log = copy.deepcopy(inp)
        log.update({'reward': rwd})
        log.update({'outstr': out})
        logs.append(log)

    with open(log_path, 'w') as fh:
        json.dump(logs, fh)


if __name__ == '__main__':
    #json_file = './output/finetune-pairs.json'
    #ds_all = Dataset.from_generator(data_generator_json,
    #    gen_kwargs={'json_file': json_file})
    #ds_train = ds_all.filter(lambda x: 'train' in x['problem'])
    #ds_test = ds_all.filter(lambda x: 'test' in x['problem'])
    #dataset = DatasetDict({'train': ds_train, 'test': ds_test})

    jsonl_file = 'arqmath-question-dups.jsonl'
    ds_all = Dataset.from_generator(data_generator_jsonl,
        gen_kwargs={'jsonl_file': jsonl_file})
    dataset = DatasetDict({'train': ds_all})
    breakpoint()
    dataset.push_to_hub("approach0/MSE-duplicate-questions")
