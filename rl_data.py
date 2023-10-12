import re
import os
import sys
import copy
import json
import torch
from functools import partial
from datasets import load_dataset
from datasets import Dataset, concatenate_datasets
from datasets.dataset_dict import DatasetDict

IGNORE_INDEX = -100


###############
# utilities
###############
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

    if debug and config.getint('local_rank') == 0:
        decode_show(batch_tok_fn, examples_tokenized['input_ids'][0])
        #decode_show(batch_tok_fn, examples_tokenized['labels'][0])

    return examples_tokenized


def limit_length(limit, string):
    if len(string) > limit:
        return string[:limit] + '...'
    else:
        return string


#####################
# dataset generator
#####################
def generate_from_json(path):
    with open(path, 'r') as fh:
        j = json.load(fh)
    for item in j:
        yield item


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


def datamap_topic_filter(config, dataset, topic='precalculus', topic_key='src_path'):
    dataset['train'] = dataset['train'].filter(lambda x: topic in x[topic_key])
    dataset['test'] = dataset['test'].filter(lambda x: topic in x[topic_key])
    return dataset


def datamap_DPO(config, dataset, dataset_key='train'):
    from tools.prompt_factory import DPO_default_prompt
    dpo_dataset_dict = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }

    for data in dataset[dataset_key]:
        judged = data['judge_buffer'][0]
        correct = judged['is_equiv']
        if correct: continue
        instr = data['instruction']
        input = data['input']
        truth = data['output']
        answer = judged['answer']
        prompt = DPO_default_prompt(instr, input)

        dpo_dataset_dict['prompt'].append(prompt)
        dpo_dataset_dict['chosen'].append(truth)
        dpo_dataset_dict['rejected'].append(answer)

    dataset = Dataset.from_dict(dpo_dataset_dict)
    return DatasetDict({dataset_key: dataset})


def datamap_double_train_for_query_and_answer(config, dataset):
    ds = dataset['train']
    qry_column = ["query"] * len(ds)
    ans_column = ["answer"] * len(ds)
    qry_ds = ds.add_column("train_for", qry_column)
    qry_ds = qry_ds.filter(lambda x: x['correct']) # ensure good query.
    ans_ds = ds.add_column("train_for", ans_column)
    dataset = DatasetDict({
        'train': concatenate_datasets([qry_ds, ans_ds])
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
def collate_none(config, batch_tok_fn, batch_data):
    return None, batch_data


def collate_prompt(config, batch_tok_fn, batch_data):
    prompts = [d['prompt'] for d in batch_data]
    eos = config.getboolean('collate_add_eos', True)
    return batch_tok_fn(prompts, eos=eos), batch_data


def collate_cot_mytrain(config, batch_tok_fn, batch_data):
    from tools.prompt_factory import cot_mytrain
    query_key = config.get('collate__query_key', 'query')
    prompts = [cot_mytrain(d[query_key]) for d in batch_data]
    eos = config.getboolean('collate_add_eos', True)
    return batch_tok_fn(prompts, eos=eos), batch_data


def collate_cot_wizard(config, batch_tok_fn, batch_data):
    from tools.prompt_factory import cot_wizard
    query_key = config.get('collate__query_key', 'query')
    prompts = [cot_wizard(d[query_key]) for d in batch_data]
    eos = config.getboolean('collate_add_eos', True)
    return batch_tok_fn(prompts, eos=eos), batch_data


def collate_tora(config, batch_tok_fn, batch_data):
    from tools.prompt_factory import prompt_tora
    query_key = config.get('collate__query_key', 'query')
    prompts = [prompt_tora(d[query_key]) for d in batch_data]
    eos = config.getboolean('collate_add_eos', True)
    return batch_tok_fn(prompts, eos=eos), batch_data


def collate_query_state_prompt(config, batch_tok_fn, batch_data):
    from tools.prompt_factory import cot2, multihop_results1
    query_key = config.get('collate__query_key', 'query')
    prompts = []
    for d in batch_data:
        if d['tool_res']:
            prompts.append(
                d['prompt'] + d['out_str'] +
                multihop_results1(d['tool_res'])
            )
        else:
            prompts.append(
                cot2(d[query_key])
            )
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
    from tools.prompt_factory import tool_prompt1 # find_good_keywords_1
    eos = config.getboolean('collate_add_eos', True)
    response_sect = '### Response:\n'
    inputs = [
        limit_length(
            config.getint('context_length'),
            tool_prompt1(
            #find_good_keywords_1(
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


def collate_phase2_infer(config, batch_tok_fn, batch_data):
    from tools.prompt_factory import tool_prompt1
    query_key = config.get('collate__query_key', 'query')
    for data in batch_data:
        query = data[query_key]
        prompt = tool_prompt1(query)
        response_sect = '### Response:\n'
        new_prompt = prompt + '\n\n' + response_sect
        data['prompt'] = new_prompt
    inputs = [d['prompt'] + '\n' for d in batch_data]
    eos = config.getboolean('collate_add_eos', True)
    inputs_tokenized = batch_tok_fn(inputs, eos=eos)
    return inputs_tokenized, batch_data


def collate_final_dataset_for_querylm(config, batch_tok_fn, batch_data):
    from tools.prompt_factory import find_good_keywords_1
    for data in batch_data:
        prompt = find_good_keywords_1(data['problem'])
        response_sect = '### Response:'
        data['prompt'] = prompt + '\n\n' + response_sect
    sources = [d['prompt'] + '\n' for d in batch_data]
    targets = [d['search_query'] + '\n' for d in batch_data]
    return collate_pr(config, batch_tok_fn, sources, targets)


def collate_final_dataset_for_judger(config, batch_tok_fn, batch_data):
    from tools.prompt_factory import ask_relevance_1
    for data in batch_data:
        prompt = ask_relevance_1(data['problem'], data['search_result'])
        response_sect = '### Response:'
        data['prompt'] = prompt + '\n\n' + response_sect
    sources = [d['prompt'] + '\n' for d in batch_data]
    targets = ['rate[' + str(d['relevance']) + ']\n' for d in batch_data]
    return collate_pr(config, batch_tok_fn, sources, targets)


def collate_final_dataset_for_generalist(config, batch_tok_fn, batch_data):
    from tools.prompt_factory import final_tool_augment_prompt1, multihop_results1
    for data in batch_data:
        if data['train_for'] == 'query':
            data['prompt'] = final_tool_augment_prompt1(data['problem'])
            data['target'] = data['aug_query']

        elif data['train_for'] == 'answer':
            data['prompt'] = final_tool_augment_prompt1(data['problem'])
            data['prompt'] += data['aug_query'] + '\n' 
            data['prompt'] += multihop_results1(data['aug_result'])

            if data['relevance'] == 2:
                assert data['correct']
                data['target'] = 'The result looks highly relevant! I will absolutely use it to answer the question.\n\n'
                data['target'] += data['answer']

            elif data['relevance'] == 1:
                assert data['correct']
                data['target'] = 'The result might be helpful, I will try using it to answer the question only if it is useful.\n\n'
                data['target'] += data['answer']

            else:
                assert not data['correct'] and data['relevance'] == 0
                data['target'] = 'The result looks irrelevant, I will completely ignore it and answer the question directly.\n\n'
                data['target'] += data['solution']

        else:
            assert False, 'invalid train_for'

    sources = [d['prompt'] for d in batch_data]
    targets = [d['target'] for d in batch_data]
    return collate_pr(config, batch_tok_fn, sources, targets)


def collate_generalist_infer(config, batch_tok_fn, batch_data):
    from tools.prompt_factory import final_tool_augment_prompt1
    query_key = config.get('collate__query_key', 'query')
    for data in batch_data:
        query = data[query_key]
        data['prompt'] = final_tool_augment_prompt1(query)
    inputs = [d['prompt'] for d in batch_data]
    eos = config.getboolean('collate_add_eos', True)
    inputs_tokenized = batch_tok_fn(inputs, eos=eos)
    return inputs_tokenized, batch_data


###############
# stop func
###############
def stop_on_common_stop_tokens(config, tokenizer, response):
    eos_token = '</s>' if tokenizer is None else tokenizer.eos_token
    if eos_token in response:
        return True
    elif '##' in response:
        return True
    return False


def stop_on_common_stop_and_boxed_tokens(config, tokenizer, response):
    common_stop = stop_on_common_stop_tokens(config, tokenizer, response)
    if common_stop:
        return True
    finished_lines = response.split('\n')[:-1]
    boxed_lines = list(map(lambda x: '\\boxed' in x, finished_lines))
    return any(boxed_lines)


###############
# reward func
###############
def reward_by_answer(config, inp, out, models, sol_key='solution'):
    from main_clean import extract_math_answer
    from math_equivalence import is_equiv

    rewards = []
    for raw, out_str in zip(inp[1], out):
        ground_truth = extract_math_answer(raw[sol_key])
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

        print('ground truth:', ground_truth)
        if equiv:
            print('correct!')
        else:
            print('wrong:', out_boxed)

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
    math_only = config.getboolean('math_only', False)
    for raw, out in zip(batch_in[1], batch_out):
        if not isinstance(out, str):
            out_str = tokenizer.decode(out)
        else:
            out_str = out
        target_docid = int(raw['qid'])
        uri = 'dups_math_only' if math_only else 'dups'
        tool_map = {
            'SEARCH': partial(
                search_mux, uri, None, docid=target_docid
            )
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
def rl_step_default(config, trainer, batch_in, batch_out, rewards):
    list_batch, batch_raw = batch_in
    inps = [d['input_ids'][0] for d in list_batch]
    outs = [ids for ids in batch_out]
    rewards = list(map(torch.tensor, rewards))
    stats = trainer.step(inps, outs, rewards)
    return stats


###############
# log func
###############
def log_problem(config, ex_output_dir, values,
    problem_key='problem', query_key='query'):
    step = values['step']
    for b, inp in enumerate(values['batch_in'][1]):
        log = copy.deepcopy(inp)
        log_name = log[problem_key].strip('.').replace('/', '_')
        log_name = f'{step:06}_batch{b}-' + log_name + '.log'
        logpath = os.path.join(ex_output_dir, log_name)
        save_answer_log(logpath, query_key, **log)


def save_answer_log(logpath, query_key, **kwargs):
    from tools.inspect_output import _output_html
    with open(logpath, 'w') as fh:
        json.dump(kwargs, fh, indent=2)
    _output_html(logpath, query_key=query_key, verbose=False)


def log_rl_default(config, ex_output_dir, values):
    step = values['step']
    stats = values['stats']
    rewards = values['rewards']
    logs = {
        'timestep': [
            f'step={step}, b={b}'
            for b in range(len(rewards))
        ],
        'response': [out for out in values['batch_outstr']]
    }
    from rl import get_cfg_json
    for col in get_cfg_json(config, 'log_columns', []):
        logs[col] = [inp[col] for inp in values['batch_in'][1]]
    values['trainer'].log_stats(stats, logs, rewards,
        columns_to_log=logs.keys())


def log_query_state(config, ex_output_dir, values,
    problem_key='problem', query_key='query'):
    step = values['step']
    for b, (inp, out) in enumerate(
        zip(values['batch_in'][1], values['batch_out'])
    ):
        log = copy.deepcopy(inp)
        log_name = log[problem_key].strip('.').replace('/', '_')
        log_name = f'{step:06}_batch{b}-' + log_name + '.log'
        logpath = os.path.join(ex_output_dir, log_name)
        with open(logpath, 'w') as fh:
            json.dump(log, fh, indent=2)


def log_json(config, ex_output_dir, step, path, j, sol=None):
    log = {
        'index': step,
        'path': path,
        'solution': sol,
        'json': j
    }
    log_name = path.strip('.').replace('/', '_')
    log_name = f'{step:06}-' + log_name + '.log'
    logpath = os.path.join(ex_output_dir, log_name)
    with open(logpath, 'w') as fh:
        json.dump(log, fh, indent=2)


if __name__ == '__main__':
    jsonl_file = 'arqmath-question-dups.jsonl'
    ds_all = Dataset.from_generator(data_generator_jsonl,
        gen_kwargs={'jsonl_file': jsonl_file})
    dataset = DatasetDict({'train': ds_all})
    breakpoint()
    dataset.push_to_hub("approach0/MSE-duplicate-questions")
