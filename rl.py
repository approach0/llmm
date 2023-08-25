import os
import gc
import json
import time
import torch
import random
import configparser
import numpy as np
from functools import partial

import sys
sys.path.insert(0, './trl')
from trl.core import respond_to_batch

import rl_data

ADAPT_CFG = "adapter_config.json"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_cfg_json(config, name, default):
    value = config.get(name, default)
    if value:
        try:
            return json.loads(value)
        except Exception as e:
            print(value, '\n', e)
            quit(1)
    else:
        return value


def config_rerope(config):
    from rerope_patch import restore_llama_attn
    from rerope_patch import replace_llama_attn_with_rerope
    restore_llama_attn()
    if rerope := get_cfg_json(config, 'rerope', False):
        replace_llama_attn_with_rerope(**rerope)


def get_peft_config(peft_attach_new=False,
    peft_lora_rank=8,
    peft_lora_dropout=0.05,
    peft_lora_alpha=16,
    peft_lora_targets=["q_proj", "v_proj"]):

    if peft_attach_new:
        adapter_config = {
            'task_type': "CAUSAL_LM",
            'r': peft_lora_rank,
            'lora_dropout': peft_lora_dropout,
            'lora_alpha': peft_lora_alpha,
            'bias': 'none',
            'target_modules': peft_lora_targets
        }

        from peft import LoraConfig
        lora_config = LoraConfig(**adapter_config)
        return lora_config
    else:
        return None


def get_models(config):
    model_path = config.get('model')
    if model_path == 'openai_api':
        from rl_openai import OpenAI_API
        kwargs = get_cfg_json(config, 'openai_init', {})
        model = OpenAI_API(**kwargs)
        ref_model = None

    elif model_path.startswith('http'):
        model = model_path
        ref_model = None
        config['model_as_server'] = '{}'
        if 'tokenizer' in config:
            del config['tokenizer']

    else:
        peft_kwargs = get_cfg_json(config, 'peft', {})
        lora_config = get_peft_config(**peft_kwargs)

        if config.get('mode') == 'rl':
            from trl import AutoModelForCausalLMWithValueHead as M
            model = M.from_pretrained(
                model_path, device_map="auto",
                peft_config=lora_config
            )

            is_peft_model = getattr(model, "is_peft_model", False)
            if is_peft_model:
                model.pretrained_model.print_trainable_parameters()
                ref_model = None
            else:
                from trl import create_reference_model
                ref_model = create_reference_model(model)

        elif config.get('mode') in ['finetune', 'inference']:

            from transformers import LlamaForCausalLM
            if config.get('load_in_8bit', False):

                from transformers import BitsAndBytesConfig
                model = LlamaForCausalLM.from_pretrained(model_path,
                    load_in_8bit=True,  quantization_config=BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                    )
                )
            else:
                model = LlamaForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.float16
                )

            if lora_config is not None:
                from peft import get_peft_model
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()
            ref_model = None

        else:
            raise NotImplemented

    tokenizer_path = config.get('tokenizer', None)
    if tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer_init_kwargs = get_cfg_json(config,
            'tokenizer_init_kwargs', {})
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
            **tokenizer_init_kwargs)
        special_tokens = get_cfg_json(config,
            'tokenizer_special_tokens', None)
        if special_tokens:
            tokenizer.add_special_tokens(special_tokens)
        print('tokenizer pad_token_id:', tokenizer.pad_token_id)
        print('tokenizer bos_token_id:', tokenizer.bos_token_id)
        print('tokenizer eos_token_id:', tokenizer.eos_token_id)
    else:
        tokenizer = None

    return tokenizer, model, ref_model


def get_rl_trainer(tokenizer, model, ref_model, **ppo_kwargs):
    if 'lr' not in ppo_kwargs:
        return None

    import bitsandbytes as bnb
    lr = ppo_kwargs.pop('lr')
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=lr)

    from trl import PPOConfig, PPOTrainer
    config = PPOConfig(**ppo_kwargs)
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optimizer=optimizer
    )

    return ppo_trainer


@torch.inference_mode()
def gen_stream(model, input_ids, context_len=4096,
    stream_interval=2, temperature=0, stop_token_ids=[]):
    from transformers.generation.logits_process import (
        TemperatureLogitsWarper
    )
    logits_process = TemperatureLogitsWarper(temperature)
    output_ids = input_ids[0].tolist()
    prompt_len = len(input_ids[0])
    max_new_tokens = context_len - prompt_len - 1
    past_key_values = out = None

    for i in range(max_new_tokens):
        if i == 0:  # prefill
            out = model(input_ids, use_cache=True)
        else:  # decoding
            last_id = torch.as_tensor([[token]],
                device=input_ids.device)
            out = model(
                input_ids=last_id,
                use_cache=True,
                past_key_values=past_key_values,
            )
        logits = out.logits # [bs, len, vocab]
        past_key_values = out.past_key_values
        last_token_logits = logits[0, -1, :]

        if temperature < 1e-5:
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            last_token_logits = logits_process(None, last_token_logits)
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]
        token = tokens[0]
        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        # Yield the output tokens
        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            yield {
                "output_ids": output_ids[prompt_len:],
                "usage": {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": i,
                    "total_tokens": prompt_len + i,
                },
                "finish_reason": None,
            }

        if stopped:
            break

    # Finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    yield {
        "output_ids": output_ids[prompt_len:],
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": i,
            "total_tokens": prompt_len + i,
        },
        "finish_reason": finish_reason,
    }

    # Clean
    del past_key_values, out, logits
    gc.collect()
    torch.cuda.empty_cache()


def batch_tokenize(config, tokenizer, texts,
    eos=True, decode=False):
    if tokenizer is None:
        return {
            'texts': [text for text in texts]
        }
    elif decode:
        decode_kwargs = get_cfg_json(config, 'decode_kwargs', {})
        return tokenizer.decode(texts, **decode_kwargs)
    else:
        if eos:
            texts = [t + tokenizer.eos_token for t in texts]
        max_length = config.getint("context_length")
        return tokenizer(texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="longest"
        )


def wrapup_collate(config, tokenizer):
    tok_fn = partial(batch_tokenize, config, tokenizer)
    col_fn = getattr(rl_data, config.get('collate_fn'))
    return partial(col_fn, config, tok_fn)


from flask import Flask
app = Flask('model as server')
@app.route('/model', methods=['GET', 'POST'])
def server_handler():
    from flask import request
    config, models = app.config['args']
    batch_in = request.json['batch_in']
    return batch_respond(config, models, batch_in)


def batch_respond(config, models, batch_in):
    bs = config.getint('batch_size')
    tokenizer, model, ref_model = models
    dict_batch, batch_raw = batch_in
    decode_kwargs = get_cfg_json(config, 'decode_kwargs', {})

    if hasattr(model, 'pretrained_model'):
        device = model.pretrained_model.device
        input_ids = dict_batch['input_ids']
        input_ids = input_ids.to(device) # bs, L

        response = respond_to_batch(model, input_ids)
        return [
            tokenizer.decode(response[b], **decode_kwargs)
            for b in range(bs)
        ]

    elif config.get('model') == 'openai_api':
        gen_kwargs = get_cfg_json(config, 'openai_gen', {})
        in_texts = dict_batch['texts']
        return model.complete(in_texts, gen_kwargs)

    elif isinstance(model, str) and model.startswith('http'):
        import requests
        res = requests.post(model, json={'batch_in': batch_in})
        if res.ok:
            try:
                return res.json()
            except:
                print(res.text)
                quit(1)
        else:
            print(res.status_code)
            quit(1)
    else:
        device = model.device
        if 'input_ids' not in dict_batch:
            collate_fn = wrapup_collate(config, tokenizer)
            dict_batch, batch_raw = collate_fn(batch_raw)
        input_ids = dict_batch['input_ids']
        input_ids = input_ids.to(device) # bs, L

        from utils2 import generate
        gen_kwargs = get_cfg_json(config, 'gen_kwargs', {})
        gen_kwargs['stop_token_ids'].append(tokenizer.eos_token_id)
        stream = gen_kwargs.pop('stream')
        for output in gen_stream(model, input_ids, **gen_kwargs):
            text = tokenizer.decode(output['output_ids'],
                **decode_kwargs)
            finr = output['finish_reason']
            if stream:
                print("\033c", end='')
                print(text)
                time.sleep(0.5)
        if stream:
            print('Finish reason:', finr)
        return [text]


def prepare_experiment(config):
    config_rerope(config)

    if config.get('mode') == 'finetune':
        deepspeed = get_cfg_json(config, 'deepspeed', None)
        deepspeed_arg = []
        if deepspeed is not None:
            import ds_config
            ds_config_file = ds_config.create_json(**deepspeed)
            deepspeed_arg = ['--deepspeed', ds_config_file]
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            print('deepspeed', local_rank, ds_config_file)

        from transformers import Trainer, TrainingArguments
        from transformers import HfArgumentParser
        parser = HfArgumentParser(TrainingArguments)
        trainer_args = get_cfg_json(config, 'trainer', [])
        trainer_args = list(map(str, trainer_args))
        hg_trainer_args = parser.parse_args_into_dataclasses(
            args=(trainer_args + deepspeed_arg) # call __post_init__
        )[0]

        # we need to force some values ...
        hg_trainer_args.remove_unused_columns = False
        hg_trainer_args._n_gpu = 1

    models = get_models(config)
    tokenizer, model, _ = models

    from datasets import load_dataset
    from torch.utils.data import DataLoader
    dataset_path = config.get('dataset')
    dataset = load_dataset(dataset_path)

    dataset_map_fn = getattr(rl_data, config.get('dataset_map_fn'), None)
    if dataset_map_fn:
        dataset = dataset_map_fn(config, dataset)

    if config.get('eval_during_train', 'no') != 'no':
        dataset = dataset['train']
        dataset = dataset.shuffle(seed=config.getint('seed'))
        dataset = dataset.train_test_split(test_size=1)
    else:
        dataset['test'] = None

    collate_fn = wrapup_collate(config, tokenizer)

    if config.get('mode') in ['rl', 'inference']:
        bs = config.getint('batch_size')
        dataloader = DataLoader(dataset['train'],
            collate_fn=collate_fn,
            batch_size=bs
        )

        if config.get('mode') == 'rl':
            trainer_kwargs = get_cfg_json(config, 'trainer', {})
            trainer = get_rl_trainer(*models, **trainer_kwargs)
        else:
            trainer = None

    elif config.get('mode') == 'finetune':
        dataloader=None
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=hg_trainer_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=partial(col_fn, config, tok_fn)
        )
        trainer.deepspeed = trainer.model_wrapped

    else:
        raise NotImplemented

    return models, trainer, dataloader, dataset


def log_config(config, logdir, filename):
    logfile = os.path.join(logdir, filename)
    with open(logfile, 'w') as fh:
        j = dict(config.items())
        json.dump(j, fh, indent=2)
        fh.write('\n')


def parse_metric_config(config):
    metric = config.get('metric', 'pass@1')
    metric_name, K = metric.split('@')
    K = int(K)
    assert K > 0
    assert metric_name in ['pass', 'maj']
    return K


def do_experiment(config, inject_args):
    inject_arguments(config, inject_args)
    output_dir = config.get('output_dir', '.')
    experiment_output_dir = os.path.join(
        config.get('output_dir'), config.name
    )
    os.makedirs(experiment_output_dir, exist_ok=True)
    log_config(config, experiment_output_dir, 'config.ini')
    log_config(inject_args, experiment_output_dir, 'inject.ini')

    K = parse_metric_config(config)
    set_seed(config.getint('seed', 42))
    models, trainer, dataloader, dataset = prepare_experiment(config)

    if app_args := get_cfg_json(config, 'model_as_server', {}):
        app.config['args'] = (config, models)
        app.run(**app_args)
        quit(0)

    tokenizer, model, _ = models
    num_train_rows = dataset['train'].num_rows

    if config.get('mode') == 'rl':
        import rl_data
        rwd_fn = getattr(rl_data, config.get('reward_fn'))
        stp_fn = getattr(rl_data, config.get('step_fn', '_'), None)
        log_fn = getattr(rl_data, config.get('log_fn', '_'), None)
        for step, batch_in in enumerate(dataloader):
            for i in range(K):
                batch_out = batch_respond(config, models, batch_in)
                rewards = rwd_fn(config, batch_in, batch_out, models)
                if stp_fn: stp_fn(config, step, trainer, rewards)
            if log_fn: log_fn(config, locals())
            print(f'Progress: {step+1} / {num_train_rows}')

        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(experiment_output_dir)

    elif config.get('mode') == 'finetune':
        from torch import autocast
        with autocast(device_type="cuda"):
            trainer.train()
        trainer.save_model(experiment_output_dir)

    elif config.get('mode') == 'inference':
        import rl_data
        rwd_fn = getattr(rl_data, config.get('reward_fn', '_'), None)
        log_fn = getattr(rl_data, config.get('log_fn', '_'), None)
        for step, batch_in in enumerate(dataloader):
            for i in range(K):
                assert config.getint('batch_size') == 1
                batch_out = batch_respond(config, models, batch_in)
                rwd_fn(config, batch_in, batch_out, models)
            if log_fn: log_fn(config, locals())
            print(f'Progress: {step+1} / {num_train_rows}')

    else:
        raise NotImplemented

    #rewards = [torch.tensor(1.0)]
    #stats = trainer.step(
    #    [input_ids[b]], [response[b]], rewards
    #)
    #from rl_mcts import mcts_query
    #mcts_query(config, tokenizer, model, trainer)


def inject_arguments(config, inject_args):
    for key, val in inject_args.items():
        print('[inject config]:', key, '=>', val)
        config[key] = str(val)


def main(*experiments, config_file='rl.ini', **inject_args):
    cfg = configparser.ConfigParser()
    cfg.read(config_file)

    for path in get_cfg_json(cfg['DEFAULT'], 'add_sys_paths', []):
        print('insert sys_path:', path)
        sys.path.insert(0, path)

    for ex in experiments:
        assert ex in cfg.sections()

    for ex in experiments:
        config = cfg[ex]
        do_experiment(config, inject_args)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(main)
