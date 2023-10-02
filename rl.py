import os
import gc
import json
import time
import torch
import random
import wandb
import configparser
import numpy as np
from functools import partial

import sys
sys.path.insert(0, './trl')

import rl_data
from rl_data import MockModel
import rl_mcts

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


def config_flash2(config):
    if config.getboolean('flash_atten', False):
        from flash_attn_monkey_patch import (
            replace_llama_attn_with_flash_attn,
        )
        replace_llama_attn_with_flash_attn()


def get_peft_config(
    peft_attach_new=False,
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

    elif 'MockModel' in model_path:
        model = getattr(rl_data, config.get('model'))()
        ref_model = None

    elif model_path.startswith('http'):
        model = model_path
        ref_model = None
        config['model_as_server'] = '{}'
        if 'tokenizer' in config:
            del config['tokenizer']

    else:
        cache_dir = config.get('cache_dir', None)
        peft_kwargs = get_cfg_json(config, 'peft', {})
        lora_config = get_peft_config(**peft_kwargs)

        if config.get('mode') == 'rl':
            from trl import AutoModelForCausalLMWithValueHead as M
            from transformers import BitsAndBytesConfig
            if config.get('load_in_8bit', False):
                model = M.from_pretrained(
                    model_path, peft_config=lora_config,
                    cache_dir=cache_dir,
                    device_map=config.get('device_map', 'auto'),
                    load_in_8bit=True,  quantization_config=BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                    )
                )
            else:
                model = M.from_pretrained(
                    model_path, peft_config=lora_config,
                    torch_dtype=torch.float16,
                    device_map=config.get('device_map', 'auto'),
                    cache_dir=cache_dir
                )

            is_peft_model = getattr(model, "is_peft_model", False)
            if is_peft_model:
                model.pretrained_model.print_trainable_parameters()
                ref_model = None
            else:
                from trl import create_reference_model
                ref_model = create_reference_model(model)

        elif config.get('mode') in ['finetune', 'inference', 'finetune-dpo']:

            from transformers import LlamaForCausalLM
            if config.getboolean('load_in_8bit', False):

                from transformers import BitsAndBytesConfig
                model = LlamaForCausalLM.from_pretrained(model_path,
                    device_map=config.get('device_map', 'auto'),
                    cache_dir=cache_dir,
                    load_in_8bit=True,  quantization_config=BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                    )
                )
            else:
                model = LlamaForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.float16,
                    device_map=config.get('device_map', 'auto'),
                    cache_dir=cache_dir
                )

            peft_existing_kwargs = get_cfg_json(config, 'peft_existing', {})
            if peft_existing_kwargs or lora_config:
                print('Loading peft:', peft_existing_kwargs or lora_config)
                if peft_existing_kwargs:
                    # existing LoRA
                    from peft import PeftModel
                    adapter_path = peft_existing_kwargs.pop('adapter_path')
                    model = PeftModel.from_pretrained(
                        model, adapter_path, **peft_existing_kwargs)
                    model = model.merge_and_unload()
                else:
                    # new LoRA
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


def get_rl_trainer(tokenizer, model, ref_model, **kwargs):
    if 'lr' not in kwargs:
        return None
    elif isinstance(model, MockModel):
        return None

    lr = kwargs.pop('lr')
    kwargs['learning_rate'] = lr

    from lion_pytorch import Lion
    from transformers import get_constant_schedule_with_warmup
    from transformers import get_cosine_schedule_with_warmup
    optimizer = Lion(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=(0.95, 0.95),
        weight_decay=1e-3
    )
    warmup_steps = kwargs.pop('warmup_steps')
    training_steps = kwargs.pop('training_steps')
    if training_steps == 0:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps
        )
    else:
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=training_steps
        )

    from trl import PPOConfig, PPOTrainer
    config = PPOConfig(**kwargs)
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )

    return ppo_trainer


@torch.inference_mode()
def gen_stream(model, input_ids, max_new_tokens=None,
    context_len=4096, stream_interval=2, temperature=0):
    from transformers.generation.logits_process import (
        TemperatureLogitsWarper
    )
    logits_process = TemperatureLogitsWarper(temperature)
    output_ids = input_ids[0].tolist()
    prompt_len = len(input_ids[0])
    if max_new_tokens is None:
        max_new_tokens = max(0, context_len - prompt_len - 1)

    past_key_values = out = None
    i = 0
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
        # out.logits: [bs, len, vocab]
        past_key_values = out.past_key_values
        last_token_logits = out.logits[0, -1, :]

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

        # Yield the output tokens
        if i % stream_interval == 0 or i == max_new_tokens - 1:
            yield {
                "output_ids": output_ids[prompt_len:],
                "usage": {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": i,
                    "total_tokens": prompt_len + i,
                },
                "finish_reason": None,
            }

    # Finish stream event, which contains finish reason
    if i >= max_new_tokens - 1:
        finish_reason = "length"
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
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


def batch_tokenize(config, tokenizer, texts,
    eos=True, decode=False, as_list=False):
    if eos and tokenizer and not decode:
        texts = [t + tokenizer.eos_token for t in texts]

    if tokenizer is None:
        return {
            'texts': [text for text in texts]
        }
    elif decode:
        decode_kwargs = get_cfg_json(config, 'decode_kwargs', {})
        return tokenizer.decode(texts, **decode_kwargs)
    elif as_list:
        max_length = config.getint("context_length")
        return [
            tokenizer(t,
                return_tensors="pt",
                max_length=max_length,
                truncation=True
            )
            for t in texts
        ]
    else:
        max_length = config.getint("context_length")
        return tokenizer(texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="longest"
        )


def wrapup_collate(config, tokenizer):
    tok_fn = partial(batch_tokenize, config, tokenizer)
    col_fn = getattr(rl_data, config.get('collate_fn', '_'), None)
    if col_fn is None:
        return tok_fn, None
    else:
        return tok_fn, partial(col_fn, config, tok_fn)


def adapt_inputs(config, tokenizer, batch_in):
    dict_batch, batch_raw = batch_in
    if 'input_ids' not in dict_batch:
        tok_fn, collate_fn = wrapup_collate(config, tokenizer)
        if 'texts' in dict_batch: # over network?
            eos = config.getboolean('collate_add_eos', True)
            input_ids = tok_fn(dict_batch['texts'], eos=eos)['input_ids']
            dict_batch['input_ids'] = input_ids
        else: # try collate then...
            dict_batch, batch_raw = collate_fn(batch_raw)
    return dict_batch, batch_raw


from flask import Flask
app = Flask('model as server')
@app.route('/model', methods=['GET', 'POST'])
def batch_respond_handler():
    from flask import request
    config, models = app.config['args']
    batch_in = request.json['batch_in']
    return batch_respond(config, models, batch_in)


def batch_respond(config, models, batch_in, trainer=None):
    bs = config.getint('batch_size')
    verbose = config.getboolean('verbose', False)
    tokenizer, model, ref_model = models
    dict_batch, batch_raw = batch_in
    decode_kwargs = get_cfg_json(config, 'decode_kwargs', {})
    stop_fn = getattr(rl_data, config.get('stop_fn', '_'), None)
    if stop_fn: stop_fn = partial(stop_fn, config, tokenizer)
    def print_if_verbose(in_texts, res):
        if verbose:
            for a, b in zip(in_texts, res): print(f'{a}{b}')

    if hasattr(model, 'pretrained_model'):
        assert trainer is not None
        device = model.pretrained_model.device
        dict_batch, batch_raw = adapt_inputs(config, tokenizer, batch_in)
        list_batch = dict_batch
        input_ids = [d['input_ids'][0].to(device) for d in list_batch]
        rl_respond_kwargs = get_cfg_json(config, 'rl_respond_kwargs', {})
        response = trainer.generate(
            input_ids, return_prompt=False,
            **rl_respond_kwargs
        )
        if get_cfg_json(config, 'model_as_server', {}):
            res = [
                tokenizer.decode(response[b], **decode_kwargs)
                for b in range(bs)
            ] # texts
            in_texts = tokenizer.decode(input_ids, **decode_kwargs)
            print_if_verbose(in_texts, res)
            return res
        else:
            return response # logits

    elif config.get('model') == 'openai_api':
        gen_kwargs = get_cfg_json(config, 'openai_gen', {})
        in_texts = dict_batch['texts']
        res = model.complete(in_texts, stop_fn, gen_kwargs)
        print_if_verbose(in_texts, res)
        return res

    elif isinstance(model, MockModel):
        return model.generate()

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
            print('Error code:', res.status_code)
            quit(1)
    else:
        device = model.device
        dict_batch, batch_raw = adapt_inputs(config, tokenizer, batch_in)
        if isinstance(dict_batch, list):
            input_ids = dict_batch[0]['input_ids'].to(device)
        else:
            input_ids = dict_batch['input_ids']
            input_ids = input_ids.to(device) # bs, L

        gen_kwargs = get_cfg_json(config, 'gen_kwargs', {})
        stream = gen_kwargs.pop('stream')
        for output in gen_stream(model, input_ids, **gen_kwargs):
            text = tokenizer.decode(output['output_ids'],
                **decode_kwargs)
            finr = output['finish_reason']
            usage = output['usage']
            if stream:
                print("\033c", end='')
                print(text)
                time.sleep(0.5)
            if stop_fn and stop_fn(text):
                output['finish_reason'] = 'stop_fn'
                break
        if stream:
            print('Usage:', usage)
            print('Finish reason:', finr)

        in_texts = [
            tokenizer.decode(input_ids[b], **decode_kwargs)
            for b in range(len(input_ids))
        ]
        print_if_verbose(in_texts, [text])
        return [text]


def prepare_experiment(config):
    config_rerope(config)
    config_flash2(config)

    if config.get('mode') in ['finetune', 'finetune-dpo']:
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
        #hg_trainer_args.remove_unused_columns = False
        hg_trainer_args._n_gpu = 1

    models = get_models(config)
    tokenizer, model, ref_model = models

    # prepare dataset
    from datasets import Dataset
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from datasets.dataset_dict import DatasetDict

    dataset_gen_fn = getattr(rl_data, config.get('dataset_gen_fn', '_'), None)
    if dataset_gen_fn:
        dataset_gen_args = get_cfg_json(config, 'dataset', {})
        dataset = Dataset.from_generator(
            dataset_gen_fn, gen_kwargs=dataset_gen_args)
        dataset = DatasetDict({'train': dataset})
    else:
        dataset_path = config.get('dataset')
        dataset = load_dataset(dataset_path)

    dataset_map_fn = getattr(rl_data, config.get('dataset_map_fn', '_'), None)
    if dataset_map_fn:
        dataset = dataset_map_fn(config, dataset)

    if config.getboolean('dataset_shuffle', False):
        dataset = dataset.shuffle(seed=config.getint('seed'))

    if config.get('eval_during_train', 'no') != 'no':
        dataset = dataset['train']
        dataset = dataset.train_test_split(test_size=1)

    def get_data_range(data):
        data_offset = config.getint('data_offset', 0)
        data_cutoff = config.getint('data_cutoff', len(data))
        data_cutoff = min(data_cutoff, len(data))
        return data_offset, data_cutoff

    print(dataset)
    _, collate_fn = wrapup_collate(config, tokenizer)

    # prepare dataset loader
    if config.get('mode') in ['rl', 'inference']:
        ds_key = config.get('dataset_key', 'train')
        print('Use dataset key:', ds_key)
        data = dataset[ds_key]
        data_range = get_data_range(data)
        data = data.select(range(*data_range))
        dataloader = DataLoader(data,
            collate_fn=collate_fn,
            batch_size=config.getint('batch_size')
        )

        if config.get('mode') == 'rl':
            trainer_kwargs = get_cfg_json(config, 'trainer', {})
            trainer = get_rl_trainer(*models, **trainer_kwargs)
        else:
            trainer = None

    elif config.get('mode') == 'finetune':
        data = dataset['train']
        data_range = get_data_range(data)
        data = data.select(range(*data_range))
        dataloader=None
        if 'test' not in dataset:
            dataset['test'] = None
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=hg_trainer_args,
            train_dataset=data,
            eval_dataset=dataset['test'],
            data_collator=collate_fn
        )
        trainer.deepspeed = trainer.model_wrapped

    elif config.get('mode') == 'finetune-dpo':
        data = dataset['train']
        data_range = get_data_range(data)
        data = data.select(range(*data_range))
        dataloader=None
        if 'test' not in dataset:
            dataset['test'] = None
        from trl import DPOTrainer
        trainer = DPOTrainer(
            model, ref_model,
            args=hg_trainer_args,
            beta=config.getfloat('dpo_beta'),
            train_dataset=data,
            eval_dataset=dataset['test'],
            tokenizer=tokenizer,
            max_length=config.getint("context_length"),
            max_target_length=config.getint("context_length"),
            max_prompt_length=config.getint("context_length")
        )
        trainer.deepspeed = trainer.model_wrapped

    else:
        raise NotImplemented

    return models, trainer, dataloader, data_range


def parse_metric_config(config):
    metric = config.get('metric', 'pass@1')
    metric_name, K = metric.split('@')
    K = int(K)
    assert K > 0
    assert metric_name in ['pass', 'maj']
    return K


def do_experiment(config, inject_args):
    # inject arguments
    inject_arguments(config, inject_args)
    # prepare logging
    if run_uid := config.get('run_uid', False):
        pass
    elif config.getboolean('wandb', False):
        wandb.init(
            project=config.name,
            name=config.get('run', None),
            config=dict(config.items())
        )
        run_uid = wandb.run.name
    else:
        from datetime import datetime
        run_uid = datetime.today().strftime('%Y-%m-%d__%H_%M_%S')

    # make ex_output_dir
    output_dir = config.get('output_dir', '.')
    ex_output_dir = os.path.join(
        config.get('output_dir'), config.name, run_uid
    )
    os.makedirs(ex_output_dir, exist_ok=True)

    K = parse_metric_config(config)
    set_seed(config.getint('seed', 42))
    models, trainer, dataloader, data_range = prepare_experiment(config)

    if app_args := get_cfg_json(config, 'model_as_server', {}):
        app.config['args'] = (config, models)
        app.run(**app_args)
        quit(0)

    tokenizer, model, _ = models
    data_offset, data_cutoff = data_range

    if config.get('mode') == 'rl':
        mcts_fn = getattr(rl_mcts, config.get('mcts_fn'))
        rwd_fn = getattr(rl_data, config.get('reward_fn'))
        stp_fn = getattr(rl_data, config.get('step_fn', '_'), None)
        log_fn = getattr(rl_data, config.get('log_fn', '_'), None)
        if log_fn: log_fn = partial(log_fn, config, ex_output_dir)
        save_steps = config.getint('rl_save_steps', 1000)
        for step, batch_in in enumerate(dataloader):
            step  = step + data_offset
            mcts_fn(step, K, config, models, batch_in, trainer,
                res_fn=batch_respond, rwd_fn=rwd_fn,
                stp_fn=stp_fn, log_fn=log_fn)
            save_tick = step % save_steps
            if save_tick == 0 and hasattr(model, 'save_pretrained'):
                    trainer.save_pretrained(ex_output_dir)
            print(f'Save tick: {save_tick} % {save_steps}')
            print(f'Progress: {step+1} / {data_cutoff}')

        if hasattr(model, 'save_pretrained'):
            trainer.save_pretrained(ex_output_dir)

    elif config.get('mode') in ['finetune', 'finetune-dpo']:
        from torch import autocast
        with autocast(device_type="cuda"):
            trainer.train()
        trainer.save_model(ex_output_dir)

    elif config.get('mode') == 'inference':
        mcts_fn = getattr(rl_mcts, config.get('mcts_fn'))
        rwd_fn = getattr(rl_data, config.get('reward_fn', '_'), None)
        log_fn = getattr(rl_data, config.get('log_fn', '_'), None)
        if log_fn: log_fn = partial(log_fn, config, ex_output_dir)
        for step, batch_in in enumerate(dataloader):
            step  = step + data_offset
            mcts_fn(step, K, config, models, batch_in, None,
                res_fn=batch_respond, rwd_fn=rwd_fn,
                stp_fn=None, log_fn=log_fn)
            print(f'Progress: {step+1} / {data_cutoff}')

    else:
        raise NotImplemented


def inject_arguments(config, inject_args):
    for key, val in inject_args.items():
        print('[inject config]:', key, '=>', val)
        config[key] = str(val)


def main(*experiments, config_file='rl.ini', **inject_args):
    cfg = configparser.ConfigParser(allow_no_value=True)
    cfg.read(config_file)

    local_rank = os.getenv("LOCAL_RANK", "0")

    for path in get_cfg_json(cfg['DEFAULT'], 'add_sys_paths', []):
        print('insert sys_path:', path)
        sys.path.insert(0, path)

    for ex in experiments:
        assert ex in cfg.sections()

    for ex in experiments:
        config = cfg[ex]
        config['local_rank'] = local_rank
        do_experiment(config, inject_args)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(main)
