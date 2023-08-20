import os
import random
import torch
import configparser
import numpy as np
from functools import partial

import sys
sys.path.insert(0, './trl')
from trl.core import respond_to_batch


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
    import json
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
    peft_lora_rank=16,
    peft_lora_dropout=0.05,
    peft_lora_alpha=32):

    if peft_attach_new:
        adapter_config = {
            'task_type': "CAUSAL_LM",
            'r': peft_lora_rank,
            'lora_dropout': peft_lora_dropout,
            'lora_alpha': peft_lora_alpha,
            'bias': 'none',
            'target_modules': [
                "q_proj",
                "v_proj",
            ]
        }

        from peft import LoraConfig
        lora_config = LoraConfig(**adapter_config)
        return lora_config
    else:
        return None


def get_model(config):
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
    else:
        tokenizer = None

    model_path = config.get('model')
    if model_path == 'openai_api':
        from rl_openai import OpenAI_API
        kwargs = get_cfg_json(config, 'openai_init', {})
        model = OpenAI_API(**kwargs)
        ref_model = None
    else:
        peft_kwargs = get_cfg_json(config, 'peft', {})
        lora_config = get_peft_config(**peft_kwargs)

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


def batch_tokenize(config, tokenizer, texts):
    if tokenizer is None:
        return {
            'texts': [text for text in texts]
        }
    else:
        max_length = config.getint("context_length")
        return tokenizer(texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )


def batch_respond(config, models, batch_in):
    bs = config.getint('batch_size')
    tokenizer, model, ref_model = models
    dict_batch, batch_raw = batch_in

    if hasattr(model, 'pretrained_model'):
        device = model.pretrained_model.device

        input_ids = dict_batch['input_ids']
        input_ids = input_ids.to(device) # bs, L
        response = respond_to_batch(model, input_ids)
        return [
            tokenizer.decode(response[b])
            for b in range(bs)
        ]
    else:
        gen_kwargs = get_cfg_json(config, 'openai_gen', {})
        in_texts = dict_batch['texts']
        return model.complete(in_texts, gen_kwargs)


def prepare_experiment(config):
    os.makedirs(config.get('output_dir', '.'), exist_ok=True)
    set_seed(config.getint('seed', 42))
    config_rerope(config)

    models = get_model(config)

    ppo_kwargs = get_cfg_json(config, 'ppo', {})
    trainer = get_rl_trainer(*models, **ppo_kwargs)
    return models, trainer


def do_experiment(config):
    models, trainer = prepare_experiment(config)
    tokenizer, model, _ = models

    from datasets import load_dataset
    from torch.utils.data import DataLoader
    dataset_path = config.get('dataset')
    dataset = load_dataset(dataset_path, split="train")

    import rl_data
    bs = config.getint('batch_size')
    tok_fn = partial(batch_tokenize, config, tokenizer)
    col_fn = getattr(rl_data, config.get('collate_fn'))
    dataloader = DataLoader(dataset,
        collate_fn=partial(col_fn, tok_fn),
        batch_size=bs
    )

    rwd_fn = getattr(rl_data, config.get('reward_fn'))
    stp_fn = getattr(rl_data, config.get('step_fn'))

    for batch_in in dataloader:
        batch_out = batch_respond(config, models, batch_in)
        rewards = rwd_fn(config, batch_in, batch_out)
        stp_fn(config, batch_in, batch_out, trainer, rewards)
        #print(batch_in[1][b]['prompt'])
        #print(batch_out[b])
        #rewards = [torch.tensor(1.0)]
        #stats = trainer.step(
        #    [input_ids[b]], [response[b]], rewards
        #)
        #from rl_mcts import mcts_query
        #mcts_query(config, tokenizer, model, trainer)


def main(*experiments, config_file='rl.ini'):
    cfg = configparser.ConfigParser()
    cfg.read(config_file)

    for ex in experiments:
        assert ex in cfg.sections()

    for ex in experiments:
        do_experiment(cfg[ex])


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(main)
