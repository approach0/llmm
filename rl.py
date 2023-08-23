import os
import json
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


def get_models(config):
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

    model_path = config.get('model')
    if model_path == 'openai_api':
        from rl_openai import OpenAI_API
        kwargs = get_cfg_json(config, 'openai_init', {})
        model = OpenAI_API(**kwargs)
        ref_model = None
    else:
        peft_kwargs = get_cfg_json(config, 'peft', {})
        lora_config = get_peft_config(**peft_kwargs)

        if config.getboolean('use_rl', True):
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
        else:
            from transformers import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16
            )

            if lora_config is not None:
                from peft import get_peft_model
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()
            ref_model = None

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


def batch_tokenize(config, tokenizer, texts,
    eos=True, decode=False):
    if tokenizer is None:
        return {
            'texts': [text for text in texts]
        }
    elif decode:
        return tokenizer.decode(texts)
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
    config_rerope(config)

    if not config.getboolean('use_rl', True):
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
    dataset = load_dataset(dataset_path, split="train")
    dataset = dataset.shuffle(seed=config.getint('seed'))
    dataset = dataset.train_test_split(test_size=1)

    import rl_data
    tok_fn = partial(batch_tokenize, config, tokenizer)
    col_fn = getattr(rl_data, config.get('collate_fn'))

    if config.getboolean('use_rl', True):
        bs = config.getint('batch_size')
        dataloader = DataLoader(dataset['train'],
            collate_fn=partial(col_fn, tok_fn),
            batch_size=bs
        )

        trainer_kwargs = get_cfg_json(config, 'trainer', {})
        trainer = get_rl_trainer(*models, **trainer_kwargs)
    else:
        dataloader=None
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=hg_trainer_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=partial(col_fn, tok_fn)
        )
        trainer.deepspeed = trainer.model_wrapped

    return models, trainer, dataloader


def do_experiment(config):
    os.makedirs(config.get('output_dir', '.'), exist_ok=True)
    set_seed(config.getint('seed', 42))

    models, trainer, dataloader = prepare_experiment(config)
    tokenizer, model, _ = models
    final_save_path = os.path.join(
        config.get('output_dir'), config.name
    )

    if config.getboolean('use_rl', True):
        import rl_data
        rwd_fn = getattr(rl_data, config.get('reward_fn'))
        stp_fn = getattr(rl_data, config.get('step_fn'))

        for step, batch_in in enumerate(dataloader):
            batch_out = batch_respond(config, models, batch_in)
            rewards = rwd_fn(config, batch_in, batch_out, models)
            stp_fn(config, step, trainer, rewards)

        model.save_pretrained(final_save_path)
    else:
        from torch import autocast
        with autocast(device_type="cuda"):
            trainer.train()
        trainer.save_model(final_save_path)

    #rewards = [torch.tensor(1.0)]
    #stats = trainer.step(
    #    [input_ids[b]], [response[b]], rewards
    #)
    #from rl_mcts import mcts_query
    #mcts_query(config, tokenizer, model, trainer)


def main(*experiments, config_file='rl.ini'):
    cfg = configparser.ConfigParser()
    cfg.read(config_file)
    #json.loads(cfg['finetune__7b_vicuna_v1_5']['trainer'])

    for ex in experiments:
        assert ex in cfg.sections()

    for ex in experiments:
        do_experiment(cfg[ex])


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(main)
