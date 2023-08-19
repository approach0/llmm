import os
import torch
import configparser
from functools import partial

import sys
sys.path.insert(0, './trl')
from trl.core import respond_to_batch


ADAPT_CFG = "adapter_config.json"


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


def get_model(tokenizer_path, model_path,
    peft_attach_new=False,
    peft_lora_rank=16,
    peft_lora_dropout=0.05,
    peft_lora_alpha=32,
    tokenizer_init_kwargs={},
    tokenizer_special_tokens=None):

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
        **tokenizer_init_kwargs)
    if tokenizer_special_tokens:
        tokenizer.add_special_tokens(tokenizer_special_tokens)

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
    else:
        lora_config = None

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
    return tokenizer(texts,
        return_tensors="pt",
        max_length=config.getint("context_length"),
        truncation=True,
        padding=True
    )


def test(config, tokenizer, model, trainer):
    inputs = batch_tokenize(config, tokenizer,
        [config.get('test_prompt')])

    device = model.pretrained_model.device
    input_ids = inputs['input_ids'].to(device) # bs, L
    print(tokenizer.decode(input_ids[0]))

    response = respond_to_batch(model, input_ids) # bs, L
    print(tokenizer.decode(response[0]))

    rewards = [torch.tensor(1.0)]
    stats = trainer.step(
        [input_ids[0]], [response[0]], rewards
    )
    print(stats)


def prepare_experiment(config):
    config_rerope(config)

    peft_kwargs = get_cfg_json(config, 'peft', {})
    tokenizer_kwargs = get_cfg_json(config, 'tokenizer_init_kwargs', {})
    special_tokens = get_cfg_json(config, 'tokenizer_special_tokens', None)
    models = get_model(
        config.get('tokenizer'),
        config.get('model'),
        tokenizer_init_kwargs=tokenizer_kwargs,
        tokenizer_special_tokens=special_tokens,
        **peft_kwargs,
    )

    ppo_kwargs = get_cfg_json(config, 'ppo', {})
    trainer = get_rl_trainer(*models, **ppo_kwargs)
    return models, trainer


def do_experiment(config):
    models, trainer = prepare_experiment(config)
    tokenizer, model, ref_model = models

    if config.get('test_prompt', False):
        test(config, tokenizer, model, trainer)
        return

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
    device = model.pretrained_model.device
    for dict_batch, batch_raw in dataloader:
        input_ids = dict_batch['input_ids']
        input_ids = input_ids.to(device) # bs, L
        response = respond_to_batch(model, input_ids)

        for b in range(bs):
            print(tokenizer.decode(response[b]))

        #from rl_mcts import mcts_query
        #mcts_query(config, tokenizer, model, trainer)
        #quit()


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
