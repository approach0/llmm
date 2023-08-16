import os
import torch
import configparser

import sys
sys.path.insert(0, './trl')
from trl.core import respond_to_batch


ADAPT_CFG = "adapter_config.json"


class State():
    def __init__(self, prompt):
        self.prompt = prompt
        self.children = []

    def branch(self, child_state):
        self.children.append(child_state)


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
    peft_lora_alpha=32):

    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

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


def simple_test(config, tokenizer, model):
    inputs = tokenizer(
        config.get('test_prompt'),
        return_tensors="pt",
        padding="longest",
        max_length=config.getint("context_length"),
        truncation=True,
    )

    device = model.pretrained_model.device

    input_ids = inputs['input_ids'].to(device) # bs, L
    print(tokenizer.decode(input_ids[0]))

    response = respond_to_batch(model, input_ids) # bs, L
    print(tokenizer.decode(response[0]))

    rewards = [torch.tensor(1.0)]
    stats = rl_trainer.step(
        [input_ids[0]], [response[0]], rewards
    )
    print(stats)


def do_experiment(config):
    config_rerope(config)

    peft_kwargs = get_cfg_json(config, 'peft', {})
    models = get_model(
        config.get('tokenizer'),
        config.get('model'),
        **peft_kwargs
    )
    tokenizer, model, ref_model = models

    ppo_kwargs = get_cfg_json(config, 'ppo', {})
    trainer = get_rl_trainer(*models, **ppo_kwargs)

    if config.get('test_prompt', False):
        simple_test(config, tokenizer, model)
        quit(0)


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
