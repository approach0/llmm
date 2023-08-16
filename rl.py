import os
import torch
import configparser

import sys
sys.path.insert(0, './trl')


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
        ref_model = create_reference_model(model)

    return tokenizer, model, ref_model


def get_rl_trainer(model):
    config = PPOConfig(
        batch_size=1,
        optimize_cuda_cache=True
    )

    import bitsandbytes as bnb
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=3e-5)
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        optimizer=optimizer
    )

    return ppo_trainer


def do_experiment(config):
    config_rerope(config)

    kwargs = get_cfg_json(config, 'peft', {})
    tokenizer, model, ref_model = get_model(
        config.get('tokenizer'),
        config.get('model'),
        **kwargs
    )


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
    #model_path = 'lmsys/vicuna-7b-v1.5'

    #tokens = tokenizer(
    #    'I need ',
    #    return_tensors="pt",
    #    padding="longest",
    #    max_length=12,
    #    truncation=True,
    #)

    #device = model.pretrained_model.device
    #prompt_ids = tokens['input_ids'].to(device) # bs, L
    #response = respond_to_batch(model, prompt_ids) # bs, L
    #print(tokenizer.decode(response[0]))

    #rl_trainer = get_rl_trainer(model)
    #rewards = [torch.tensor(1.0)]
    #stats = rl_trainer.step(
    #    [prompt_ids[0]],
    #    [response[0]],
    #    rewards
    #)
#
#from trl import PPOConfig, PPOTrainer, 
#from trl import create_reference_model
#from trl.core import respond_to_batch
#
#from rerope_patch import replace_llama_attn_with_rerope
