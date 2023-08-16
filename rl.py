import torch
import transformers
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM
import bitsandbytes as bnb

from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from trl import create_reference_model
from trl.core import respond_to_batch

from rerope_patch import replace_llama_attn_with_rerope


def get_rl_trainer(model):
    config = PPOConfig(
        batch_size=1,
        optimize_cuda_cache=True
    )

    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=3e-5)
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=None,
        tokenizer=tokenizer,
        optimizer=optimizer
    )

    return ppo_trainer


if __name__ == '__main__':
    attn = transformers.models.llama.modeling_llama.LlamaAttention
    replace_llama_attn_with_rerope(attn)

    model_path = 'lmsys/vicuna-7b-v1.5'
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    from peft import LoraConfig, get_peft_model
    TARGET_MODULES = [
        "q_proj",
        "v_proj",
    ]
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16, lora_dropout=0.05,
        lora_alpha=32, bias='none',
        target_modules=TARGET_MODULES
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_path, device_map="auto",
        peft_config=lora_config
    )

    is_peft_model = getattr(model, "is_peft_model", False)

    if is_peft_model:
        model.pretrained_model.print_trainable_parameters()

    tokens = tokenizer(
        'I need ',
        return_tensors="pt",
        padding="longest",
        max_length=12,
        truncation=True,
    )

    device = model.pretrained_model.device
    question_tensors = tokens['input_ids'].to(device)
    response_tensors = respond_to_batch(model, question_tensors)
    print(tokenizer.decode(response_tensors[0]))

    rl_trainer = get_rl_trainer(model)

    rewards = [torch.tensor(1.0)]
    stats = rl_trainer.step([question_tensors[0]], [response_tensors[0]], rewards)
    #print(stats)
