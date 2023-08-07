import torch; print(torch.cuda.is_available())
print(torch.cuda.is_bf16_supported())

import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()

from transformers import LlamaForCausalLM

from flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)
replace_llama_attn_with_flash_attn()
