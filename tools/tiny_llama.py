from transformers import LlamaConfig
from transformers import LlamaForCausalLM


def tiny_llama():
    llama_cfg = LlamaConfig()
    llama_cfg.num_hidden_layers = 1
    llama_cfg.hidden_size = 8
    llama_cfg.intermediate_size = 8
    llama_cfg.num_attention_heads = 2
    llama_cfg.num_key_value_heads = 2
    print(llama_cfg)
    model = LlamaForCausalLM(llama_cfg)
    return model


model = tiny_llama()
model.save_pretrained('./output/tiny_llama')
print('saved.')
