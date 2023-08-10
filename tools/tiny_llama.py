from transformers import LlamaConfig
from transformers import LlamaForCausalLM


def tiny_llama(target_model_path='lmsys/vicuna-13b-v1.5-16k', **kargs):
    config = LlamaConfig.from_pretrained(target_model_path, **kargs)
    config.num_hidden_layers = 1
    print(config)
    model = LlamaForCausalLM(config)
    return model


if __name__ == '__main__':
    model = tiny_llama()
    print('saving...')
    model.save_pretrained('./output/tiny_llama')
    print('saved.')
