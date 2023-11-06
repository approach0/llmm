import os
import torch
from transformers import LlamaConfig
from transformers import LlamaTokenizer
from llms.llama import LlamaForCausalLM
from generate import Generater


def rank_print(*args, **kargs):
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f'rank#{local_rank}:', *args, **kargs)


def main(token_path, model_path,
    prompt='My name is Mariama, my favorite ',
    debug=False, seed=3407):
    local_rank = int(os.environ["LOCAL_RANK"])

    rank_print('loading tokenizer...')
    tokenizer = LlamaTokenizer.from_pretrained(token_path)

    rank_print('loading model...')
    config = LlamaConfig.from_pretrained(model_path)
    model = LlamaForCausalLM(config)
    rank_print('allocated.')

    ckpt = os.path.join(model_path, "state_dict.pt")
    state_dict = torch.load(ckpt)
    rank_print('state dict loaded.')

    model.load_state_dict(state_dict)
    model.to('cuda:0')
    rank_print('model loaded.')

    rank_print('Prompt:', prompt)

    model = Generater(model, tokenizer)
    answer = model.generate([prompt], debug=debug, max_new_tokens=16)
    rank_print('Answer:', answer)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
