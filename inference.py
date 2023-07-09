import os
from load_hg_ckpt import load_hg_llama
from transformers import LlamaConfig
from transformers import LlamaTokenizer
from llms.llama import LlamaForCausalLM
from generate import Generater
import bmtrain


def rank_print(*args, **kargs):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = -1
    if 'rank' in kargs:
        rank = kargs['rank']
        del kargs['rank']
    if rank == -1 or local_rank == rank:
        print(f'rank#{local_rank}:', *args, **kargs)


def main(token_path, model_path,
    prompt='My name is Mariama, my favorite ',
    debug=False, seed=3407):
    local_rank = int(os.environ["LOCAL_RANK"])

    bmtrain.init_distributed(seed=seed)

    rank_print('loading tokenizer...')
    tokenizer = LlamaTokenizer.from_pretrained(token_path)

    rank_print('allocating model...')
    config = LlamaConfig.from_pretrained(model_path)
    model = LlamaForCausalLM(config) # allocate space
    rank_print('allocated.')

    rank_print('loading checkpoints from rank0 ...')
    ckpt = os.path.join(model_path, "state_dict.pt")
    model = bmtrain.load(model, ckpt, strict=True)

    rank_print('Prompt:', prompt, rank=0)
    if local_rank == 0:
        model = Generater(model, tokenizer)
        model.generate([prompt], debug=debug)
    bmtrain.synchronize()


if __name__ == '__main__':
    import fire
    fire.Fire(main)
