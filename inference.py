import os
from load_hg_ckpt import load_hg_llama
from transformers import LlamaConfig
from transformers import LlamaTokenizer
from llms.llama import LlamaForCausalLM
from generate import Generater
import bmtrain as bmt


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
    debug=False, device='cpu', seed=3407):

    bmt.init_distributed(seed=seed)

    rank_print('loading tokenizer...')
    tokenizer = LlamaTokenizer.from_pretrained(token_path)

    rank_print('allocating model...')
    config = LlamaConfig.from_pretrained(model_path)
    model = LlamaForCausalLM(config) # allocate space
    rank_print('allocated.')

    ckpt = os.path.join(model_path, "state_dict.pt")
    bmt.load(model, ckpt, strict=True)
    bmt.synchronize()

    #model = Generater(model, tokenizer)
    rank_print('Prompt:', prompt, rank=0)
    #return model.generate([prompt], debug=debug)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
