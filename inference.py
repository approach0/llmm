import fire
from load_hg_ckpt import load_hg_llama
from transformers import LlamaTokenizer
from generate import Generater


def main(path, prompt='My name is Mariama, my favorite ',
    debug=True, device='cpu'):
    tokenizer = LlamaTokenizer.from_pretrained(path)
    model = load_hg_llama(path, debug=debug, device=device)
    model = Generater(model, tokenizer)
    print('Prompt:', prompt)
    return model.generate([prompt], debug=debug)


if __name__ == '__main__':
    fire.Fire(main)
