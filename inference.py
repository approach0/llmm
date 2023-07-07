import fire
from load_hg_ckpt import load_hg_llama
from transformers import LlamaTokenizer
from generate import Generater


def main(path):
    tokenizer = LlamaTokenizer.from_pretrained(path)
    model = load_hg_llama(path, debug=False)
    model = Generater(model, tokenizer)
    return model.generate(['My name is Mariama, my favorite '], debug=True)


if __name__ == '__main__':
    fire.Fire(main)
