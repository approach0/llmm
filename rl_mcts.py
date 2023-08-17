from tools.prompt_factory import *


def mcts_query(config, tokenizer, model, trainer):
    inputs = tokenizer(
        multihop_simple('Solve $x^2 = 4$.'),
        return_tensors="pt"
    ).to('cuda')
    output = model.generate(**inputs, max_length=1024, do_sample=True, temperature=0.7)
    answer = tokenizer.decode(output[0])
    print(answer)
    print('\n\n')


if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read('rl.ini')
    config = cfg['test']
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    mcts_query(config, tokenizer, model, None)
