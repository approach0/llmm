import sys
sys.path.insert(0, '../FastChat')

#from fastchat.serve.inference import
from fastchat.model.model_adapter import load_model, get_generate_stream_function
from fastchat.utils import get_context_length
from fastchat.model.model_adapter import get_model_adapter

#model_path = 'lmsys/vicuna-33b-v1.3'


def api_init(device='cuda', n_gpus=1, model_path='lmsys/vicuna-7b-v1.3'):
    model, tokenizer = load_model(model_path, device, n_gpus)
    generate_stream_func = get_generate_stream_function(model, model_path)

    context_len = get_context_length(model.config)
    print('model loaded. ctx len:', context_len)

    return model, tokenizer, generate_stream_func, context_len, device, model_path


def api(prompt, args=None, debug=False):
    model, tokenizer, generate_stream_func, context_len, device, model_path = args

    adapter = get_model_adapter(model_path)
    template = adapter.get_default_conv_template(model_path)

    actual_prompt = f'{template.roles[0]}: {prompt}\n\n{template.roles[1]}:'
    if debug:
        print(actual_prompt)

    gen_params = {
        "model": model_path,
        "prompt": actual_prompt,
        "temperature": 0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 1024,
        "stop": None,
        "stop_token_ids": [],
        "echo": False,
    }

    output_stream = generate_stream_func(
        model,
        tokenizer,
        gen_params,
        device,
        context_len=context_len,
        judge_sent_end=False
    )

    cur_text = None
    for cur in output_stream:
        cur_text = cur['text']
        if debug:
            print("\033c", end='')
            #print(prompt, end='')
            print(cur_text)
    return cur_text


if __name__ == '__main__':
    api_args = api_init()
    output = api('How to download a windows?', args=api_args, debug=True)
    print(output)
