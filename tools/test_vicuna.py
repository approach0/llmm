import sys
sys.path.insert(0, '../FastChat')

#from fastchat.serve.inference import
from fastchat.model.model_adapter import load_model, get_generate_stream_function
from fastchat.utils import get_context_length

model_path = 'lmsys/vicuna-7b-v1.3'

model, tokenizer = load_model(model_path, 'cpu', 0)
generate_stream_func = get_generate_stream_function(model, model_path)

gen_params = {
    "model": model_path,
    "prompt": 'hello!',
    "temperature": 0,
    "repetition_penalty": 1.0,
    "max_new_tokens": 512,
    "stop": None,
    "stop_token_ids": [],
    "echo": False,
}

context_len = get_context_length(model.config)
output_stream = generate_stream_func(
    model,
    tokenizer,
    gen_params,
    'cpu',
    context_len=context_len,
    judge_sent_end=False
)

for cur in output_stream:
    print(cur['text'])
