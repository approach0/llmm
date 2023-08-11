import os
import gc
import json
import time
import torch

from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM


def merge_adapter(origin_model_path, adapter_path, output_path='./tmp',
    load_in_8bit=False, cache_dir=None, is_trainable=False):

    model = LlamaForCausalLM.from_pretrained(origin_model_path,
        load_in_8bit=load_in_8bit, cache_dir=cache_dir)
    model_and_lora = PeftModel.from_pretrained(model, adapter_path,
        adapter_name='default', is_trainable=is_trainable)
    model = model_and_lora.merge_and_unload()
    print(model)
    model.save_pretrained(output_path)


def quantize(tokenizer_path, model_path, quantized_model_path):
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    quantize_config = BaseQuantizeConfig(
        bits=4, group_size=128, desc_act=False
    )

    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    examples = [
        tokenizer(
            "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
        )
    ]
    model = AutoGPTQForCausalLM.from_pretrained(model_path,
        quantize_config)
    model.to('cuda')
    model.quantize(examples)
    model.save_quantized(quantized_model_path)


@torch.inference_mode()
def generate_stream(model, tokenizer, prompt, device, context_len=2048,
    max_new_tokens=512, stream_interval=2, mode='greedy', stop_token_ids=[]):
    len_prompt = len(prompt)
    stop_token_ids.append(tokenizer.eos_token_id)
    input_ids = tokenizer(prompt).input_ids
    max_src_len = context_len - max_new_tokens - 1
    input_ids = input_ids[-max_src_len:]
    output_ids = list(input_ids)
    input_echo_len = len(input_ids)
    past_key_values = out = None
    sent_interrupt = False
    for i in range(max_new_tokens):
        if i == 0:  # prefill
            out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:  # decoding
            out = model(
                input_ids=torch.as_tensor(
                    [[token] if not sent_interrupt else output_ids], device=device
                ),
                use_cache=True,
                past_key_values=past_key_values if not sent_interrupt else None,
            )
            sent_interrupt = False
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0, -1, :]

        if mode == 'greedy':
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        elif mode == 'sample':
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]
        else:
            raise NotImplementedError
        token = tokens[0]
        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        # Yield the output tokens
        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            tmp_output_ids = output_ids[input_echo_len:]
            rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

            yield {
                "text": output,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": i,
                    "total_tokens": input_echo_len + i,
                },
                "finish_reason": None,
            }

        if stopped:
            break

    # Finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # Clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


def load_model(tokenizer_path, model_path):
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, legacy=False)
    model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto")
    return tokenizer, model


def generate(debug=False, **kargs):
    output_stream = generate_stream(**kargs)

    cur_text, finish_reason = None, None
    for cur in output_stream:
        cur_text = cur['text']
        finish_reason = cur["finish_reason"]
        if debug:
            print("\033c", end='')
            print(cur_text)
            time.sleep(0.5)

    if debug: print('finish reason:', finish_reason)
    return cur_text


def test(prompt):
    # maximum 2 gpus
    gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    tokenizer, model = load_model('lmsys/vicuna-7b-v1.3', 'lmsys/vicuna-7b-v1.3')
    device = 'cuda:' + gpus.split(',')[0]
    generate(tokenizer=tokenizer, model=model, prompt=prompt,
        device=device, debug=True)


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire({
        'merge_adapter': merge_adapter,
        'quantize': quantize,
        'test': test,
    })
