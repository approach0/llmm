from tools.prompt_factory import *
from functools import partial

import rl_data
from rl_tools import (
    search_mux,
    has_any_captured,
    tool_invoke,
    ToolError
)


class State():
    def __init__(self, prompt):
        self.prompt = prompt
        self.children = []

    def branch(self, child_state):
        self.children.append(child_state)


def rl_query_lm(step, k, config, models, batch_in, trainer,
    res_fn, rwd_fn, stp_fn=None, log_fn=None):
    tokenizer, model, ref_model = models
    list_batch, batch_raw = batch_in
    print(tokenizer.decode(list_batch[0]['input_ids'][0]))

    batch_out = res_fn(config, models, batch_in, trainer=trainer)
    print(tokenizer.decode(batch_out[0]))
    rewards = rwd_fn(config, batch_in, batch_out, models)

    if config.getboolean('compare_refout', False):
        import numpy as np
        with model.pretrained_model.disable_adapter():
            ref_batch_out = res_fn(config, models, batch_in, trainer=trainer)
        ref_rewards = rwd_fn(config, batch_in, ref_batch_out, models)
        cmp_rewards = np.array(rewards) - np.array(ref_rewards)
        rewards = cmp_rewards.tolist()

    if stp_fn and trainer:
        stats = stp_fn(config, trainer, batch_in, batch_out, rewards)

    if log_fn:
        batch_outstr = [
            out if isinstance(out, str) else tokenizer.decode(out)
            for out in batch_out
        ]
        log_fn(config, locals())


def infer_query_lm(step, k, config, models, batch_in, trainer,
    res_fn, rwd_fn, stp_fn=None, log_fn=None):
    tokenizer, model, ref_model = models
    dict_batch, batch_raw = batch_in
    batch_out = res_fn(config, models, batch_in)

    from rl_openai import OpenAI_API
    from rl import get_cfg_json
    gpt3_5 = OpenAI_API(**get_cfg_json(config, 'openai_init', {}))
    stop_fn = getattr(rl_data, config.get('stop_fn', '_'), None)
    stop_fn = partial(stop_fn, config, tokenizer)

    rewards = []
    for inp, out_str in zip(batch_in[1], batch_out):
        out_str = out_str.replace('</s>', '').replace('<s>', '')
        prompt = inp['prompt']
        query = inp['query']

        uri = 'mabowdor'
        tool_map = {
            'SEARCH': partial(
                search_mux, uri, query
            )
        }
        if not has_any_captured(out_str, tool_map):
            rewards.append(0.)
            continue

        pre_invoke, tool_res = tool_invoke(out_str, tool_map)

        if isinstance(tool_res, ToolError):
            rewards.append(0.)
            continue
        elif len(tool_res) == 0:
            rewards.append(0.)
            continue

        new_prompt = ia_mytrain(query, out_str, *tool_res)
        responses = gpt3_5.complete([new_prompt],
            stop_fn, get_cfg_json(config, 'openai_gen', {}))
        print(responses[0])
        #breakpoint()


if __name__ == '__main__':
    pass
