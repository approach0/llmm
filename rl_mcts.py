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


def direct_answering(step, k, config, models, batch_in, trainer,
    res_fn, rwd_fn, stp_fn=None, log_fn=None):
    tokenizer, model, ref_model = models
    dict_batch, batch_raw = batch_in
    batch_out = res_fn(config, models, batch_in)

    rewards = rwd_fn(config, batch_in, batch_out, models,
        sol_key='output')
    if log_fn: log_fn(locals(),
        problem_key='src_path', query_key='instruction')


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
        log_fn(locals())


def infer_query_lm(step, k, config, models, batch_in, trainer,
    res_fn, rwd_fn, stp_fn=None, log_fn=None):
    tokenizer, model, ref_model = models
    dict_batch, batch_raw = batch_in
    batch_out = res_fn(config, models, batch_in)

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
            inp['tool_res'] = None
        else:
            pre_invoke, tool_res = tool_invoke(out_str, tool_map)
            inp['tool_res'] = tool_res

    log_fn(locals(), problem_key='problem', query_key='query')


if __name__ == '__main__':
    pass
