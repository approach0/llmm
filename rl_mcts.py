from tools.prompt_factory import *
from functools import partial

from rl import get_cfg_json

import rl_data
from rl_tools import (
    search_mux,
    has_any_captured,
    tool_invoke,
    ToolError
)


def direct_answering(step, k, config, models, batch_in, trainer,
    res_fn, rwd_fn, stp_fn=None, log_fn=None):
    tokenizer, model, ref_model = models
    dict_batch, batch_raw = batch_in
    batch_out = res_fn(config, models, batch_in)
    rewards = rwd_fn(config, batch_in, batch_out, models,
        **get_cfg_json(config, 'reward_args', {})
    )
    if log_fn:
        log_fn(locals(), **get_cfg_json(config, 'log_args', {}))


def rl_query_lm(step, k, config, models, batch_in, trainer,
    res_fn, rwd_fn, stp_fn=None, log_fn=None):
    tokenizer, model, ref_model = models
    list_batch, batch_raw = batch_in
    print(tokenizer.decode(list_batch[0]['input_ids'][0]))

    batch_out = res_fn(config, models, batch_in, trainer=trainer)
    print(tokenizer.decode(batch_out[0]))
    rewards = rwd_fn(config, batch_in, batch_out, models,
        **get_cfg_json(config, 'reward_args', {})
    )

    if config.getboolean('compare_refout', False):
        import numpy as np
        with model.pretrained_model.disable_adapter():
            ref_batch_out = res_fn(config, models, batch_in, trainer=trainer)
        ref_rewards = rwd_fn(config, batch_in, ref_batch_out, models,
            **get_cfg_json(config, 'reward_args', {})
        )
        cmp_rewards = np.array(rewards) - np.array(ref_rewards)
        rewards = cmp_rewards.tolist()

    if stp_fn and trainer:
        stats = stp_fn(config, trainer, batch_in, batch_out, rewards)

    if log_fn:
        batch_outstr = [
            out if isinstance(out, str) else tokenizer.decode(out)
            for out in batch_out
        ]
        log_fn(locals(), **get_cfg_json(config, 'log_args', {}))


def infer_query_lm(step, k, config, models, batch_in, trainer,
    res_fn, rwd_fn, stp_fn=None, log_fn=None):
    tokenizer, model, ref_model = models
    dict_batch, batch_raw = batch_in
    batch_out = res_fn(config, models, batch_in)
    query_key = config.get('collate__query_key', 'query')

    for inp, out_str in zip(batch_in[1], batch_out):
        out_str = out_str.replace('</s>', '').replace('<s>', '')
        inp['out_str'] = out_str
        uri = 'mabowdor'
        query = inp[query_key]
        tool_map = {
            'SEARCH': partial(
                search_mux, uri, query
            )
        }
        if not has_any_captured(out_str, tool_map):
            inp['tool_res'] = None
        else:
            pre_invoke, tool_res = tool_invoke(out_str, tool_map)
            if isinstance(tool_res, ToolError):
                inp['tool_res'] = None
            else:
                inp['tool_res'] = tool_res

    log_fn(locals(), **get_cfg_json(config, 'log_args', {}))


class Node():
    def __init__(self, node_type, state):
        assert node_type in ['Q', 'K', 'R', 'A']
        self.node_type = node_type
        self.state = state
        self.children = []
        self.parent = None

    def branch(self, node_type, child_state):
        node = Node(node_type, child_state)
        self.children.append(node)
        node.parent = self
        return node

    def print(self):
        pass

    @staticmethod
    def gn(config, models, tok_fn, res_fn, inp):
        prompts = tok_fn([inp], eos=False)
        out = res_fn(config, models, (prompts, None))
        return out[0]

    def keywords(self, config, models, tok_fn, res_fn, tm):
        assert self.node_type == 'Q'
        inp = tool_prompt1(self.state)
        inp += '\n\n### Response:\n'
        out = Node.gn(config, models, tok_fn, res_fn, inp)
        if has_any_captured(out, tm):
            out, _ = tool_invoke(out, tm,
                dryrun=True, args=[self.state])
            return out
        else:
            return None

    def search(self, config, models, tok_fn, res_fn, tm):
        assert self.node_type == 'K'
        assert self.parent.node_type == 'Q'
        _, res = tool_invoke(self.state, tm,
                args=self.parent.state)
        if isinstance(res, ToolError):
            print('ToolError:', res)
            return None
        elif len(res) == 0:
            print('Empty results!')
            return None
        else:
            return res

    def query(self, config, models, tok_fn, res_fn, tm):
        out = self.keywords(config, models,
            tok_fn, res_fn, tm)
        if out is None:
            return None
        k_node = self.branch('K', out)
        results = k_node.search(config, models,
            tok_fn, res_fn, tm)
        for res in results:
            k_node.branch('R', res)
        return k_node


def mcts_explore(step, k, config, models, batch_in, trainer,
    res_fn, rwd_fn, stp_fn=None, log_fn=None):
    from rl import batch_tokenize
    tokenizer, model, ref_model = models
    tok_fn = partial(batch_tokenize, config, tokenizer)
    tool_map = {
        'SEARCH': partial(
            search_mux, 'mabowdor'
        )
    }
    params = config, models, tok_fn, res_fn, tool_map

    root = Node('Q', batch_in['input'][0])
    curr = root
    while True:
        out = curr.query(*params)
        breakpoint()
        quit(1)


if __name__ == '__main__':
    pass
