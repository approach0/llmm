import torch
from tools.prompt_factory import *
from rl_tools import search_mux, has_any_captured, tool_invoke, ToolError
from functools import partial


class State():
    def __init__(self, prompt):
        self.prompt = prompt
        self.children = []

    def branch(self, child_state):
        self.children.append(child_state)


def mcts(config, models, batch_in, trainer,
    res_fn=None, rwd_fn=None, stp_fn=None):
    tokenizer, model, ref_model = models
    batch_out = res_fn(config, models, batch_in)

    rewards = []
    outs = []
    inps = []
    for input_ids, raw, out in zip(
        batch_in[0]['input_ids'], batch_in[1], batch_out):
        #breakpoint()
        if not isinstance(out, str):
            out_str = tokenizer.decode(out)
        else:
            out_str = out
        target_docid = int(raw['qid'])
        tool_map = {
            'SEARCH': partial(search_mux, 'dups', None, docid=target_docid)
        }
        if not has_any_captured(out_str, tool_map):
            rewards.append(0.)
        else:
            pre_invoke, tool_res = tool_invoke(out_str, tool_map)
            if isinstance(tool_res, ToolError):
                rewards.append(1.)
            elif len(tool_res) == 0:
                rewards.append(1.)
            else:
                score = tool_res[0][2]
                rewards.append(score)
        outs.append(out)
        inps.append(input_ids)
    rewards = list(map(torch.tensor, rewards))
    stats = trainer.step(inps, outs, rewards)
    return stats


if __name__ == '__main__':
    pass
