from tools.prompt_factory import *


class State():
    def __init__(self, prompt):
        self.prompt = prompt
        self.children = []

    def branch(self, child_state):
        self.children.append(child_state)


def mcts(config, models, batch_in,
    res_fn=None, rwd_fn=None, stp_fn=None):

    breakpoint()
    batch_out = res_fn(config, models, batch_in)
    for raw, out_str in zip(batch_in[1], batch_out):
        target_docid = raw['qid']
    a=1+1


if __name__ == '__main__':
    pass
