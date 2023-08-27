from tools.prompt_factory import *

class State():
    def __init__(self, prompt):
        self.prompt = prompt
        self.children = []

    def branch(self, child_state):
        self.children.append(child_state)


def mcts(step, config, models, batch_in, trainer,
    res_fn, rwd_fn, stp_fn=None, log_fn=None):
    tokenizer, model, ref_model = models
    batch_out = res_fn(config, models, batch_in)
    rewards = rwd_fn(config, batch_in, batch_out, models)
    if stp_fn:
        stats = stp_fn(trainer, batch_in, batch_out, rewards)
    if log_fn:
        log_fn(config, locals())


if __name__ == '__main__':
    pass
