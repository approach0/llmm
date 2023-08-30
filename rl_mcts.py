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
    quit()

    if stp_fn and trainer:
        stats = stp_fn(trainer, batch_in, batch_out, rewards)
    if log_fn:
        batch_outstr = [
            out if isinstance(out, str) else tokenizer.decode(out)
            for out in batch_out
        ]
        log_fn(config, locals())


if __name__ == '__main__':
    pass
