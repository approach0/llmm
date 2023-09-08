from tools.prompt_factory import *

class State():
    def __init__(self, prompt):
        self.prompt = prompt
        self.children = []

    def branch(self, child_state):
        self.children.append(child_state)


def mcts(step, k, config, models, batch_in, trainer,
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


if __name__ == '__main__':
    pass
