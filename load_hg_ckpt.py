import os
import fire
import json
import torch


# almost copied from https://github.com/pytorch/pytorch/blob/789b1437e945336f83c915ab2f2dd283ac472191/torch/nn/modules/module.py#L1919
def load(module, state_dict, prefix='', debug=False):
    used_keys = set()
    for name, child in module._modules.items():
        if child is not None:
            leaf = f'{prefix}{name}.weight'
            if debug: print('loading param', leaf)
            if leaf in state_dict:
                t = state_dict[leaf]
                with torch.no_grad():
                    assert child.weight.shape == t.shape
                    module._modules[name].to_empty(device='cpu')
                    module._modules[name].weight.copy_(t)
                used_keys.add(leaf)
            child_prefix = prefix + name + '.'
            child_state_dict = {
                k: v
                for k, v in state_dict.items()
                if k.startswith(child_prefix)
            }
            u = load(child, child_state_dict, child_prefix, debug)
            used_keys = used_keys.union(u)
    return used_keys


def load_hg_llama(path, debug=False):
    from llms.llama import LlamaModel
    from transformers import LlamaConfig

    config = LlamaConfig.from_pretrained(path)
    print('Creating model ...')
    with torch.device("meta"):
        model = LlamaModel(config)

    with open(os.path.join(path, "pytorch_model.bin.index.json"), 'r') as f:
        index = json.load(f)
    shards = set([v for k, v in index["weight_map"].items()])
    src_state_dict = {}
    for shard in shards:
        print('Loading model shard:', shard)
        state_dict = torch.load(os.path.join(path, shard))
        src_state_dict.update(state_dict)

    if debug: print(model.layers[30].mlp.gate_proj.weight)
    used_keys = load(model, src_state_dict, 'model.', debug=debug)
    if debug: print(model.layers[30].mlp.gate_proj.weight)
    all_keys = set(src_state_dict.keys())
    unused_keys = all_keys.difference(used_keys)
    print('Unused keys:', unused_keys)
    return model


if __name__ == '__main__':
    fire.Fire(load_hg_llama)
