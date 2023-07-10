import os
import fire
import json
import torch
import shutil


# almost copied from https://github.com/pytorch/pytorch/blob/789b1437e945336f83c915ab2f2dd283ac472191/torch/nn/modules/module.py#L1919
def load(module, state_dict, prefix='', debug=False):
    used_keys = set()
    params_and_buffers = list(module._modules.items())
    params_and_buffers += list(module._buffers.items())
    for item in params_and_buffers:
        name, child = item
        if child is not None:
            is_buffer = True if torch.is_tensor(child) else False
            leaf = f'{prefix}{name}' if is_buffer else f'{prefix}{name}.weight'
            if debug: print('loading param:', leaf)
            if leaf in state_dict:
                t = state_dict[leaf]
                with torch.no_grad():
                    if is_buffer:
                        assert child.shape == t.shape
                        module.register_buffer(name, t, persistent=True)
                    else:
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
            if is_buffer: continue
            u = load(child, child_state_dict, child_prefix, debug)
            used_keys = used_keys.union(u)

    return used_keys


def load_hg_llama(path, debug=False, device='cpu'):
    from llms.llama import LlamaForCausalLM
    from transformers import LlamaConfig

    config = LlamaConfig.from_pretrained(path)
    print('Creating model ...')
    with torch.device("meta"):
        model = LlamaForCausalLM(config)

    with open(os.path.join(path, "pytorch_model.bin.index.json"), 'r') as f:
        index = json.load(f)
    shards = set([v for k, v in index["weight_map"].items()])
    src_state_dict = {}
    for shard in shards:
        print('Loading model shard:', shard)
        state_dict = torch.load(os.path.join(path, shard))
        for k, v in state_dict.items():
            k = k.replace('input_layernorm', 'norm1')
            k = k.replace('post_attention_layernorm', 'norm2')
            src_state_dict[k] = v

    if debug: print(model.model.layers[30].mlp.gate_proj.weight)
    used_keys = load(model, src_state_dict, '', debug=debug)
    if debug: print(model.model.layers[30].mlp.gate_proj.weight)
    all_keys = set(src_state_dict.keys())
    unused_keys = all_keys.difference(used_keys)
    if len(unused_keys) > 0:
        print('Unused keys:', unused_keys)

    return model.to(device)


def convert(inpath, outpath, debug=False):
    os.makedirs(outpath, exist_ok=True)
    cfg_src = os.path.join(inpath, "config.json")
    cfg_dst = os.path.join(outpath, "config.json")
    shutil.copyfile(cfg_src, cfg_dst)
    model = load_hg_llama(inpath, debug=debug, device='cpu')
    states = model.state_dict()
    if debug: print('saving', states.keys())
    torch.save(states, os.path.join(outpath, "state_dict.pt"))


if __name__ == '__main__':
    fire.Fire(convert)
