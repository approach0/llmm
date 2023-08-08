import os
import sys
import json
import tempfile


def get_script_dir():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def create_json(base_cfg_file='ds_config_zero3.json',
    fp16=None, bf16=None, remove_train_args=False,
    en_param_offload=False, en_act_ackpt=False, en_sparse_attn=False):

    script_dir = get_script_dir()
    base_cfg_file_path = os.path.join(script_dir, base_cfg_file)

    with open(base_cfg_file_path, 'r') as fh:
        config = json.load(fh)

    if fp16 is not None:
        config['fp16']['enabled'] = fp16

    if bf16 is not None:
        config['bf16']['enabled'] = bf16

    if remove_train_args:
        config['train_batch_size'] = 1
        del config['gradient_accumulation_steps']
        del config['train_micro_batch_size_per_gpu']
        del config['gradient_clipping']
        del config['optimizer']
        del config['scheduler']

    # deepspeed features
    if en_param_offload:
        config['zero_optimization']['offload_param'] = {
            "device": "cpu",
            "pin_memory": True
        }

    if en_act_ackpt:
        config['activation_checkpointing'] = {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": True,
            "number_checkpoints": 8,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        }

    if en_sparse_attn:
        config['sparse_attention'] = {
            "mode": "bigbird",
            "block": 16,
            "different_layout_per_head": False,
            "num_global_blocks": 1,
            "num_random_blocks": 3,
            "num_sliding_window_blocks": 3
        }

    # save json file
    tmp_file = tempfile.NamedTemporaryFile().name
    with open(tmp_file, 'w') as fh:
        json.dump(config, fh, indent=2)

    return tmp_file


if __name__ == '__main__':
    import fire
    os.environ["PAGER"] = 'cat'
    fire.Fire(create_json)
