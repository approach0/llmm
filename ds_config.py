import os
import sys
import json
import tempfile
import transformers


# https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/trainer#transformers.TrainingArguments
def inject_json(config, train_args):
    # batch sizes
    train_args.gradient_accumulation_steps = config["gradient_accumulation_steps"]
    train_args.per_device_train_batch_size = config['train_micro_batch_size_per_gpu']

    # learning rate
    train_args.learning_rate = config['scheduler']['params']['warmup_max_lr']

    # warm-up steps
    train_args.warmup_steps = config['scheduler']['params']['warmup_num_steps']

    #print(train_args.to_json_string())
    return train_args


def get_script_dir():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def create_json(base_cfg_file='ds_config_zero3.json', world_size=1,
    gradient_accumulation_steps=1, train_micro_batch_size_per_gpu=1,
    learning_rate=1e-5, warmup_steps=100,
    en_param_offload=False, en_act_ackpt=False, en_sparse_attn=False):

    script_dir = get_script_dir()
    base_cfg_file_path = os.path.join(script_dir, base_cfg_file)

    with open(base_cfg_file_path, 'r') as fh:
        config = json.load(fh)

    # batch sizes
    config["gradient_accumulation_steps"] = gradient_accumulation_steps
    config["train_micro_batch_size_per_gpu"] = train_micro_batch_size_per_gpu
    config["train_batch_size"] = (world_size *
        gradient_accumulation_steps * train_micro_batch_size_per_gpu)

    # learning rate
    config['scheduler']['params']['warmup_max_lr'] = learning_rate

    # warm-up steps
    config['scheduler']['params']['warmup_num_steps'] = warmup_steps

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
