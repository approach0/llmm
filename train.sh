export NCCL_BLOCKING_WAIT=1  # Set this variable to use the NCCL backend
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1 # direct access between GPUs? using NVLink or PCI.
# See https://github.com/NVIDIA/nccl/issues/631

#export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_DISTRIBUTED_DEBUG=OFF

deepspeed \
    --include=localhost:5,6 \
    --master_port 8921 \
    train.py \
    \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --data_file ./data/finetune-pairs.json \
    --dryrun False \
    --ctx_length 2048 \
    --datamap_nprocs 10 \
    --use_flash_att2 True \
    --load_8bit False \
    --num_train_epochs 3 \
    \
    --output_dir ./output \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 2 \
    --report_to "tensorboard" \
    \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_grad_norm 1.0 \
    --learning_rate 1e-5 \
    --fp16 True \
    --bf16 False \
    --deepspeed $(python ds_config.py \
        --en_param_offload False \
        --en_act_ackpt False \
        --en_sparse_attn True \
    )
