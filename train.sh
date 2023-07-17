export NCCL_BLOCKING_WAIT=1  # Set this variable to use the NCCL backend
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1 # direct access between GPUs? using NVLink or PCI.
# See https://github.com/NVIDIA/nccl/issues/631

#export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_DISTRIBUTED_DEBUG=OFF

deepspeed \
    --include=localhost:4,5,6,7 \
    --master_port 8921 \
    train.py \
    --output_dir ./output \
    --num_train_epochs 3 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --bf16_full_eval True \
    --logging_steps 2 \
    --warmup_steps 3 \
    --report_to "tensorboard" \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 2 \
    --deepspeed ds_config_zero3.json \
    --model_name_or_path ~/llama-models/7B-hgf-new/
