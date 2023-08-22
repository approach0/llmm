export NCCL_BLOCKING_WAIT=1  # Set this variable to use the NCCL backend
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1 # direct access between GPUs? using NVLink or PCI.
# See https://github.com/NVIDIA/nccl/issues/631

#export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_DISTRIBUTED_DEBUG=OFF

#--data_file ./data/finetune-pairs.json \

deepspeed \
    --include=localhost:0 \
    --master_port 8921 \
    train.py \
    \
    --model_name_or_path approach0/mathy-vicuna-13B-FFT \
    --data_file approach0/MATH-picky-test \
    --debug_single_layer False \
    --dryrun False \
    --ctx_length 1500 \
    --datamap_nprocs 10 \
    --use_flash_att2 True \
    --load_8bit False \
    --num_train_epochs 2 \
    \
    --output_dir ./output \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --report_to "tensorboard" \
    \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 12 \
    --max_grad_norm 1.0 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --fp16 False \
    --bf16 True \
    --deepspeed $(python ds_config.py \
        --en_param_offload False \
        --en_act_ackpt False \
        --en_sparse_attn False \
    )
