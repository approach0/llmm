export NCCL_BLOCKING_WAIT=1  # Set this variable to use the NCCL backend
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1 # direct access between GPUs? using NVLink or PCI.
# See https://github.com/NVIDIA/nccl/issues/631

#export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_DISTRIBUTED_DEBUG=OFF

#deepspeed --num_gpus 1 ds_test.py ~/llama-models/7B-hgf-new/
#deepspeed --num_gpus 1 ds_test.py ~/llama-models/13B-hgf-new/
#deepspeed --num_gpus 2 ds_test.py ~/llama-models/30B-hgf/
#deepspeed --num_gpus 4 ds_test.py ~/llama-models/65B-hgf-new/


deepspeed \
    --include=localhost:0 \
    --no_local_rank \
    --master_port 8922 \
    ds_test.py \
    \
    --model_name_or_path approach0/mathy-vicuna-13B-FFT \
    --ctx_length 2048 \
    --use_flash_att2 True \
    --load_8bit False \
    --infer_interface gradio \
    --interface_port 8988 \
    \
    --deepspeed $(python ds_config.py \
        --remove_train_args \
        --fp16 False \
        --bf16 True \
        --en_param_offload False \
        --en_act_ackpt False \
        --en_sparse_attn False \
    )
