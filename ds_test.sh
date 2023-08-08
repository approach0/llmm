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

export CUDA_VISIBLE_DEVICES=6
#deepspeed --num_gpus 1 \
#    ds_test.py \
#    output/alpaca_tokenizer \
#    output/7B-lora-trained

#deepspeed --num_gpus 1 \
#    ds_test.py \
#    output/7B-lora-mser-ckpt \
#    output/7B-lora-mser

#deepspeed --num_gpus 2 \
#    ds_test.py \
#    output/checkpoint-6500 \
#    output/13B-lora-trained-2ep

#deepspeed --num_gpus 1 \
#    ds_test.py \
#    lmsys/vicuna-13b-v1.5 \
#    lmsys/vicuna-13b-v1.5

deepspeed --num_gpus 1 \
    --no_local_rank \
    ds_test.py \
    \
    --model_name_or_path lmsys/vicuna-13b-v1.5-16k \
    --ctx_length 2048 \
    --use_flash_att2 True \
    --load_8bit False \
    \
    --deepspeed $(python ds_config.py \
        --remove_train_args \
        --fp16 False \
        --bf16 True \
        --en_param_offload True \
        --en_act_ackpt False \
        --en_sparse_attn True \
    )
