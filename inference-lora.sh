export NCCL_BLOCKING_WAIT=1  # Set this variable to use the NCCL backend
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_DISTRIBUTED_DEBUG=OFF

#python inference-lora.py convert \
#    ~/llama-models/7B-hgf-new \
#    ./output/checkpoint-9700/ \
#    ./output/7B-lora-trained

deepspeed \
    --include=localhost:4,5,6,7 \
    --master_port 8921 \
    --no_local_rank \
    inference-lora.py infer \
    ./output/checkpoint-9700/ \
    ./output/7B-lora-trained
