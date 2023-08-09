export NCCL_BLOCKING_WAIT=1  # Set this variable to use the NCCL backend
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_DISTRIBUTED_DEBUG=OFF
export CUDA_LAUNCH_BLOCKING=1

#python inference-lora.py convert \
#    ~/llama-models/7B-hgf-new \
#    ./output/7B-lora-mser-ckpt \
#    ./output/7B-lora-mser

# test the original model w/o instruct fine-tuning
#python inference-lora.py infer \
#    ~/llama-models/7B-hgf-new \
#    ~/llama-models/7B-hgf-new

# test the LoRA-trained model w/ instruct fine-tuning
#python inference-lora.py infer \
#    ./output/checkpoint-9700/ \
#    ./output/7B-lora-trained

#deepspeed \
#    --no_local_rank \
#    --include=localhost:4,5 \
#    --master_port 8921 \
#    inference-lora.py infer \
#    ~/llama-models/7B-hgf-new \
#    ~/llama-models/7B-hgf-new

#python inference-lora.py quantize \
#    ./output/checkpoint-6500/ \
#    ./output/7B-lora-trained \
#    ./output/7B-lora-gptq

python inference-lora.py convert \
    lmsys/vicuna-13b-v1.5-16k \
    ./output/runs/Aug08_04-44-32_ckpt \
    ./output/runs/Aug08_04-44-32_merged
