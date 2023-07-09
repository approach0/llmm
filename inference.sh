torchrun \
    --master_addr localhost \
    --master_port 8991 \
    --nproc_per_node 2 \
    --nnodes 1 \
    inference.py \
    --token_path ~/llama-models/7B-hgf-new \
    --model_path ./checkpoints/7B \
    --device=cuda
