export NCCL_BLOCKING_WAIT=1  # Set this variable to use the NCCL backend
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_DISTRIBUTED_DEBUG=OFF

(cd BMTrain && python3 setup.py build)
export PYTHONPATH=`pwd`/BMTrain/build/lib.linux-x86_64-cpython-310/bmtrain

torchrun \
    --master_addr localhost \
    --master_port 8991 \
    --nproc_per_node 4 \
    --nnodes 1 \
    inference.py \
    --token_path ~/llama-models/7B-hgf-new \
    --model_path ./checkpoints/7B
