export NCCL_BLOCKING_WAIT=1  # Set this variable to use the NCCL backend
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1 # direct access between GPUs? using NVLink or PCI.
# See https://github.com/NVIDIA/nccl/issues/631

#export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_DISTRIBUTED_DEBUG=OFF

copy_for_publish() {
    cp $1/config.json $2
    cp $1/generation_config.json $2
    cp $1/pytorch_model-*.bin $2
    cp $1/pytorch_model.bin.index.json $2

    cp $1/special_tokens_map.json $2
    cp $1/tokenizer_config.json $2
    cp $1/tokenizer.model $2
}
#copy_for_publish output/13B-mathy-FFT ../azbert/ckpt/


MODEL_PATH=$1
CUDA_DEVICES=${2-0}

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
python utils2.py test \
    --cache_dir None \
    --model_path $MODEL_PATH \
    'Solve $x^2 = 4$.'
