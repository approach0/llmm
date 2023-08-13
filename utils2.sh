export NCCL_BLOCKING_WAIT=1  # Set this variable to use the NCCL backend
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1 # direct access between GPUs? using NVLink or PCI.
# See https://github.com/NVIDIA/nccl/issues/631

#export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_DISTRIBUTED_DEBUG=OFF

export CUDA_VISIBLE_DEVICES=4
python utils2.py test \
    --cache_dir ./data \
    --model_path WizardLM/WizardMath-13B-V1.0 \
    'Solve $x^2 = 4$.'
