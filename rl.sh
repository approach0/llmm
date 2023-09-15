export NCCL_BLOCKING_WAIT=1  # Set this variable to use the NCCL backend
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1 # direct access between GPUs? using NVLink or PCI.
# See https://github.com/NVIDIA/nccl/issues/631

#export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_DISTRIBUTED_DEBUG=OFF

deepspeed_launch() {
    experiment=$1
    devices=${2-0}
    port=${3-8921}
    
    set -x
    deepspeed \
        --include=localhost:$devices \
        --master_port $port \
        --no_local_rank \
        rl.py $experiment
}

detached_rl() {
    ID="$1-$RANDOM"
    tmux new-session -c `pwd` -s $ID -d
    tmux send-keys -t $ID "conda activate $CONDA_DEFAULT_ENV"
    tmux send-keys -t $ID Enter
    tmux send-keys -t $ID "python rl.py "
    for arg in $@; do
        tmux send-keys -t $ID "\"$arg\" "
    done
    tmux send-keys -t $ID Enter
}

case $1 in
    batch_rl)
        detached_rl inference__chatgpt_prm --run_uid=2023-09-13__01_57_17 \
                --data_offset 1234 --data_cutoff 2000 # 7448 maximum

        detached_rl inference__chatgpt_prm --run_uid=2023-09-13__01_57_17 \
                --data_offset 2000 --data_cutoff 3000 # 7448 maximum

        detached_rl inference__chatgpt_prm --run_uid=2023-09-13__01_57_17 \
                --data_offset 3000 --data_cutoff 4000 # 7448 maximum

        detached_rl inference__chatgpt_prm --run_uid=2023-09-13__01_57_17 \
                --data_offset 4000 --data_cutoff 5000 # 7448 maximum

        detached_rl inference__chatgpt_prm --run_uid=2023-09-13__01_57_17 \
                --data_offset 5000 --data_cutoff 6000 # 7448 maximum

        detached_rl inference__chatgpt_prm --run_uid=2023-09-13__01_57_17 \
                --data_offset 6000 --data_cutoff 7448 # 7448 maximum
    ;;

    dpo1)
        deepspeed_launch finetune_dpo__prm_vs_chatgpt 1,2,3,4,5,6,7
    ;;
esac
