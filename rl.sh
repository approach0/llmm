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
    opts=${4}
    
    set -x
    deepspeed \
        --include=localhost:$devices \
        --master_port $port \
        --no_local_rank \
        rl.py $experiment $opts
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
        # export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export WANDB_RUN_GROUP=baremin_dpo_training
        deepspeed_launch dpo__prm_vs_chatgpt 1,2,3,4,5,6,7 8986 ""
    ;;

    batch_chatgpt_gen_trees)
        detached_rl mcts_explore_fulltopics_using_chatgpt "--run_uid collection --data_offset 0    --data_cutoff 1000"
        detached_rl mcts_explore_fulltopics_using_chatgpt "--run_uid collection --data_offset 1000 --data_cutoff 2000"
        detached_rl mcts_explore_fulltopics_using_chatgpt "--run_uid collection --data_offset 2000 --data_cutoff 3000"
        detached_rl mcts_explore_fulltopics_using_chatgpt "--run_uid collection --data_offset 3000 --data_cutoff 4000"
        detached_rl mcts_explore_fulltopics_using_chatgpt "--run_uid collection --data_offset 4000 --data_cutoff 5000"
        detached_rl mcts_explore_fulltopics_using_chatgpt "--run_uid collection --data_offset 5000 --data_cutoff 6300"
    ;;

    finetune1)
        # export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export WANDB_RUN_GROUP=finetune_mathy_fft_as_querylm
        deepspeed_launch finetune_mathy_fft_as_querylm 4,5,6,7 8989 ""
    ;;

    finetune2)
        # export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export WANDB_RUN_GROUP=finetune_mathy_fft_as_judger
        deepspeed_launch finetune_mathy_fft_as_judger 4,5,6,7 8991 ""
    ;;

    finetune3)
        export WANDB_RUN_GROUP=finetune_generalist_on_final_dataset
        deepspeed_launch finetune_generalist_on_final_dataset 0,1,2,3 8992 ""
esac
