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
    tmux send-keys -t $ID "export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    tmux send-keys -t $ID Enter
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
        export WANDB_RUN_GROUP=watgpu-wizard
        deepspeed_launch finetune_generalist_on_wizard 0,1,2,3 8993 "--run watgpu-wizard"
    ;;

    finetune4)
        export WANDB_RUN_GROUP=GCR-try2
        deepspeed_launch finetune_generalist_on_final_dataset 0 8992 "--run GCR-try2"
    ;;

    batch_infer_generalist)
        detached_rl inference__generalist "--run_uid collection --data_offset 0   --data_cutoff 100"
        detached_rl inference__generalist "--run_uid collection --data_offset 200 --data_cutoff 300"
        detached_rl inference__generalist "--run_uid collection --data_offset 300 --data_cutoff 400"
        detached_rl inference__generalist "--run_uid collection --data_offset 400 --data_cutoff 500"
        detached_rl inference__generalist "--run_uid collection --data_offset 500 --data_cutoff 550"
    ;;

    batch_infer_mammoth)
        detached_rl inference__7b_mammoth "--run_uid collection --data_offset 0   --data_cutoff 100"
        detached_rl inference__7b_mammoth "--run_uid collection --data_offset 200 --data_cutoff 300"
        detached_rl inference__7b_mammoth "--run_uid collection --data_offset 300 --data_cutoff 400"
        detached_rl inference__7b_mammoth "--run_uid collection --data_offset 400 --data_cutoff 500"
        detached_rl inference__7b_mammoth "--run_uid collection --data_offset 500 --data_cutoff 550"
    ;;

    batch_infer_generalist_w_4gpus_and_specified_model)
        #model=output/finetune_generalist_on_wizard-small-traindata/watgpu-wizard
        #run_uid=wizard_ra

        model=output/finetune_generalist_on_mammoth-small-traindata/watgpu
        run_uid=mammoth_ra

        export CUDA_VISIBLE_DEVICES=0
        detached_rl inference__generalist "--model $model --run_uid $run_uid --data_offset  0 --data_cutoff 35"
        detached_rl inference__generalist "--model $model --run_uid $run_uid --data_offset 35 --data_cutoff 70"
        detached_rl inference__generalist "--model $model --run_uid $run_uid --data_offset 105 --data_cutoff 140"
        detached_rl inference__generalist "--model $model --run_uid $run_uid --data_offset 140 --data_cutoff 175"

        export CUDA_VISIBLE_DEVICES=1
        detached_rl inference__generalist "--model $model --run_uid $run_uid --data_offset 175 --data_cutoff 210"
        detached_rl inference__generalist "--model $model --run_uid $run_uid --data_offset 210 --data_cutoff 245"
        detached_rl inference__generalist "--model $model --run_uid $run_uid --data_offset 245 --data_cutoff 280"
        detached_rl inference__generalist "--model $model --run_uid $run_uid --data_offset 280 --data_cutoff 315"

        export CUDA_VISIBLE_DEVICES=2
        detached_rl inference__generalist "--model $model --run_uid $run_uid --data_offset 315 --data_cutoff 350"
        detached_rl inference__generalist "--model $model --run_uid $run_uid --data_offset 350 --data_cutoff 385"
        detached_rl inference__generalist "--model $model --run_uid $run_uid --data_offset 385 --data_cutoff 420"
        detached_rl inference__generalist "--model $model --run_uid $run_uid --data_offset 420 --data_cutoff 455"

        export CUDA_VISIBLE_DEVICES=3
        detached_rl inference__generalist "--model $model --run_uid $run_uid --data_offset 455 --data_cutoff 490"
        detached_rl inference__generalist "--model $model --run_uid $run_uid --data_offset 490 --data_cutoff 525"
        detached_rl inference__generalist "--model $model --run_uid $run_uid --data_offset 525 --data_cutoff 560"
    ;;
esac
