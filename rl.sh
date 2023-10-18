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
    set +x
}

detached_rl() {
    ID="$1-$RANDOM"
    echo "new-session: $ID"
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
    sleep 1m
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

    batch_rl_space_explore)
        export CUDA_VISIBLE_DEVICES=0,1
        #export model=./output/merged-extractor-mammoth-13b-highlora
        export run=$(echo $model | sed -e 's-/-_-g' -e 's-\.-_-g')__dense_reward_and_logs
        export WANDB_RUN_GROUP=group_$run

        set -x
        python rl.py rl_generalist_space_explore --run $run --model $model
    ;;

    batch_chatgpt_gen_trees)
        detached_rl mcts_explore_fulltopics_using_chatgpt "--run_uid collection --data_offset 0    --data_cutoff 1000"
        detached_rl mcts_explore_fulltopics_using_chatgpt "--run_uid collection --data_offset 1000 --data_cutoff 2000"
        detached_rl mcts_explore_fulltopics_using_chatgpt "--run_uid collection --data_offset 2000 --data_cutoff 3000"
        detached_rl mcts_explore_fulltopics_using_chatgpt "--run_uid collection --data_offset 3000 --data_cutoff 4000"
        detached_rl mcts_explore_fulltopics_using_chatgpt "--run_uid collection --data_offset 4000 --data_cutoff 5000"
        detached_rl mcts_explore_fulltopics_using_chatgpt "--run_uid collection --data_offset 5000 --data_cutoff 6300"
    ;;

    batch_chatgpt_gen_trees_on_querylm)
        detached_rl mcts_explore_trees_using_chatgpt "--run_uid collection --data_offset 0    --data_cutoff 2000"
        detached_rl mcts_explore_trees_using_chatgpt "--run_uid collection --data_offset 2000 --data_cutoff 4000"
        detached_rl mcts_explore_trees_using_chatgpt "--run_uid collection --data_offset 4000 --data_cutoff 6000"
        detached_rl mcts_explore_trees_using_chatgpt "--run_uid collection --data_offset 6000 --data_cutoff 8000"
        detached_rl mcts_explore_trees_using_chatgpt "--run_uid collection --data_offset 8000 --data_cutoff 10000"
        detached_rl mcts_explore_trees_using_chatgpt "--run_uid collection --data_offset 10000 --data_cutoff 12000"
    ;;

    batch_infer_w_16v100s)
        #experiment=inference_baseline
        #model=TIGER-Lab/MAmmoTH-13B
        #run_uid=13b_mammoth_baseline

        experiment=inference_baseline_using_vllm
        model=WizardLM/WizardMath-13B-V1.0
        run_uid=13b_wizardmath_baseline

        export CUDA_VISIBLE_DEVICES=0
        detached_rl $experiment "--run_uid $run_uid --model $model --tokenizer $model --data_offset  0 --data_cutoff 35"
        export CUDA_VISIBLE_DEVICES=1
        detached_rl $experiment "--run_uid $run_uid --model $model --tokenizer $model --data_offset 35 --data_cutoff 70"
        export CUDA_VISIBLE_DEVICES=2
        detached_rl $experiment "--run_uid $run_uid --model $model --tokenizer $model --data_offset 105 --data_cutoff 140"
        export CUDA_VISIBLE_DEVICES=3
        detached_rl $experiment "--run_uid $run_uid --model $model --tokenizer $model --data_offset 140 --data_cutoff 175"

        export CUDA_VISIBLE_DEVICES=4
        detached_rl $experiment "--run_uid $run_uid --model $model --tokenizer $model --data_offset 175 --data_cutoff 210"
        export CUDA_VISIBLE_DEVICES=5
        detached_rl $experiment "--run_uid $run_uid --model $model --tokenizer $model --data_offset 210 --data_cutoff 245"
        export CUDA_VISIBLE_DEVICES=6
        detached_rl $experiment "--run_uid $run_uid --model $model --tokenizer $model --data_offset 245 --data_cutoff 280"
        export CUDA_VISIBLE_DEVICES=7
        detached_rl $experiment "--run_uid $run_uid --model $model --tokenizer $model --data_offset 280 --data_cutoff 315"

        export CUDA_VISIBLE_DEVICES=8
        detached_rl $experiment "--run_uid $run_uid --model $model --tokenizer $model --data_offset 315 --data_cutoff 350"
        export CUDA_VISIBLE_DEVICES=9
        detached_rl $experiment "--run_uid $run_uid --model $model --tokenizer $model --data_offset 350 --data_cutoff 385"
        export CUDA_VISIBLE_DEVICES=10
        detached_rl $experiment "--run_uid $run_uid --model $model --tokenizer $model --data_offset 385 --data_cutoff 420"
        export CUDA_VISIBLE_DEVICES=11
        detached_rl $experiment "--run_uid $run_uid --model $model --tokenizer $model --data_offset 420 --data_cutoff 455"

        export CUDA_VISIBLE_DEVICES=12
        detached_rl $experiment "--run_uid $run_uid --model $model --tokenizer $model --data_offset 455 --data_cutoff 490"
        export CUDA_VISIBLE_DEVICES=13
        detached_rl $experiment "--run_uid $run_uid --model $model --tokenizer $model --data_offset 490 --data_cutoff 525"
        export CUDA_VISIBLE_DEVICES=14
        detached_rl $experiment "--run_uid $run_uid --model $model --tokenizer $model --data_offset 525 --data_cutoff 560"
        export CUDA_VISIBLE_DEVICES=15
        detached_rl $experiment "--run_uid $run_uid --model $model --tokenizer $model --data_offset 70 --data_cutoff 105"
    ;;

    batch_infer_w_16v100s_all_topics)
        # TEST: export CUDA_VISIBLE_DEVICES=0; python rl.py inference__generalist_using_vllm --topic precalculus --run test $model_args
        #model_args="--model WizardLM/WizardMath-13B-V1.0 --collate_fn collate_cot_wizard"
        #model_args="--model EleutherAI/llemma_7b --collate_fn collate_llemma"
        #model_args="--model meta-math/MetaMath-13B-V1.0 --collate_fn collate_metamath"
        model_args="--model GAIR/GAIRMath-Abel-13b --collate_fn collate_abel"
        #experiment=inference__generalist_using_vllm
        experiment=inference_baseline_using_vllm

        cnt=0
        for topic in 'intermediate_algebra' 'counting_and_probability' 'geometry' 'precalculus' 'prealgebra' 'number_theory' 'algebra'; do
            run_uid=$(echo $model_args | awk '{print $2}' | sed -e 's-/-_-g' -e 's-\.-_-g')__$topic

            export CUDA_VISIBLE_DEVICES=$((cnt+0))
            detached_rl $experiment "--topic $topic --run_uid $run_uid $model_args --data_offset 0"
            export CUDA_VISIBLE_DEVICES=$((cnt+1))
            detached_rl $experiment "--topic $topic --run_uid $run_uid $model_args --data_offset 500"

            ((cnt=cnt+2))
        done
    ;;

    batch_infer_missing_w_16v100s)
        experiment=inference__generalist
        model_args="--model GAIR/GAIRMath-Abel-13b --collate_fn collate_abel"
        run_uid=$(echo $model_args | awk '{print $2}' | sed -e 's-/-_-g' -e 's-\.-_-g')__$topic

        cnt=0
        for i in 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104; do
            target=$(echo $i|sed -e 's/,//g')
            echo $cnt $target
            export CUDA_VISIBLE_DEVICES=$cnt
            detached_rl $experiment "--run_uid $run_uid $model_args --tokenizer $model --data_offset $target"
            (( cnt++ ))
        done
    ;;

    batch_finetune_on_final_dataset)
        set -e
        rm -rf ~/.cache/huggingface/datasets/

        export WANDB_RUN_GROUP=13B-wizardmath-generalist-complete_data
        export model=WizardLM/WizardMath-13B-V1.0
        deepspeed_launch finetune_generalist_13b_lora_4A6000 0,1,2,3 8992 "--model $model --tokenizer $model --run $WANDB_RUN_GROUP"

        export WANDB_RUN_GROUP=13B-mammoth-generalist-complete_data
        export model=TIGER-Lab/MAmmoTH-13B
        deepspeed_launch finetune_generalist_13b_lora_4A6000 0,1,2,3 8992 "--model $model --tokenizer $model --run $WANDB_RUN_GROUP"
    ;;
esac
