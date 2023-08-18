SETUP=${1}
TOPIC=${2}
DEVICES=${3-0}
FILTER=${4-None} # 235.json, 439.json
# could be found with larger index: 218.json

METRIC=maj@10

if [[ -z $SETUP ]]; then
    echo "please specify SETUP."
    exit 1
fi
if [[ -z $TOPIC ]]; then
    echo "please specify TOPIC. (e.g., number_theory, precalculus)"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${DEVICES}

detached_experiment() {
    ID=${1}-${2}
    tmux new-session -c `pwd` -s $ID -d
    tmux send-keys -t $ID "conda activate $CONDA_DEFAULT_ENV"
    tmux send-keys -t $ID Enter
    tmux send-keys -t $ID "./experiments.sh $1 $2 $3"
    tmux send-keys -t $ID Enter
}

case $SETUP in
    batch)
    detached_experiment askkey_chatgpt precalculus 0
    detached_experiment askkey_td003 precalculus 0
    detached_experiment askkey_chatgpt algebra 0
    detached_experiment askkey_td003 algebra 0
    detached_experiment askkey_chatgpt prealgebra 0
    detached_experiment askkey_td003 prealgebra 0
    detached_experiment askkey_chatgpt intermediate_algebra 0
    detached_experiment askkey_td003 intermediate_algebra 0
    ;;

    batch2)
    #detached_experiment cot-vicuna-7b-w_groundtruth_query precalculus 7
    #detached_experiment ia-vicuna-7b-w_groundtruth_query precalculus 7

    #detached_experiment cot-vicuna-13b-w_groundtruth_query precalculus 5
    #detached_experiment ia-vicuna-13b-w_groundtruth_query precalculus 6

    #detached_experiment cot-vicuna-33b-w_groundtruth_query precalculus 3,4
    #detached_experiment ia-vicuna-33b-w_groundtruth_query precalculus 5,6

    detached_experiment cot-chatgpt-2023-03-15-w_groundtruth_query precalculus 0
    detached_experiment ia-chatgpt-2023-03-15-w_groundtruth_query precalculus 0
    ;;

    batch3)
    detached_experiment mh_chatgpt_online_pass3 precalculus 0
    detached_experiment mh_chatgpt_MATH_pass3 precalculus 0
    detached_experiment mh_chatgpt_a0_pass3 precalculus 0
    detached_experiment mh_chatgpt_mabowdor_pass3 precalculus 0
    ;;

    batch4)
    #detached_experiment 33B-mathy-lora precalculus 0
    detached_experiment 13B-mathy-fft-maj5 precalculus 6
    detached_experiment 13B-wizard-math-maj5 precalculus 7
    ;;

    mh_chatgpt_online_pass3)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=mh --run_pass=chatgpt --args="[]" --skip_existing True \
        --metric pass@3 --search_tool=online
    ;;

    mh_chatgpt_MATH_pass3)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=mh --run_pass=chatgpt --args="[]" --skip_existing True \
        --metric pass@3 --search_tool=MATH
    ;;

    mh_chatgpt_a0_pass3)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=mh --run_pass=chatgpt --args="[]" --skip_existing True \
        --metric pass@3 --search_tool=a0
    ;;

    mh_chatgpt_mabowdor_pass3)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=mh --run_pass=chatgpt --args="[]" --skip_existing True \
        --metric pass@3 --search_tool=mabowdor
    ;;

    askkey_chatgpt)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=askkey --run_pass=chatgpt --args="[]" --skip_existing False \
        --output_marking False
    ;;

    askkey_td003)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=askkey --run_pass=td003 --args="[]" --skip_existing False \
        --output_marking False
    ;;

    ds_infer)
    python tools/multi-hops.py --logname mathy --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot --run_pass=ds --args='http://127.0.0.1:8988/generate' \
        --skip_existing True
    ;;

    manual)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=manual --run_pass=chatgpt --args="[]" --skip_existing False \
        --search_tool=online
    ;;

    manual-picky)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=manual-picky --run_pass=chatgpt --args="[]" --skip_existing True \
        --search_tool=mabowdor --train_or_test train
    ;;

    ia-chatgpt-315-train-pass3)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=ia --run_pass=chatgpt --args="[]" --skip_existing True \
        --search_tool=mabowdor --train_or_test train --metric maj@3
    ;;

    33B-mathy-lora)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot --run_pass=utils --skip_existing True \
        --args="['greedy', 2048, 'output/runs/Aug10_22-57-43_GCRAZGDL1578_ckpt/', 'lmsys/vicuna-33b-v1.3', 'output/runs/Aug10_22-57-43_GCRAZGDL1578_ckpt/']"

    ;;

    13B-mathy-lora)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot --run_pass=utils --skip_existing True \
        --args="['greedy', 4096, 'output/13B-mathy', 'lmsys/vicuna-13b-v1.5', 'output/13B-mathy']"

    ;;

    13B-mathy-fft)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot --run_pass=utils --skip_existing True \
        --args="['greedy', 4096, 'approach0/mathy-vicuna-13B-FFT', 'approach0/mathy-vicuna-13B-FFT', {'cache_dir': './data'}]"
    ;;

    13B-mathy-fft-goodprompt)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot_mytrain --run_pass=utils --skip_existing True \
        --args="['greedy', 4096, 'approach0/mathy-vicuna-13B-FFT', 'approach0/mathy-vicuna-13B-FFT', {'cache_dir': './data'}]"
    ;;

    13B-mathy-fft-maj5)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot_mytrain --run_pass=utils --skip_existing True \
        --metric maj@5 \
        --args="['sample', 4096, 'approach0/mathy-vicuna-13B-FFT', 'approach0/mathy-vicuna-13B-FFT', {'cache_dir': './data'}]"
    ;;

    13B-wizard-math)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot --run_pass=utils --skip_existing True \
        --args="['greedy', 4096, 'WizardLM/WizardMath-13B-V1.0', 'WizardLM/WizardMath-13B-V1.0', {'cache_dir': './data'}]"
    ;;

    13B-wizard-math-goodprompt)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot_wizard --run_pass=utils --skip_existing True \
        --args="['greedy', 4096, 'WizardLM/WizardMath-13B-V1.0', 'WizardLM/WizardMath-13B-V1.0', {'cache_dir': './data'}]"
    ;;

    13B-wizard-math-maj5)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot_wizard --run_pass=utils --skip_existing True \
        --metric maj@5 \
        --args="['sample', 4096, 'WizardLM/WizardMath-13B-V1.0', 'WizardLM/WizardMath-13B-V1.0', {'cache_dir': './data'}]"
    ;;

    batch_mathy_wizard)
    detached_experiment 13B-wizard-math-maj10 precalculus 6
    detached_experiment 13B-mathy-fft-maj10 precalculus 7
    ;;

    direct-vicuna-7b)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=direct --run_pass=vicuna --args="[1,'lmsys/vicuna-7b-v1.3']" \
        --metric $METRIC
    ;;

    cot-vicuna-7b)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot --run_pass=vicuna --args="[1,'lmsys/vicuna-7b-v1.3']" \
        --metric $METRIC
    ;;

    cot-vicuna-7b-w_groundtruth_query)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
    --prompt_mode=cot --run_pass=vicuna --args="[1,'lmsys/vicuna-7b-v1.3']" \
    --metric $METRIC --ground_truth_dir=./output/datasets/MATH/test/precalculus/run__manual/
    ;;

    ia-vicuna-7b)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=ia --run_pass=vicuna --args="[1,'lmsys/vicuna-7b-v1.3']" \
        --metric $METRIC
    ;;

    ia-vicuna-7b-w_groundtruth_query)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
    --prompt_mode=ia --run_pass=vicuna --args="[1,'lmsys/vicuna-7b-v1.3']" \
    --metric $METRIC --ground_truth_dir=./output/datasets/MATH/test/precalculus/run__manual/
    ;;

    direct-vicuna-13b)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=direct --run_pass=vicuna --args="[1,'lmsys/vicuna-13b-v1.3']" \
        --metric $METRIC
    ;;

    cot-vicuna-13b)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot --run_pass=vicuna --args="[1,'lmsys/vicuna-13b-v1.3']" \
        --metric $METRIC
    ;;

    cot-vicuna-13b-w_groundtruth_query)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
    --prompt_mode=cot --run_pass=vicuna --args="[1,'lmsys/vicuna-13b-v1.3']" \
    --metric $METRIC --ground_truth_dir=./output/datasets/MATH/test/precalculus/run__manual/
    ;;

    ia-vicuna-13b)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=ia --run_pass=vicuna --args="[1,'lmsys/vicuna-13b-v1.3']" \
        --metric $METRIC
    ;;

    ia-vicuna-13b-w_groundtruth_query)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
    --prompt_mode=ia --run_pass=vicuna --args="[1,'lmsys/vicuna-13b-v1.3']" \
    --metric $METRIC --ground_truth_dir=./output/datasets/MATH/test/precalculus/run__manual/
    ;;

    direct-vicuna-33b)
    export TRANSFORMERS_CACHE='./cache'
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=direct --run_pass=vicuna --args="[2,'lmsys/vicuna-33b-v1.3']" \
        --metric $METRIC
    ;;

    cot-vicuna-33b)
    export TRANSFORMERS_CACHE='./cache'
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot --run_pass=vicuna --args="[2,'lmsys/vicuna-33b-v1.3']" \
        --metric $METRIC
    ;;

    cot-vicuna-33b-w_groundtruth_query)
    export TRANSFORMERS_CACHE='./cache'
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
    --prompt_mode=cot --run_pass=vicuna --args="[2,'lmsys/vicuna-33b-v1.3']" \
    --metric $METRIC --ground_truth_dir=./output/datasets/MATH/test/precalculus/run__manual/
    ;;

    ia-vicuna-33b)
    export TRANSFORMERS_CACHE='./cache'
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=ia --run_pass=vicuna --args="[2,'lmsys/vicuna-33b-v1.3']" \
        --metric $METRIC
    ;;

    ia-vicuna-33b-w_groundtruth_query)
    export TRANSFORMERS_CACHE='./cache'
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
    --prompt_mode=ia --run_pass=vicuna --args="[2,'lmsys/vicuna-33b-v1.3']" \
    --metric $METRIC --ground_truth_dir=./output/datasets/MATH/test/precalculus/run__manual/
    ;;

    direct-chatgpt-2023-03-15)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=direct --run_pass=chatgpt --args="[]" --metric $METRIC
    ;;

    cot-chatgpt-2023-03-15)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot --run_pass=chatgpt --args="[]" --metric $METRIC
    ;;

    cot-chatgpt-2023-03-15-w_groundtruth_query)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot --run_pass=chatgpt --args="[]" --metric $METRIC \
        --ground_truth_dir=./output/datasets/MATH/test/precalculus/run__manual/
    ;;

    ia-chatgpt-2023-03-15)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=ia --run_pass=chatgpt --args="[]" --metric $METRIC
    ;;

    ia-chatgpt-2023-03-15-w_groundtruth_query)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=ia --run_pass=chatgpt --args="[]" --metric $METRIC \
        --ground_truth_dir=./output/datasets/MATH/test/precalculus/run__manual/
    ;;

    cot-gpt4-2023-july)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot --run_pass=gpt4 --args="[]" --metric $METRIC
    ;;

    ia-gpt4-2023-july)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=ia --run_pass=gpt4 --args="[]" --metric $METRIC
    ;;

    example-vicuna-7b)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=example2 --run_pass=vicuna --args="[1,'lmsys/vicuna-7b-v1.3']" \
        --metric $METRIC
    ;;

    example-chatgpt-2023-03-15)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=example2 --run_pass=chatgpt --args="[]" --metric $METRIC
    ;;

    mh-chatgpt-2023-03-15)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=mh --run_pass=chatgpt --args="[]" --skip_existing False \
        --metric $METRIC
    ;;

    mh-vicuna-7b)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=mh --run_pass=vicuna --args="[1,'lmsys/vicuna-7b-v1.3']" \
        --metric $METRIC
    ;;
esac
