SETUP=${1}
TOPIC=${2}
FILTER=${3-None}
if [[ -z $SETUP ]]; then
    echo "please specify SETUP."
    exit 1
fi
if [[ -z $TOPIC ]]; then
    echo "please specify TOPIC. (e.g., number_theory, precalculus)"
    exit 1
fi

detached_experiment() {
    ID=$(cat /dev/urandom | tr -cd 'a-f0-9' | head -c 10)
    tmux new-session -c `pwd` -s batch_experiment_$ID -d
    tmux send-keys -t batch_experiment_$ID "$1"
    tmux send-keys -t batch_experiment_$ID Enter
}

case $SETUP in
    test)
    ./experiments.sh mh-chatgpt-2023-03-15 ${TOPIC} ${FILTER}
    ;;

    batch)
    detached_experiment './experiments.sh direct-vicuna-7b algebra'
    detached_experiment './experiments.sh direct-vicuna-7b prealgebra'
    detached_experiment './experiments.sh direct-vicuna-7b intermediate_algebra'

    detached_experiment './experiments.sh ia-vicuna-7b algebra'
    detached_experiment './experiments.sh ia-vicuna-7b prealgebra'
    detached_experiment './experiments.sh ia-vicuna-7b intermediate_algebra'

    #detached_experiment './experiments.sh direct-vicuna-7b precalculus'
    #detached_experiment './experiments.sh cot-vicuna-7b    precalculus'
    #detached_experiment './experiments.sh ia-vicuna-7b     precalculus'
    #detached_experiment './experiments.sh direct-vicuna-13b precalculus'
    #detached_experiment './experiments.sh cot-vicuna-13b    precalculus'
    #detached_experiment './experiments.sh ia-vicuna-13b     precalculus'
    #detached_experiment './experiments.sh direct-vicuna-33b precalculus'
    #detached_experiment './experiments.sh cot-vicuna-33b    precalculus'
    #detached_experiment './experiments.sh ia-vicuna-33b     precalculus'

    #detached_experiment './experiments.sh mh-chatgpt-2023-03-15 precalculus'
    ;;

    direct-vicuna-7b)
    export CUDA_VISIBLE_DEVICES=0
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=direct --run_pass=vicuna --args="[1,'lmsys/vicuna-7b-v1.3']"
    ;;

    cot-vicuna-7b)
    export CUDA_VISIBLE_DEVICES=0
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot --run_pass=vicuna --args="[1,'lmsys/vicuna-7b-v1.3']"
    ;;

    ia-vicuna-7b)
    export CUDA_VISIBLE_DEVICES=1
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=ia --run_pass=vicuna --args="[1,'lmsys/vicuna-7b-v1.3']"
    ;;

    direct-vicuna-13b)
    export CUDA_VISIBLE_DEVICES=2
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=direct --run_pass=vicuna --args="[1,'lmsys/vicuna-13b-v1.3']"
    ;;

    cot-vicuna-13b)
    export CUDA_VISIBLE_DEVICES=3
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot --run_pass=vicuna --args="[1,'lmsys/vicuna-13b-v1.3']"
    ;;

    ia-vicuna-13b)
    export CUDA_VISIBLE_DEVICES=4
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=ia --run_pass=vicuna --args="[1,'lmsys/vicuna-13b-v1.3']"
    ;;

    direct-vicuna-33b)
    export CUDA_VISIBLE_DEVICES=5,7
    export TRANSFORMERS_CACHE='./cache'
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=direct --run_pass=vicuna --args="[2,'lmsys/vicuna-33b-v1.3']"
    ;;

    cot-vicuna-33b)
    export CUDA_VISIBLE_DEVICES=5,7
    export TRANSFORMERS_CACHE='./cache'
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot --run_pass=vicuna --args="[2,'lmsys/vicuna-33b-v1.3']"
    ;;

    ia-vicuna-33b)
    export CUDA_VISIBLE_DEVICES=5,7
    export TRANSFORMERS_CACHE='./cache'
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=ia --run_pass=vicuna --args="[2,'lmsys/vicuna-33b-v1.3']"
    ;;

    direct-chatgpt-2023-03-15)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=direct --run_pass=chatgpt --args="[]"
    ;;

    cot-chatgpt-2023-03-15)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot --run_pass=chatgpt --args="[]"
    ;;

    ia-chatgpt-2023-03-15)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=ia --run_pass=chatgpt --args="[]"
    ;;

    cot-gpt4-2023-july)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot --run_pass=gpt4 --args="[]"
    ;;

    ia-gpt4-2023-july)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=ia --run_pass=gpt4 --args="[]"
    ;;

    example-vicuna-7b)
    export CUDA_VISIBLE_DEVICES=0
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=example2 --run_pass=vicuna --args="[1,'lmsys/vicuna-7b-v1.3']"
    ;;

    example-vicuna-13b)
    export CUDA_VISIBLE_DEVICES=5
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=example2 --run_pass=vicuna --args="[1,'lmsys/vicuna-13b-v1.3']"
    ;;

    example-vicuna-33b)
    export CUDA_VISIBLE_DEVICES=3,7
    export TRANSFORMERS_CACHE='./cache'
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=example2 --run_pass=vicuna --args="[2,'lmsys/vicuna-33b-v1.3']"
    ;;

    example-chatgpt-2023-03-15)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=example2 --run_pass=chatgpt --args="[]"
    ;;

    mh-chatgpt-2023-03-15)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=mh --run_pass=chatgpt --args="[]" --skip_existing True
    ;;
esac
