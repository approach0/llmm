SETUP=${1}
TOPIC=${2}
DEVICES=${3-0}
FILTER=${4-None} # 235.json, 439.json
# could be found with larger index: 218.json
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
    tmux send-keys -t $ID "./experiments.sh $1 $2 $3"
    tmux send-keys -t $ID Enter
}

case $SETUP in
    manual)
    #python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
    #    --prompt_mode=manual --run_pass=vicuna --args="[1,'lmsys/vicuna-13b-v1.3']"

    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=manual --run_pass=chatgpt --args="[]"
    ;;

    batch)
    #detached_experiment direct-vicuna-7b precalculus 0
    #detached_experiment cot-vicuna-7b precalculus 1

    #detached_experiment direct-vicuna-13b algebra 2
    #detached_experiment cot-vicuna-13b algebra 3

    #detached_experiment direct-vicuna-13b prealgebra 4
    #detached_experiment cot-vicuna-13b prealgebra 5

    #detached_experiment direct-vicuna-13b intermediate_algebra 6
    #detached_experiment cot-vicuna-13b intermediate_algebra 7

    detached_experiment direct-vicuna-13b precalculus 6
    detached_experiment cot-vicuna-13b precalculus 7
    ;;

    direct-vicuna-7b)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=direct --run_pass=vicuna --args="[1,'lmsys/vicuna-7b-v1.3']"
    ;;

    cot-vicuna-7b)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot --run_pass=vicuna --args="[1,'lmsys/vicuna-7b-v1.3']"
    ;;

    ia-vicuna-7b)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=ia --run_pass=vicuna --args="[1,'lmsys/vicuna-7b-v1.3']"
    ;;

    direct-vicuna-13b)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=direct --run_pass=vicuna --args="[1,'lmsys/vicuna-13b-v1.3']"
    ;;

    cot-vicuna-13b)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot --run_pass=vicuna --args="[1,'lmsys/vicuna-13b-v1.3']"
    ;;

    ia-vicuna-13b)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=ia --run_pass=vicuna --args="[1,'lmsys/vicuna-13b-v1.3']"
    ;;

    direct-vicuna-33b)
    export TRANSFORMERS_CACHE='./cache'
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=direct --run_pass=vicuna --args="[2,'lmsys/vicuna-33b-v1.3']"
    ;;

    cot-vicuna-33b)
    export TRANSFORMERS_CACHE='./cache'
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot --run_pass=vicuna --args="[2,'lmsys/vicuna-33b-v1.3']"
    ;;

    ia-vicuna-33b)
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
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=example2 --run_pass=vicuna --args="[1,'lmsys/vicuna-7b-v1.3']"
    ;;

    example-vicuna-13b)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=example2 --run_pass=vicuna --args="[1,'lmsys/vicuna-13b-v1.3']"
    ;;

    example-vicuna-33b)
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

    mh-vicuna-7b)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=mh --run_pass=vicuna --args="[1,'lmsys/vicuna-7b-v1.3']"
    ;;
esac
