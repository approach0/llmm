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
	--skip_existing False
    ;;

    manual)
    #python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
    #    --prompt_mode=manual --run_pass=vicuna --args="[1,'lmsys/vicuna-13b-v1.3']"

    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=manual --run_pass=chatgpt --args="[]" --skip_existing True
    ;;


    13B-mathy)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=cot --run_pass=utils --skip_existing True \
        --args="['cuda', 'output/13B-mathy', 'lmsys/vicuna-13b-v1.5', 'output/13B-mathy']"

    ;;

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
        --prompt_mode=mh --run_pass=chatgpt --args="[]" --skip_existing True --metric $METRIC
    ;;

    mh-vicuna-7b)
    python tools/multi-hops.py --logname $SETUP --topic $TOPIC --fname_filter $FILTER \
        --prompt_mode=mh --run_pass=vicuna --args="[1,'lmsys/vicuna-7b-v1.3']" \
        --metric $METRIC
    ;;
esac
