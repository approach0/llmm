SETUP=${1}
if [[ -z $SETUP ]]; then
    echo "please specify SETUP."
fi

case $SETUP in
    cot-vicuna-7b)
    export CUDA_VISIBLE_DEVICES=2
    python tools/multi-hops.py --logname $SETUP \
        --prompt_mode=cot --run_pass=vicuna --args="[1,'lmsys/vicuna-7b-v1.3']"
    ;;

    ia-vicuna-7b)
    export CUDA_VISIBLE_DEVICES=2
    python tools/multi-hops.py --logname $SETUP \
        --prompt_mode=ia --run_pass=vicuna --args="[1,'lmsys/vicuna-7b-v1.3']"
    ;;

    cot-vicuna-13b)
    export CUDA_VISIBLE_DEVICES=4
    python tools/multi-hops.py --logname $SETUP \
        --prompt_mode=cot --run_pass=vicuna --args="[1,'lmsys/vicuna-13b-v1.3']"
    ;;

    ia-vicuna-13b)
    export CUDA_VISIBLE_DEVICES=5
    python tools/multi-hops.py --logname $SETUP \
        --prompt_mode=ia --run_pass=vicuna --args="[1,'lmsys/vicuna-13b-v1.3']"
    ;;


    cot-vicuna-33b)
    export CUDA_VISIBLE_DEVICES=2,4
    export TRANSFORMERS_CACHE='./cache'
    python tools/multi-hops.py --logname $SETUP \
        --prompt_mode=cot --run_pass=vicuna --args="[2,'lmsys/vicuna-33b-v1.3']"
    ;;

    ia-vicuna-33b)
    export CUDA_VISIBLE_DEVICES=3,7
    export TRANSFORMERS_CACHE='./cache'
    python tools/multi-hops.py --logname $SETUP \
        --prompt_mode=ia --run_pass=vicuna --args="[2,'lmsys/vicuna-33b-v1.3']"
    ;;

    cot-chatgpt-2022-june)
    python tools/multi-hops.py --logname $SETUP \
        --prompt_mode=cot --run_pass=chatgpt --args="[]"
    ;;

    ia-chatgpt-2022-june)
    python tools/multi-hops.py --logname $SETUP \
        --prompt_mode=ia --run_pass=chatgpt --args="[]"
    ;;
esac
