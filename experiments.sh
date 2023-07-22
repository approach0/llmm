#export CUDA_VISIBLE_DEVICES=2
#python tools/multi-hops.py --mode=cot --pass_name=vicuna --args="[1,'lmsys/vicuna-7b-v1.3']"

export CUDA_VISIBLE_DEVICES=2
python tools/multi-hops.py --mode=ia --pass_name=vicuna --args="[1,'lmsys/vicuna-7b-v1.3']"
