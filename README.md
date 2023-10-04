# LLMM (Large Language Model for Math)

* HF LLaMA: https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama
* Annotated BERT: https://github.com/w32zhong/annotated-bert

## Usage
```sh
python inference.py ~/llama-models/7B-hgf-new/ --debug=False
```
```txt
Creating model ...
Loading model shard: pytorch_model-00002-of-00002.bin
Loading model shard: pytorch_model-00001-of-00002.bin
Prompt: My name is Mariama, my favorite
2016 film is La La Land and my favorite food is chocolate chip cookies. I love being active and
am always looking for new things to do around Chicago. I am currently a junior majoring in
Communication with a focus in Strategic Communication and a minor in Spanish. After graduation,
I plan to move to a city with a good public transportation system, get a job and enjoy life. I
am so excited to be a part of the Communication Interns this summer and look forward to learning
about the industry and developing skills that will help me in the future.
```

## Wandb
```sh
wandb login
```

## Setup
```sh
conda create --name llmm -c conda-forge python=3.8
conda activate llmm

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
python -c 'import torch; print(torch.cuda.is_available())'

conda install cuda -c nvidia/label/cuda-11.8.0 # must match torch version!
pip3 install packaging
pip3 install flash-attn==2.3.0

pip3 install transformers==4.31.0
pip3 install deepspeed==0.10.3
pip3 install peft==0.4.0

pip3 install -r requirements.txt
```

## Slurm
See instructions: https://watgpu.cs.uwaterloo.ca/slurm.html, or https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm

To see the time limit for a job:
```sh
squeue
scontrol show job -dd 483 | grep TimeLimit
scontrol show job -dd 479 | grep TRES=
salloc --account=rrg-jimmylin --gpus-per-node=v100l:1 --cpus-per-task=2 --time=0-01:10 --mem=80G
```
