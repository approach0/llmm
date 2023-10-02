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

## Tensorboard
```sh
tensorboard dev upload --logdir ./output/runs/Jul12_11-00-08_watgpu-100/
```

## Setup
```sh
conda create --name llmm -c conda-forge python=3.8
conda activate llmm

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
python -c 'import torch; print(torch.cuda.is_available())'

conda install cuda -c nvidia/label/cuda-11.8.0 # must match torch version!
pip3 install packaging
pip3 install flash-attn

pip3 install transformers==4.28.1
pip3 install deepspeed==0.10.3
pip3 install git+https://github.com/huggingface/peft
```
