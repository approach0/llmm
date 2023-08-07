conda create -y -n llmm python=3.10
conda activate llmm

pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118
pip3 install deepspeed==0.10.0
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit

python3 ./test-setup.py

pip3 install transformers
pip3 install sentencepiece
pip3 install peft
pip install -r pya0/requirements.txt
