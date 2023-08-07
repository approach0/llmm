conda create -n llmm python=3.10
conda activate llmm

pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu116
conda install -c "nvidia/label/cuda-11.6.2" cuda-toolkit
pip3 install deepspeed==0.10.0

python3 ./test-setup.py

pip3 install transformers==4.30.2
pip3 install sentencepiece
pip3 install peft
conda install -c conda-forge gxx_linux-64=11.2.0
