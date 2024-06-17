mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && rm -rf ~/miniconda3/miniconda.sh && ~/miniconda3/bin/conda init bash && source ~/.profile
conda create -n hallo python=3.10 && conda activate hallo

nvidia-smi

apt update

apt install git nano tree git-lfs ffmpeg libcublas11 libcufft10 libcudart11.0 libegl1-mesa libegl1 xvfb
apt-get install libegl1-mesa libgles2-mesa libgl1-mesa-dev
export DISPLAY=:99
Xvfb ${DISPLAY} -screen 0 "1024x768x24" -ac +render -noreset -nolisten tcp  &

#### wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update && apt-get -y install cuda-toolkit-12-3
#### wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && dpkg -i cuda-keyring_1.0-1_all.deb && apt-get update && apt-get -y install cuda


git clone https://github.com/ad3002/hallo-webui.git && cd hallo-webui

#### sh ./install.sh

git lfs install
git clone https://huggingface.co/fudan-generative-ai/hallo pretrained_models
wget -O pretrained_models/hallo/net.pth https://huggingface.co/fudan-generative-ai/hallo/resolve/main/hallo/net.pth?download=true

pip install -r requirements.txt 
pip install -e .

echo "Install GPU libraries"

pip install torch==2.2.2+cu121 torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu

export XDG_RUNTIME_DIR=/tmp/runtime-$USER
mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR

# wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
# sh cuda_12.1.0_530.30.02_linux.run
