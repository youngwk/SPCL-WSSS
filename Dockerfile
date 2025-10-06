FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    python3.6 python3.6-dev python3-pip \
    git build-essential cmake \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN python3.6 -m pip install --upgrade pip setuptools wheel

RUN python3.6 -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt /tmp/
RUN python3.6 -m pip install -r /tmp/requirements.txt

WORKDIR /workspace

