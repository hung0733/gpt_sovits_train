FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# 設置環境變量，避免安裝期間出現互動提示
ENV DEBIAN_FRONTEND=noninteractive

# 安裝 Python 同埋系統依賴
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 確保 python 指令指向 python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# 升級 pip
RUN pip install --no-cache-dir --upgrade pip

# 安裝 PyTorch (支援 CUDA 12.4)
RUN pip install --no-cache-dir torch torchvision torchaudio

# 安裝 UVR5 相關套件及 ONNX Runtime
RUN pip install --no-cache-dir \
    librosa \
    soundfile \
    ffmpeg-python \
    pyyaml \
    tqdm \
    onnxruntime-gpu \
    rotary-embedding-torch \
    ml-collections

WORKDIR /app

COPY . .

ENTRYPOINT ["python", "main.py"]