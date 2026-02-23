# 改用 Docker Hub 官方 PyTorch 鏡像，同樣內置 CUDA 12.1
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# 設置環境變量
ENV DEBIAN_FRONTEND=noninteractive

# 因為官方 PyTorch 鏡像比較精簡，建議補裝 git 同 ffmpeg
RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

# 預先安裝 ASR 必備庫
RUN pip install --no-cache-dir funasr modelscope


# 設定工作目錄
WORKDIR /app

COPY main.py .

COPY asr ./asr

# 建立掛載點預算
RUN mkdir -p /input_wavs /output_data /root/.cache/modelscope

ENTRYPOINT ["python", "main.py"]