FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir librosa soundfile ffmpeg-python pyyaml tqdm onnxruntime-gpu

# 喺原本嘅 pip install 加入以下套件
RUN pip install rotary-embedding-torch ml-collections

WORKDIR /app

COPY . .

ENTRYPOINT ["python", "main.py"]