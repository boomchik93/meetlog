# Образ с CUDA + cuDNN: нужен nvcc (сборка llama-cpp) и cudnn (ctranslate2)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# python 3.11 + инструменты сборки + ffmpeg для декодирования аудио
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
        build-essential cmake git ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# делаем python3.11 основным и ставим pip
RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && python -m ensurepip --upgrade \
    && python -m pip install --upgrade pip

WORKDIR /app

# сначала зависимости — слой кэшируется отдельно от кода
COPY requirements.txt .

# llama-cpp-python строго с CUDA, иначе пересказ уйдёт на CPU и не уложится в бюджет
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1
RUN python -m pip install --no-cache-dir -r requirements.txt

# whisper.cpp под GPU: собираем из исходников с CUDA (Pascal-совместим)
ENV GGML_CUDA=1
RUN python -m pip install --no-cache-dir --no-binary pywhispercpp pywhispercpp

COPY . .

EXPOSE 8000
CMD ["python", "run.py"]
