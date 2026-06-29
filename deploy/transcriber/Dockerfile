# CPU-сборка: без CUDA, лёгкий образ
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV DEBIAN_FRONTEND=noninteractive

# ffmpeg для декодирования аудио (mp3/m4a/ogg/...), build-essential на всякий случай
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch ставим CPU-версией отдельно (меньше и без CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "run.py"]
