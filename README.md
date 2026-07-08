# Транскрибатор

Локальный сервис для обработки аудиозаписей разговоров. На вход — аудиофайл, на выход — транскрибация с разметкой спикеров и структурированный разбор: темы, решения, задачи, риски.

Всё работает локально. Данные наружу не отправляются.

## Стек

- **Транскрибация:** Whisper (faster-whisper / whisper.cpp)
- **Диаризация:** resemblyzer
- **Суммаризация:** Qwen2.5-7B-Instruct (GGUF, llama.cpp)
- **Сервер:** FastAPI + uvicorn
- **Язык:** Python 3.11
- **Деплой:** Docker + NVIDIA Container Toolkit

## Пайплайн

1. Загрузка аудио, ресэмпл до 16 кГц, фильтрация телефонных каналов
2. Транскрибация (VAD-нарезка + Whisper)
3. Определение спикеров (стерео — по каналам, моно — по голосовому отпечатку)
4. Суммаризация (LLM выделяет темы, решения, задачи, риски)
5. Сохранение результата в `OUTPUTS/`

## Структура

```
├── run.py                          # точка входа
├── config/config.yml               # конфигурация
├── Dockerfile / docker-compose.yml
├── src/transcriber/
│   ├── config/settings.py          # чтение конфига, распределение ресурсов
│   ├── core/
│   │   ├── audio.py                # загрузка и обработка звука
│   │   ├── transcriber.py          # распознавание речи
│   │   ├── whisper_worker.py       # изолированный процесс whisper.cpp
│   │   └── diarizer.py             # определение спикеров
│   ├── services/
│   │   ├── summarizer.py           # суммаризация
│   │   ├── bootstrap.py            # загрузка моделей при первом запуске
│   │   └── storage.py              # сохранение результатов
│   └── api/
│       ├── app.py                  # FastAPI-сервер
│       └── pipeline.py             # оркестратор пайплайна
├── deploy/                         # скрипты деплоя
└── admin/                          # админ-панель (мониторинг сервера)
```

## Запуск (Docker)

Требования: Docker, nvidia-container-toolkit, GPU с ~11 ГБ VRAM.

```bash
docker compose up --build
```

При первом запуске скачаются модели (~5-6 ГБ). Сервер поднимется на `http://localhost:8000`.

## Запуск (без Docker)

```bash
pip install -r requirements.txt
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
python run.py
```

## API

```bash
# обработка файла
curl -F file=@запись.mp3 http://localhost:8000/api/transcribe > result.json

# статус сервера
curl http://localhost:8000/api/health
```

Форматы: mp3, wav, m4a, ogg, flac, webm.

## Конфигурация

`config/config.yml` — описание железа, выбор моделей, ручные переопределения. Программа автоматически распределяет ресурсы (GPU-слои, контекст, потоки) по указанным параметрам.

```yaml
hardware:
  device: "auto"      # auto / cuda / mps / cpu
  vram_gb: 11
  ram_gb: 32
  cpu_threads: "auto"
```

## Формат ответа

```json
{
  "status": "success",
  "language": "ru",
  "text": "полный текст",
  "segments": [
    { "start": 0.0, "end": 3.5, "text": "Добрый день", "speaker": "SPEAKER_00" }
  ],
  "speakers": { "SPEAKER_00": "текст первого", "SPEAKER_01": "текст второго" },
  "summary": {
    "title": "Тема обсуждения",
    "summary": "Краткое описание",
    "topics": [],
    "decisions": [],
    "action_items": [],
    "risks": []
  }
}
```

## Лицензия

MIT — см. файл [LICENSE](LICENSE).
