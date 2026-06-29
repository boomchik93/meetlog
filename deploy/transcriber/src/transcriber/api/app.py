import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from transcriber.api.pipeline import Pipeline
from transcriber.services import bootstrap, metrics_db, storage

# поддерживаемые форматы аудио
ALLOWED_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}

# веб-морда
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

pipeline = Pipeline()


@asynccontextmanager
async def lifespan(app):
    # БД метрик/задач + модели
    try:
        metrics_db.init()
    except Exception as error:
        print(f"[metrics] не удалось открыть БД: {error}")
    bootstrap.ensure_models()
    pipeline.load_models()
    yield


app = FastAPI(title="Транскрибатор", lifespan=lifespan)


@app.get("/")
def index():
    # простая веб-страница: загрузка аудио + скачивание разбора
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/health")
def health():
    name, kind = pipeline.recognizer.model_name
    return {
        "status": "ok",
        "whisper": {"model": name, "backend": kind, "ready": pipeline.recognizer.ready},
        "llm_ready": pipeline.summarizer.ready,
        "diarization_ready": pipeline.diarizer.ready,
    }


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"формат {ext} не поддерживается")

    # сохраняем загруженный файл во временный
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    t0 = time.time()
    try:
        result = pipeline.run(tmp_path)
    finally:
        os.unlink(tmp_path)
    proc_sec = round(time.time() - t0, 1)

    # шаг 4: раскладка по папкам
    saved = storage.save_result(result, file.filename or "запись")

    result["status"] = "success"
    result["saved_to"] = str(saved)
    result["proc_sec"] = proc_sec

    # событие задачи в БД (для архива и графиков в админ-панели)
    try:
        audio_sec = max((s.get("end", 0) for s in result.get("segments", [])), default=0)
        metrics_db.insert_task({
            "filename": file.filename, "audio_sec": round(audio_sec, 1),
            "proc_sec": proc_sec, "status": "success",
            "title": (result.get("summary") or {}).get("title"),
            "saved_to": str(saved),
        })
    except Exception as error:
        print(f"[metrics] не записал задачу: {error}")

    return JSONResponse(result)
