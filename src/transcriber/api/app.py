import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from transcriber.api.pipeline import Pipeline
from transcriber.services import bootstrap, storage

# поддерживаемые форматы аудио
ALLOWED_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}

# веб-морда
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

pipeline = Pipeline()


@asynccontextmanager
async def lifespan(app):
    # скачиваем модели и грузим один раз на старте
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

    try:
        result = pipeline.run(tmp_path)
    finally:
        os.unlink(tmp_path)

    # шаг 4: раскладка по папкам
    saved = storage.save_result(result, file.filename or "запись")

    result["status"] = "success"
    result["saved_to"] = str(saved)
    return JSONResponse(result)
