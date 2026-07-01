import asyncio
import json
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from transcriber.api.pipeline import Pipeline
from transcriber.services import bootstrap, storage

ALLOWED_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "OUTPUTS")

pipeline = Pipeline()

# словарь задач: job_id -> {"status", "filename", "submitted_at", "result", "saved_to", "error"}
jobs: dict[str, dict] = {}

# очередь для последовательной обработки (одна задача за раз)
job_queue: asyncio.Queue = asyncio.Queue()


async def worker():
    while True:
        job_id, tmp_path, original_name = await job_queue.get()
        jobs[job_id]["status"] = "processing"
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, pipeline.run, tmp_path
            )
            saved = storage.save_result(result, original_name)
            jobs[job_id]["status"] = "done"
            jobs[job_id]["result"] = result
            jobs[job_id]["saved_to"] = str(saved)
        except Exception as e:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            job_queue.task_done()


@asynccontextmanager
async def lifespan(app):
    bootstrap.ensure_models()
    pipeline.load_models()
    asyncio.create_task(worker())
    yield


app = FastAPI(title="Транскрибатор", lifespan=lifespan)


@app.get("/")
def index():
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

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    job_id = str(uuid.uuid4())
    original_name = file.filename or "запись"
    jobs[job_id] = {
        "status": "queued",
        "filename": original_name,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "result": None,
        "saved_to": None,
        "error": None,
    }
    await job_queue.put((job_id, tmp_path, original_name))

    return JSONResponse({"job_id": job_id, "status": "queued"}, status_code=202)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "задача не найдена")
    resp = {
        "job_id": job_id,
        "status": job["status"],
        "filename": job["filename"],
        "submitted_at": job["submitted_at"],
        "saved_to": job["saved_to"],
        "error": job["error"],
    }
    if job["status"] == "done":
        resp["result"] = job["result"]
    return JSONResponse(resp)


@app.get("/api/history")
def history():
    """Список всех обработанных файлов из папки OUTPUTS."""
    out = Path(OUTPUT_DIR)
    items = []
    if out.exists():
        for json_file in sorted(out.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                stat = json_file.stat()
                rel = str(json_file.relative_to(out))
                # берём краткий summary из файла без загрузки всего
                data = json.loads(json_file.read_text(encoding="utf-8"))
                summary = data.get("summary") or {}
                items.append({
                    "path": rel,
                    "filename": json_file.name,
                    "folder": json_file.parent.name,
                    "size_kb": round(stat.st_size / 1024, 1),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                    "title": summary.get("title") or "",
                    "summary_text": summary.get("summary") or "",
                })
            except Exception:
                pass
    return JSONResponse(items)


@app.get("/api/history/download")
def download_result(path: str):
    """Скачать JSON-файл из OUTPUTS по относительному пути."""
    safe = Path(OUTPUT_DIR) / Path(path)
    # защита от path traversal
    try:
        safe = safe.resolve()
        base = Path(OUTPUT_DIR).resolve()
        safe.relative_to(base)
    except ValueError:
        raise HTTPException(400, "недопустимый путь")
    if not safe.exists() or safe.suffix != ".json":
        raise HTTPException(404, "файл не найден")
    return FileResponse(str(safe), media_type="application/json", filename=safe.name)
