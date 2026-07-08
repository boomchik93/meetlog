import asyncio
import json
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from transcriber.api.pipeline import Pipeline
from transcriber.services import bootstrap, storage
from transcriber.services.jobstore import store

ALLOWED_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "OUTPUTS")

pipeline = Pipeline()

# очередь для последовательной обработки (одна задача за раз).
# сама очередь в памяти, но каждая задача продублирована в БД (store):
# при перезапуске незавершённые задачи восстанавливаются из БД в эту очередь.
job_queue: asyncio.Queue = asyncio.Queue()


def _log(message, tag=None, job_id=None, level="info"):
    """Пишем и в консоль (как раньше), и в БД-журнал."""
    print(message)
    try:
        store.log(message, level=level, tag=tag, job_id=job_id)
    except Exception:
        pass


async def worker():
    while True:
        job_id, upload_path, original_name = await job_queue.get()
        store.mark_processing(job_id)
        _log(f"[очередь] взял в работу: {original_name}", tag="очередь", job_id=job_id)
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, pipeline.run, upload_path
            )
            saved = storage.save_result(result, original_name)
            store.mark_done(job_id, result, saved)
            _log(f"[очередь] готово: {saved}", tag="очередь", job_id=job_id)
        except Exception as e:
            store.mark_error(job_id, str(e))
            _log(f"[очередь] ошибка: {e}", tag="очередь", job_id=job_id, level="error")
        finally:
            # исходник больше не нужен — удаляем сохранённую загрузку
            try:
                os.unlink(upload_path)
            except OSError:
                pass
            job_queue.task_done()


def _recover_unfinished():
    """При старте возвращаем в очередь задачи, прерванные перезапуском."""
    pending = store.unfinished_jobs()
    if not pending:
        return
    recovered, dropped = 0, 0
    for job in pending:
        upload = job.get("upload_path")
        if upload and os.path.exists(upload):
            store.requeue(job["id"])
            job_queue.put_nowait((job["id"], upload, job["filename"]))
            recovered += 1
        else:
            # исходник не сохранился (старые задачи до персистентности) —
            # не теряем молча, помечаем ошибкой, чтобы было видно в истории
            store.mark_error(job["id"], "исходный файл недоступен после перезапуска")
            dropped += 1
    _log(
        f"[восстановление] возвращено в очередь: {recovered}, "
        f"не удалось восстановить: {dropped}",
        tag="восстановление",
    )


@asynccontextmanager
async def lifespan(app):
    bootstrap.ensure_models()
    pipeline.load_models()
    asyncio.create_task(worker())
    _recover_unfinished()
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

    original_name = file.filename or "запись"
    # сохраняем загрузку в постоянную папку (не /tmp): переживёт перезапуск
    upload_path = os.path.join(store.uploads_dir, f"{uuid.uuid4().hex}{ext}")
    with open(upload_path, "wb") as dst:
        shutil.copyfileobj(file.file, dst)

    job_id = store.create_job(original_name, upload_path)
    _log(f"[очередь] принят файл: {original_name}", tag="очередь", job_id=job_id)
    await job_queue.put((job_id, upload_path, original_name))

    return JSONResponse({"job_id": job_id, "status": "queued"}, status_code=202)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = store.get_job(job_id)
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


@app.get("/api/jobs")
def jobs_list():
    """Активные задачи (в очереди и в работе) — для вкладки процессов."""
    return JSONResponse(store.active_jobs())


@app.get("/api/logs")
def logs(limit: int = 200, job_id: str = None):
    """Журнал обработки из БД."""
    return JSONResponse(store.recent_logs(limit=limit, job_id=job_id))


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
