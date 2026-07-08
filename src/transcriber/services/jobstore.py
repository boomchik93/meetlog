"""
Персистентное хранилище задач и логов (SQLite).

Зачем: очередь и статусы задач раньше жили только в оперативной памяти.
Любой перезапуск процесса (перезагрузка, docker restart, нехватка памяти)
терял всё, что не успело досчитаться: файл пропадал из вкладки процессов
и не появлялся в истории. Теперь задача пишется на диск сразу при приёме,
а загруженный аудиофайл сохраняется в постоянную папку, а не в /tmp. При
старте сервера незавершённые задачи возвращаются в очередь и досчитываются.

Одна БД на весь проект: таблица jobs (жизненный цикл задач) и таблица logs
(события обработки — тот самый журнал вместо разрозненных print в консоль).
"""
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path


# папка для БД и загруженных файлов (переживает перезапуск, лежит на volume)
DATA_DIR = os.getenv("DATA_DIR", "data")
DB_PATH = os.path.join(DATA_DIR, "jobs.db")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")

# статусы задачи
QUEUED = "queued"
PROCESSING = "processing"
DONE = "done"
ERROR = "error"


def _now():
    return datetime.now().isoformat(timespec="seconds")


class JobStore:
    """Обёртка над SQLite. Потокобезопасна: один lock на запись."""

    def __init__(self, db_path=DB_PATH, uploads_dir=UPLOADS_DIR):
        self.db_path = db_path
        self.uploads_dir = uploads_dir
        Path(os.path.dirname(db_path) or ".").mkdir(parents=True, exist_ok=True)
        Path(uploads_dir).mkdir(parents=True, exist_ok=True)
        # check_same_thread=False — обращаемся из воркера и из обработчиков FastAPI
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL — чтобы чтение истории не блокировалось записью статуса задачи
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id            TEXT PRIMARY KEY,
                    status        TEXT NOT NULL,
                    filename      TEXT NOT NULL,
                    upload_path   TEXT,
                    submitted_at  TEXT NOT NULL,
                    started_at    TEXT,
                    finished_at   TEXT,
                    saved_to      TEXT,
                    result_json   TEXT,
                    error         TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                CREATE INDEX IF NOT EXISTS idx_jobs_submitted ON jobs(submitted_at);

                CREATE TABLE IF NOT EXISTS logs (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts        TEXT NOT NULL,
                    level     TEXT NOT NULL,
                    tag       TEXT,
                    job_id    TEXT,
                    message   TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_logs_ts ON logs(ts);
                CREATE INDEX IF NOT EXISTS idx_logs_job ON logs(job_id);
                """
            )
            self._conn.commit()

    # --- задачи ---

    def create_job(self, filename, upload_path):
        """Регистрируем задачу в статусе queued. Возвращает job_id."""
        job_id = str(uuid.uuid4())
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, status, filename, upload_path, submitted_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (job_id, QUEUED, filename, upload_path, _now()),
            )
            self._conn.commit()
        return job_id

    def mark_processing(self, job_id):
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status=?, started_at=? WHERE id=?",
                (PROCESSING, _now(), job_id),
            )
            self._conn.commit()

    def mark_done(self, job_id, result, saved_to):
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status=?, finished_at=?, saved_to=?, result_json=? WHERE id=?",
                (DONE, _now(), str(saved_to), json.dumps(result, ensure_ascii=False), job_id),
            )
            self._conn.commit()

    def mark_error(self, job_id, error):
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status=?, finished_at=?, error=? WHERE id=?",
                (ERROR, _now(), str(error), job_id),
            )
            self._conn.commit()

    def get_job(self, job_id):
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def active_jobs(self):
        """Задачи в работе или в очереди — для вкладки процессов."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE status IN (?, ?) ORDER BY submitted_at",
                (QUEUED, PROCESSING),
            ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def unfinished_jobs(self):
        """Незавершённые задачи для восстановления при старте (сначала старые)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE status IN (?, ?) ORDER BY submitted_at",
                (QUEUED, PROCESSING),
            ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def requeue(self, job_id):
        """Вернуть прерванную задачу обратно в очередь (сброс в queued)."""
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status=?, started_at=NULL WHERE id=?",
                (QUEUED, job_id),
            )
            self._conn.commit()

    def _row_to_job(self, row):
        job = dict(row)
        job["result"] = json.loads(job["result_json"]) if job.get("result_json") else None
        job.pop("result_json", None)
        return job

    # --- журнал ---

    def log(self, message, level="info", tag=None, job_id=None):
        with self._lock:
            self._conn.execute(
                "INSERT INTO logs (ts, level, tag, job_id, message) VALUES (?, ?, ?, ?, ?)",
                (_now(), level, tag, job_id, message),
            )
            self._conn.commit()

    def recent_logs(self, limit=200, job_id=None):
        with self._lock:
            if job_id:
                rows = self._conn.execute(
                    "SELECT ts, level, tag, job_id, message FROM logs "
                    "WHERE job_id=? ORDER BY id DESC LIMIT ?",
                    (job_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT ts, level, tag, job_id, message FROM logs "
                    "ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]


# один общий экземпляр на весь проект
store = JobStore()
