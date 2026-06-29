import json
import os
import sqlite3
import time

DB_PATH = os.getenv("METRICS_DB", "/data/metrics.db")


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def init():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS samples(
            ts INTEGER, cpu REAL, cores TEXT,
            mem_used INTEGER, mem_total INTEGER, mem_pct REAL,
            swap_used INTEGER, swap_total INTEGER,
            load1 REAL, load5 REAL, load15 REAL, disks TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS tasks(
            ts INTEGER, filename TEXT, audio_sec REAL, proc_sec REAL,
            status TEXT, title TEXT, saved_to TEXT)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_samples_ts ON samples(ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tasks_ts ON tasks(ts)")


def insert_sample(s):
    with _conn() as c:
        c.execute(
            "INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (s["ts"], s["cpu"], json.dumps(s["cores"]),
             s["mem_used"], s["mem_total"], s["mem_pct"],
             s["swap_used"], s["swap_total"],
             s["load1"], s["load5"], s["load15"], json.dumps(s["disks"])),
        )


def insert_task(t):
    with _conn() as c:
        c.execute(
            "INSERT INTO tasks VALUES(?,?,?,?,?,?,?)",
            (int(time.time()), t.get("filename"), t.get("audio_sec"),
             t.get("proc_sec"), t.get("status"), t.get("title"), t.get("saved_to")),
        )


def prune(days=30):
    cutoff = int(time.time()) - days * 86400
    with _conn() as c:
        c.execute("DELETE FROM samples WHERE ts < ?", (cutoff,))
        c.execute("DELETE FROM tasks WHERE ts < ?", (cutoff,))


# --- чтение (используется панелью через ssh python -c) ---
def samples_since(seconds):
    since = int(time.time()) - int(seconds)
    with _conn() as c:
        rows = c.execute(
            "SELECT ts,cpu,cores,mem_pct,mem_used,mem_total,swap_used,"
            "load1,load5,load15,disks FROM samples WHERE ts >= ? ORDER BY ts", (since,)
        ).fetchall()
    return [
        {"ts": r[0], "cpu": r[1], "cores": json.loads(r[2]), "mem_pct": r[3],
         "mem_used": r[4], "mem_total": r[5], "swap_used": r[6],
         "load1": r[7], "load5": r[8], "load15": r[9], "disks": json.loads(r[10])}
        for r in rows
    ]


def tasks_recent(limit=200):
    with _conn() as c:
        rows = c.execute(
            "SELECT ts,filename,audio_sec,proc_sec,status,title,saved_to "
            "FROM tasks ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    keys = ["ts", "filename", "audio_sec", "proc_sec", "status", "title", "saved_to"]
    return [dict(zip(keys, r)) for r in rows]
