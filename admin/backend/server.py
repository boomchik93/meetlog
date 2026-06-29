import asyncio
import json
import os
import shlex

import paramiko
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

HOST = os.environ["SRV_HOST"]
USER = os.environ["SRV_USER"]
PASS = os.environ["SRV_PASS"]
CONTAINER = os.getenv("SRV_CONTAINER", "transcriber-transcriber-1")
SERVICE_PORT = os.getenv("SRV_SERVICE_PORT", "8000")
DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")

app = FastAPI(title="Админ-панель транскрибатора")


# --- SSH helpers ---
def _client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=20, banner_timeout=20)
    return c


def run(cmd, timeout=60):
    c = _client()
    try:
        _, out, err = c.exec_command(cmd, timeout=timeout)
        return out.read().decode("utf-8", "replace"), err.read().decode("utf-8", "replace")
    finally:
        c.close()


def container_py(code):
    # выполняем питон ВНУТРИ контейнера — там есть модуль metrics_db и сама БД
    cmd = f"docker exec {shlex.quote(CONTAINER)} python -c {shlex.quote(code)}"
    out, err = run(cmd)
    return out, err


# --- API ---
@app.get("/api/history")
def history(seconds: int = 3600):
    code = ("import json;from transcriber.services import metrics_db;"
            f"print(json.dumps(metrics_db.samples_since({int(seconds)})))")
    out, err = container_py(code)
    try:
        return JSONResponse(json.loads(out))
    except Exception:
        return JSONResponse({"error": err or out, "data": []}, status_code=502)


@app.get("/api/tasks")
def tasks(limit: int = 200):
    code = ("import json;from transcriber.services import metrics_db;"
            f"print(json.dumps(metrics_db.tasks_recent({int(limit)})))")
    out, err = container_py(code)
    try:
        return JSONResponse(json.loads(out))
    except Exception:
        return JSONResponse({"error": err or out, "data": []}, status_code=502)


@app.get("/api/result")
def result(path: str):
    # безопасность: только внутри OUTPUTS
    if ".." in path or not path.startswith("OUTPUTS"):
        return JSONResponse({"error": "bad path"}, status_code=400)
    out, err = run(f"docker exec {shlex.quote(CONTAINER)} cat {shlex.quote('/app/' + path)}")
    try:
        return JSONResponse(json.loads(out))
    except Exception:
        return JSONResponse({"error": err or "not found"}, status_code=404)


@app.get("/api/service")
def service():
    out, _ = run(f"curl -s --max-time 5 localhost:{SERVICE_PORT}/api/health")
    try:
        return JSONResponse(json.loads(out))
    except Exception:
        return JSONResponse({"status": "down"}, status_code=200)


@app.get("/api/info")
def info():
    out, _ = run("uname -a; echo '---'; nproc; echo '---'; "
                 "free -m | awk 'NR==2{print $2}'; echo '---'; uptime -p")
    return {"host": HOST, "container": CONTAINER, "raw": out}


# --- веб-терминал (websocket <-> ssh pty) ---
@app.websocket("/ws/terminal")
async def terminal(ws: WebSocket):
    await ws.accept()
    cli = _client()
    chan = cli.invoke_shell(term="xterm-256color", width=120, height=32)
    chan.settimeout(0.0)
    loop = asyncio.get_event_loop()

    async def pump_out():
        while True:
            await asyncio.sleep(0.02)
            try:
                data = await loop.run_in_executor(None, _recv, chan)
            except Exception:
                break
            if data is None:
                continue
            if data == b"":
                break
            try:
                await ws.send_bytes(data)
            except Exception:
                break

    task = asyncio.create_task(pump_out())
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            text = msg.get("text")
            data = msg.get("bytes")
            if text is not None:
                if text.startswith('{"resize"'):
                    d = json.loads(text)["resize"]
                    chan.resize_pty(width=int(d["cols"]), height=int(d["rows"]))
                else:
                    chan.send(text)
            elif data is not None:
                chan.send(data)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        task.cancel()
        chan.close()
        cli.close()


def _recv(chan):
    if chan.recv_ready():
        return chan.recv(8192)
    if chan.closed or chan.eof_received:
        return b""
    return None


# --- статика собранного фронтенда (в конце, чтобы не перебить /api) ---
if os.path.isdir(DIST):
    app.mount("/", StaticFiles(directory=DIST, html=True), name="static")
