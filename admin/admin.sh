#!/usr/bin/env bash
# =============================================================================
# Запуск админ-панели одной командой.
#   ./admin.sh <ip> <username> [password]
# Подключается по SSH к серверу с транскрибатором и открывает дашборд.
# Требует: python3, Node/npm (для разовой сборки фронтенда).
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

HOST="${1:-}"
SSH_USER="${2:-}"
PASS="${3:-}"
PORT="${ADMIN_PORT:-8765}"

if [ -z "$HOST" ] || [ -z "$SSH_USER" ]; then
  echo "Использование: $0 <ip> <username> [password]"; exit 1
fi
if [ -z "$PASS" ]; then
  read -r -s -p "Пароль $SSH_USER@$HOST: " PASS; echo
fi

# 1. фронтенд (собираем один раз)
if [ ! -d "$HERE/frontend/dist" ]; then
  command -v npm >/dev/null 2>&1 || { echo "Нужен Node/npm для сборки фронтенда (brew install node)"; exit 1; }
  echo "Собираю фронтенд (разово, пару минут)..."
  ( cd "$HERE/frontend" && npm install && npm run build )
fi

# 2. бэкенд (venv с paramiko/fastapi)
VENV="$HERE/.venv"
if [ ! -d "$VENV" ]; then
  echo "Ставлю зависимости бэкенда..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r "$HERE/backend/requirements.txt"
fi

# 3. запуск
export SRV_HOST="$HOST" SRV_USER="$SSH_USER" SRV_PASS="$PASS"
export SRV_CONTAINER="${SRV_CONTAINER:-transcriber-transcriber-1}"
echo
echo "✅ Панель: http://localhost:$PORT"
echo "   (Ctrl+C — остановить)"
( sleep 2 && { open "http://localhost:$PORT" 2>/dev/null || xdg-open "http://localhost:$PORT" 2>/dev/null || true; } ) &
exec "$VENV/bin/python" -m uvicorn server:app --app-dir "$HERE/backend" --host 127.0.0.1 --port "$PORT"
