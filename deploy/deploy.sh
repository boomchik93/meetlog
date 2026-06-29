#!/usr/bin/env bash
# =============================================================================
# Отправка CPU-пакета "Транскрибатора" на удалённый сервер по SFTP.
#
# Использование:
#   ./deploy.sh <ip> <username> [password] [remote_dir]
#
#   <ip>         адрес сервера
#   <username>   пользователь SSH
#   [password]   пароль (если не указать — спросит интерактивно, безопаснее)
#   [remote_dir] куда положить на сервере (по умолчанию: домашняя папка)
#
# Пример:
#   ./deploy.sh 1.2.3.4 root
#   ./deploy.sh 1.2.3.4 root mypass /opt
#
# Требуется: expect (на macOS обычно есть; иначе: brew install expect)
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ARCHIVE="${ARCHIVE:-$HERE/transcriber-cpu.tar.gz}"
SRC_DIR="$HERE/transcriber"

HOST="${1:-}"
SSH_USER="${2:-}"
PASS="${3:-}"
REMOTE_DIR="${4:-.}"

usage() {
  echo "Использование: $0 <ip> <username> [password] [remote_dir]"
  exit 1
}

[ -z "$HOST" ] && usage
[ -z "$SSH_USER" ] && usage
command -v expect >/dev/null 2>&1 || { echo "Ошибка: нужен expect (brew install expect)"; exit 1; }

# свежая переупаковка, если рядом лежит папка пакета
if [ -d "$SRC_DIR" ]; then
  echo "Переупаковываю пакет..."
  ( cd "$HERE" \
    && find transcriber -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true \
    && find transcriber -name '.DS_Store' -delete 2>/dev/null || true \
    && tar czf "$ARCHIVE" transcriber )
fi

[ -f "$ARCHIVE" ] || { echo "Ошибка: не найден архив $ARCHIVE"; exit 1; }

if [ -z "$PASS" ]; then
  read -r -s -p "Пароль для $SSH_USER@$HOST: " PASS
  echo
fi

ARNAME="$(basename "$ARCHIVE")"
echo "Отправляю $ARNAME -> $SSH_USER@$HOST:$REMOTE_DIR/"

# значения передаём через окружение: expect читает их как данные,
# поэтому спецсимволы в пароле/путях ([ ] $ \ " %) не ломают скрипт
DEPLOY_PASS="$PASS" \
DEPLOY_HOST="$HOST" \
DEPLOY_USER="$SSH_USER" \
DEPLOY_SRC="$ARCHIVE" \
DEPLOY_DST="$REMOTE_DIR/$ARNAME" \
expect <<'EOF'
set timeout -1
set pass $env(DEPLOY_PASS)
set host $env(DEPLOY_HOST)
set user $env(DEPLOY_USER)
set src  $env(DEPLOY_SRC)
set dst  $env(DEPLOY_DST)
spawn sftp -oStrictHostKeyChecking=accept-new -oUserKnownHostsFile=/dev/null $user@$host
expect {
  -re {[Pp]assword:} { send -- "$pass\r" }
  -re {sftp>} {}
  timeout { puts "таймаут подключения"; exit 1 }
}
expect "sftp>"
send -- "put \"$src\" \"$dst\"\r"
expect "sftp>"
send -- "bye\r"
expect eof
EOF

echo
echo "✅ Архив отправлен."
echo "Дальше на сервере (по SSH):"
echo "  ssh $SSH_USER@$HOST"
echo "  cd $REMOTE_DIR && tar xzf $ARNAME && cd transcriber && ./start.sh"
echo
echo "Потом открыть в браузере:  http://$HOST:8000"
