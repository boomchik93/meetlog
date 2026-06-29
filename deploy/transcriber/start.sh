#!/usr/bin/env bash
# Запуск транскрибатора одной командой (CPU).
set -e
cd "$(dirname "$0")"

echo "=== Транскрибатор (CPU) ==="
echo "Сборка образа и запуск. Первый запуск скачает модели (~5-6 ГБ) — это займёт время."
echo

docker compose up --build -d

echo
echo "Контейнер запускается. Модели грузятся при первом старте — следите за логами:"
echo "    docker compose logs -f"
echo
echo "Когда увидите 'Application startup complete' — откройте в браузере:"
echo "    http://<IP-этого-сервера>:8000"
echo
echo "Остановить:  docker compose down"
