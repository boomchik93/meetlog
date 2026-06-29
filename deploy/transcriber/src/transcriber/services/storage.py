import json
import os
import re
from datetime import datetime
from pathlib import Path


OUTPUT_DIR = os.getenv("OUTPUT_DIR", "OUTPUTS")
MAX_TITLE_LEN = 60
# запрещённые в имени файла/папки символы
BAD_CHARS = r'[\\/:*?"<>|\n\r\t]'


def save_result(result, audio_filename, base_dir=OUTPUT_DIR):
    """Результат пайплайна -> папка с JSON. Возвращает путь к файлу."""
    title = _pick_title(result)
    date = datetime.now().strftime("%d.%m.%Y")
    folder = _safe_name(f"разбор обсуждения {title} от {date}")

    folder_path = Path(base_dir) / folder
    folder_path.mkdir(parents=True, exist_ok=True)

    name = _safe_name(Path(audio_filename).stem) + ".json"
    out_path = folder_path / name
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[storage] сохранил разбор: {out_path}")
    return out_path


def _pick_title(result):
    """Тема для имени папки берётся из summary.title."""
    summary = result.get("summary") or {}
    title = (summary.get("title") or "").strip()
    if not title:
        title = "без темы"
    return _clip(title, MAX_TITLE_LEN)


def _safe_name(name):
    cleaned = re.sub(BAD_CHARS, " ", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "без названия"


def _clip(text, limit):
    return text if len(text) <= limit else text[:limit].rstrip()
