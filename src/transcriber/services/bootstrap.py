import os

from transcriber.config.settings import settings


def ensure_models():
    """Скачать всё, что не качается автоматически."""
    ensure_ru_model()
    ensure_ggml_model()
    ensure_llm_model()


def ensure_ru_model():
    """Сконвертировать русскую HF-модель Whisper в формат ct2 (один раз)."""
    if not settings.whisper_ru_model:
        return
    out_dir = settings.whisper_ru_ct2_dir
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        print(f"[bootstrap] русская ct2-модель на месте: {out_dir}")
        return

    import subprocess
    print(f"[bootstrap] конвертирую {settings.whisper_ru_model} -> ct2 int8 (качает ~3 ГБ, разово)")
    cmd = [
        "ct2-transformers-converter",
        "--model", settings.whisper_ru_model,
        "--output_dir", out_dir,
        "--quantization", "int8",
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
        "--force",
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"[bootstrap] русская ct2-модель готова: {out_dir}")
    except Exception as error:
        print(f"[bootstrap] конвертация не удалась ({error}); останусь на обычной модели")


def ensure_ggml_model():
    """Скачать ggml-модель whisper.cpp, если её нет."""
    if settings.whisper_engine != "whispercpp":
        return
    if os.path.exists(settings.whisper_ggml_path):
        print(f"[bootstrap] whisper.cpp модель на месте: {settings.whisper_ggml_path}")
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[bootstrap] huggingface_hub не установлен, ggml не скачать")
        return

    target_dir = os.path.dirname(settings.whisper_ggml_path) or "."
    os.makedirs(target_dir, exist_ok=True)
    print(f"[bootstrap] качаю whisper.cpp {settings.whisper_ggml_repo}/{settings.whisper_ggml_model}")
    path = hf_hub_download(
        repo_id=settings.whisper_ggml_repo,
        filename=settings.whisper_ggml_model,
        local_dir=target_dir,
    )
    print(f"[bootstrap] whisper.cpp модель скачана: {path}")


def ensure_llm_model():
    """Скачать GGUF-модель LLM в local_path, если её там нет."""
    if not settings.llm_enabled:
        return
    if os.path.exists(settings.llm_path):
        print(f"[bootstrap] LLM на месте: {settings.llm_path}")
        return
    if not settings.llm_repo or not settings.llm_filename:
        print("[bootstrap] repo/filename для LLM не заданы, пропускаю загрузку")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[bootstrap] huggingface_hub не установлен, LLM не скачать")
        return

    target_dir = os.path.dirname(settings.llm_path) or "."
    os.makedirs(target_dir, exist_ok=True)
    print(f"[bootstrap] качаю LLM {settings.llm_repo}/{settings.llm_filename}")
    path = hf_hub_download(
        repo_id=settings.llm_repo,
        filename=settings.llm_filename,
        local_dir=target_dir,
    )
    print(f"[bootstrap] LLM скачана: {path}")
