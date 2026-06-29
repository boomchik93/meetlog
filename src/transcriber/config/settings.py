# загрузка конфига и распределение ресурсов под железо
# Settings читает config.yml, смотрит на gpu/память и решает куда класть модели
import os

import yaml
import torch


# путь к конфигу можно задать переменной окружения, иначе берём дефолтный
DEFAULT_CONFIG_PATH = os.getenv("CONFIG_PATH", "config/config.yml")


class Hardware:
    def __init__(self, device, vram_gb, ram_gb, cpu_threads):
        self.device = device          # "cuda", "mps" или "cpu"
        self.vram_gb = vram_gb        # видеопамять в гигабайтах
        self.ram_gb = ram_gb          # оперативная память в гигабайтах
        self.cpu_threads = cpu_threads

    @property
    def has_gpu(self):
        return self.device == "cuda"

    def __repr__(self):
        return (f"Hardware(device={self.device}, vram={self.vram_gb}GB, "
                f"ram={self.ram_gb}GB, threads={self.cpu_threads})")


class Settings:
    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.raw = self._read_file(config_path)

        # разбираем железо
        self.hardware = self._detect_hardware(self.raw.get("hardware", {}))

        # настройки сервера
        server = self.raw.get("server", {})
        self.host = server.get("host", "0.0.0.0")
        self.port = int(os.getenv("PORT", server.get("port", 8000)))
        self.log_level = server.get("log_level", "info")

        # настройки whisper
        whisper = self.raw.get("whisper", {})
        self.whisper_hf_model = whisper.get("hf_model", "bond005/whisper-podlodka-turbo")
        self.whisper_use_hf = whisper.get("use_hf_model", True)
        self.whisper_fallback = os.getenv("WHISPER_MODEL", whisper.get("fallback_model", "large-v3"))
        self.whisper_beam_size = int(whisper.get("beam_size", 5))
        self.whisper_compute_cfg = whisper.get("compute_type", "auto")
        # whisper можно увести на cpu отдельно от LLM (для старых GPU)
        self.whisper_device_cfg = whisper.get("device", "auto")
        # движок распознавания и модель whisper.cpp
        self.whisper_engine = whisper.get("engine", "whispercpp")
        self.whisper_ggml_repo = whisper.get("ggml_repo", "ggerganov/whisper.cpp")
        self.whisper_ggml_model = whisper.get("ggml_model", "ggml-large-v3.bin")
        self.whisper_ggml_path = whisper.get("ggml_path", "/models/ggml-large-v3.bin")
        # русская дообученная модель -> конвертируется в ct2 (пусто = обычный fallback)
        self.whisper_ru_model = whisper.get("ru_model", "")
        self.whisper_ru_ct2_dir = whisper.get("ru_ct2_dir", "/models/whisper-ru-ct2")

        # шумоподавление аудио
        audio = self.raw.get("audio", {})
        self.audio_denoise = bool(audio.get("denoise", False))

        # настройки llm
        llm = self.raw.get("llm", {})
        self.llm_enabled = llm.get("enabled", True)
        self.llm_repo = llm.get("repo", "")
        self.llm_filename = llm.get("filename", "")
        self.llm_path = llm.get("local_path", "")
        # LLM чинит транскрипт перед саммари
        self.llm_correct = bool(llm.get("correct_transcript", False))

        # диаризация
        diar = self.raw.get("diarization", {})
        self.diarization_enabled = diar.get("enabled", True)

        # считаем параметры моделей под железо, затем применяем overrides
        self._plan_resources()
        self._apply_overrides(self.raw.get("overrides", {}))

    # --- чтение файла ---
    def _read_file(self, path):
        if not os.path.exists(path):
            print(f"[настройки] конфиг {path} не найден, беру значения по умолчанию")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # --- определение железа ---
    def _detect_hardware(self, cfg):
        # DEVICE из окружения перебивает конфиг (удобно при запуске)
        device = os.getenv("DEVICE", cfg.get("device", "auto"))
        if device == "auto":
            device = self._auto_device()

        cpu_threads = cfg.get("cpu_threads", "auto")
        if cpu_threads == "auto":
            cpu_threads = os.cpu_count() or 4

        vram_gb = float(cfg.get("vram_gb", 0))
        ram_gb = float(cfg.get("ram_gb", 16))

        # если выбрали cuda, но видеопамять не указали — попробуем спросить у GPU
        if device == "cuda" and vram_gb == 0:
            vram_gb = self._read_gpu_vram()

        hw = Hardware(device, vram_gb, ram_gb, int(cpu_threads))
        print(f"[настройки] железо: {hw}")
        return hw

    def _pick_compute(self):
        # ручной выбор из конфига имеет приоритет
        if self.whisper_compute_cfg != "auto":
            return self.whisper_compute_cfg
        # на старых GPU (Pascal, cc<7) float16 медленный — берём int8
        try:
            major = torch.cuda.get_device_capability(0)[0]
        except Exception:
            major = 0
        return "float16" if major >= 7 else "int8_float32"

    def _auto_device(self):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _read_gpu_vram(self):
        try:
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            return round(total_bytes / (1024 ** 3), 1)
        except Exception:
            return 0.0

    # --- распределение ресурсов под железо ---
    def _plan_resources(self):
        hw = self.hardware

        # 1. Куда положить Whisper.
        # Whisper небольшой (~1.6 ГБ), кладём на GPU если он есть.
        # whisper.device может принудительно перебить общий выбор
        if self.whisper_device_cfg != "auto":
            use_gpu = self.whisper_device_cfg == "cuda"
        else:
            use_gpu = hw.has_gpu

        if use_gpu:
            self.whisper_device = "cuda"
            self.whisper_compute = self._pick_compute()
            self.whisper_dtype = torch.float16 if self.whisper_compute == "float16" else torch.float32
        else:
            # на mac (mps) faster-whisper всё равно работает через cpu
            self.whisper_device = "cpu"
            self.whisper_compute = self.whisper_compute_cfg if self.whisper_compute_cfg != "auto" else "int8"
            self.whisper_dtype = torch.float32

        # 2. Сколько слоёв LLM положить на GPU.
        # LLM большая (~6.5 ГБ с контекстом). Кладём на GPU, только если
        # видеопамяти заметно больше, чем нужно Whisper'у.
        if hw.has_gpu and hw.vram_gb >= 10:
            self.llm_gpu_layers = -1   # вся модель на GPU
        elif hw.has_gpu and hw.vram_gb >= 6:
            self.llm_gpu_layers = 20   # часть слоёв на GPU
        else:
            self.llm_gpu_layers = 0    # только процессор

        # 3. Размер контекста LLM.
        # Большой контекст ест много памяти, а нам столько не нужно:
        # транскрипт разговора всё равно обрезается. Считаем по свободной
        # памяти — это то, что осталось после весов самой модели (~5 ГБ).
        total_memory = hw.vram_gb if self.llm_gpu_layers != 0 else hw.ram_gb
        free_memory = total_memory - 5
        if free_memory >= 12:
            self.llm_context = 16384
        elif free_memory >= 5:
            self.llm_context = 8192
        else:
            self.llm_context = 4096

        # 4. Потоки для LLM. На GPU потоки почти не важны, на CPU — берём все.
        self.llm_threads = hw.cpu_threads

        print(f"[настройки] план: whisper на {self.whisper_device}, "
              f"llm_gpu_layers={self.llm_gpu_layers}, "
              f"llm_context={self.llm_context}, llm_threads={self.llm_threads}")

    # --- ручные переопределения ---
    def _apply_overrides(self, overrides):
        gpu_layers = overrides.get("llm_gpu_layers")
        if gpu_layers is not None:
            self.llm_gpu_layers = int(gpu_layers)
            print(f"[настройки] override: llm_gpu_layers={self.llm_gpu_layers}")

        context = overrides.get("llm_context")
        if context is not None:
            self.llm_context = int(context)
            print(f"[настройки] override: llm_context={self.llm_context}")

        threads = overrides.get("llm_threads")
        if threads is not None:
            self.llm_threads = int(threads)
            print(f"[настройки] override: llm_threads={self.llm_threads}")


# один общий объект настроек на весь проект
settings = Settings()
