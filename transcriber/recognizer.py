"""
Распознавание речи.

Whisper обрабатывает окна максимум по 30 секунд. Если отдать ему длинную
запись целиком, он режет её сам, но таймкоды каждого окна считает от нуля —
сегменты приходят с ломаным временем. Поэтому нарезку делаем сами по VAD,
а время берём из границ нарезки.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

from transcriber.settings import settings


MIN_PIECE_SECONDS = 0.3
RATE = 16000


class TextPiece:
    def __init__(self, start, end, text):
        self.start = round(start, 2)
        self.end = round(end, 2)
        self.text = text.strip()

    def to_dict(self):
        return {"start": self.start, "end": self.end, "text": self.text}


class SpeechRecognizer:
    def __init__(self):
        self.hf_model = None
        self.hf_processor = None
        self.fast_model = None
        self.wcpp = None
        self.ready = False

    def load(self):
        if settings.whisper_engine == "whispercpp":
            self._load_whispercpp()
        elif settings.whisper_use_hf:
            self._load_main_model()
            if self.hf_model is None:
                self._load_backup_model()
        else:
            self._load_backup_model()
        self.ready = True

    def _load_whispercpp(self):
        # pywhispercpp здесь НЕ импортируем: его libggml конфликтует с llama.cpp
        # в одном процессе. Модель грузится в отдельном воркере.
        path = settings.whisper_ggml_path
        if not os.path.exists(path):
            print(f"[распознавание] ggml модель не найдена: {path}")
            return
        self.wcpp = True
        print(f"[распознавание] whisper.cpp (GPU, воркер): {path}")

    def _load_main_model(self):
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            name = settings.whisper_hf_model
            print(f"[распознавание] гружу русскую модель: {name}")
            self.hf_processor = AutoProcessor.from_pretrained(name)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                name,
                torch_dtype=settings.whisper_dtype,
                device_map="auto",
            )
            model.eval()
            # иначе язык не задать вручную
            model.generation_config.forced_decoder_ids = None
            self.hf_model = model
            print("[распознавание] русская модель готова")
        except Exception as error:
            print(f"[распознавание] не вышло загрузить русскую модель: {error}")
            self.hf_model = None

    def _load_backup_model(self):
        from faster_whisper import WhisperModel
        name = settings.whisper_fallback
        print(f"[распознавание] гружу запасную модель: {name}")
        self.fast_model = WhisperModel(
            name,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute,
            cpu_threads=settings.hardware.cpu_threads,
        )
        print("[распознавание] запасная модель готова")

    @property
    def model_name(self):
        if self.wcpp is not None:
            return settings.whisper_ggml_model, "whisper.cpp"
        if self.hf_model is not None:
            return settings.whisper_hf_model, "huggingface"
        return settings.whisper_fallback, "faster-whisper"

    def transcribe(self, audio, is_phone=False):
        """Аудио -> список TextPiece."""
        if self.wcpp is not None:
            pieces = self._transcribe_whispercpp(audio, is_phone)
        elif self.hf_model is not None:
            pieces = self._transcribe_main(audio, is_phone)
        else:
            pieces = self._transcribe_backup(audio, is_phone)
        return self._remove_repeats(pieces)

    def _transcribe_whispercpp(self, audio, is_phone):
        wav = np.ascontiguousarray(audio, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            np.save(tmp, wav)
            npy_path = tmp.name

        worker = os.path.join(os.path.dirname(__file__), "whisper_worker.py")
        env = dict(os.environ)
        # libwhisper/libggml лежат отдельно и видны только процессу-воркеру
        whisper_libs = "/opt/wlib"
        env["LD_LIBRARY_PATH"] = whisper_libs + ":" + env.get("LD_LIBRARY_PATH", "")

        try:
            res = subprocess.run(
                [sys.executable, worker, npy_path, settings.whisper_ggml_path],
                capture_output=True, text=True, env=env,
            )
        finally:
            os.unlink(npy_path)

        if res.returncode != 0:
            print(f"[распознавание] whisper.cpp воркер упал: {res.stderr[-400:]}")
            return []

        data = json.loads(res.stdout.strip().splitlines()[-1])
        pieces = []
        for seg in data:
            text = seg["text"].strip()
            if text:
                # t0/t1 приходят в сантисекундах
                pieces.append(TextPiece(seg["t0"] / 100.0, seg["t1"] / 100.0, text))
        return pieces

    def _transcribe_main(self, audio, is_phone):
        voice_parts = self._find_voice(audio, is_phone)
        print(f"[распознавание] нашёл {len(voice_parts)} кусков с речью")

        pieces = []
        for part in voice_parts:
            start_sample = int(part["start"] * RATE)
            end_sample = int(part["end"] * RATE)
            chunk = audio[start_sample:end_sample]

            if len(chunk) < int(RATE * MIN_PIECE_SECONDS):
                continue

            text = self._recognize_chunk(chunk)
            if text:
                pieces.append(TextPiece(part["start"], part["end"], text))
        return pieces

    def _find_voice(self, audio, is_phone):
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        options = VadOptions(
            threshold=0.4 if is_phone else 0.5,
            min_speech_duration_ms=200,
            max_speech_duration_s=28.0,     # с запасом до окна Whisper в 30 сек
            min_silence_duration_ms=800 if is_phone else 500,
            speech_pad_ms=400,
        )
        raw = get_speech_timestamps(audio, options, sampling_rate=RATE)

        parts = []
        for item in raw:
            start = item["start"] / RATE
            end = item["end"] / RATE
            if end - start >= MIN_PIECE_SECONDS:
                parts.append({"start": start, "end": end})
        return parts

    def _recognize_chunk(self, chunk):
        inputs = self.hf_processor(chunk, sampling_rate=RATE, return_tensors="pt")
        features = inputs.input_features.to(self.hf_model.device,
                                            dtype=settings.whisper_dtype)

        with torch.no_grad():
            result = self.hf_model.generate(
                features,
                task="transcribe",
                language="russian",
                num_beams=settings.whisper_beam_size,
                no_repeat_ngram_size=3,
            )
        text = self.hf_processor.batch_decode(result, skip_special_tokens=True)[0]
        return text.strip()

    def _transcribe_backup(self, audio, is_phone):
        # faster-whisper сам режет по VAD и отдаёт корректные таймкоды
        segments, _ = self.fast_model.transcribe(
            audio,
            beam_size=settings.whisper_beam_size,
            best_of=5,
            temperature=[0.0, 0.2, 0.4, 0.6] if is_phone else [0.0, 0.2, 0.4],
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 800 if is_phone else 500,
                "speech_pad_ms": 400,
            },
            language="ru",
            condition_on_previous_text=False,
            no_speech_threshold=0.65 if is_phone else 0.6,
            compression_ratio_threshold=2.4 if is_phone else 2.0,
            log_prob_threshold=-1.0,
            word_timestamps=False,
        )
        pieces = []
        for seg in segments:
            if seg.text.strip():
                pieces.append(TextPiece(seg.start, seg.end, seg.text))
        return pieces

    def _remove_repeats(self, pieces):
        """Схлопываем подряд идущие одинаковые куски — частое залипание Whisper."""
        result = []
        for piece in pieces:
            if result and piece.text == result[-1].text:
                result[-1].end = piece.end
            else:
                result.append(piece)
        return result
