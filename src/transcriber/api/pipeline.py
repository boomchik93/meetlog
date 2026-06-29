import gc

from transcriber.core.audio import AudioLoader
from transcriber.core.transcriber import SpeechRecognizer
from transcriber.core.diarizer import SpeakerDiarizer
from transcriber.services.summarizer import Summarizer
from transcriber.config.settings import settings


# если средняя громкость канала ниже этого — считаем канал пустым
SILENT_CHANNEL_LEVEL = 1e-4


class Pipeline:
    """Полный путь обработки одного аудиофайла."""

    def __init__(self):
        self.loader = AudioLoader()
        self.recognizer = SpeechRecognizer()
        self.diarizer = SpeakerDiarizer()
        self.summarizer = Summarizer()

    def load_models(self):
        """Загружаем все модели. Вызывается один раз при старте сервера."""
        self.recognizer.load()
        if settings.diarization_enabled:
            self.diarizer.load()
        self.summarizer.load()

    # --- полный проход ---
    def run(self, audio_path):
        """Транскрибация + (коррекция) + пересказ."""
        result = self.process(audio_path)

        # LLM чинит ошибки распознавания, саммари строим по исправленному тексту
        corrected = None
        if settings.llm_correct and self.summarizer.ready:
            corrected = self.summarizer.correct_transcript(result["segments"])
            if corrected:
                result["text_corrected"] = corrected

        result["summary"] = self.summarizer.summarize(
            result["segments"], result["speakers"], corrected_text=corrected
        )
        return result

    # --- обработка файла ---
    def process(self, audio_path):
        """Главный метод: путь к файлу -> словарь с результатом."""
        audio = self.loader.load(audio_path)

        if self._is_stereo_phone(audio):
            pieces = self._handle_two_channels(audio)
        else:
            pieces = self._handle_one_channel(audio)

        speakers = self._collect_speakers(pieces)
        speaker_names = self._identify_speaker_names(pieces)
        pieces = self._enrich_segments(pieces, speaker_names)
        gc.collect()

        return {
            "language": "ru",
            "text": " ".join(p["text"] for p in pieces),
            "segments": pieces,
            "speakers": speakers,
            "speaker_names": speaker_names,
            "is_phone": audio.is_phone,
        }

    def _is_stereo_phone(self, audio):
        """Проверяем, что есть два канала и оба не пустые."""
        import numpy as np
        if audio.channel_count < 2:
            return False
        level0 = float(np.abs(audio.get_channel(0)).mean())
        level1 = float(np.abs(audio.get_channel(1)).mean())
        print(f"[оркестратор] громкость каналов: {level0:.5f} и {level1:.5f}")
        return level0 > SILENT_CHANNEL_LEVEL and level1 > SILENT_CHANNEL_LEVEL

    def _handle_two_channels(self, audio):
        """Два канала: каждый канал = отдельный человек."""
        print("[оркестратор] стерео: канал = спикер")
        pieces0 = self.recognizer.transcribe(audio.get_channel(0), audio.is_phone)
        pieces1 = self.recognizer.transcribe(audio.get_channel(1), audio.is_phone)

        result = []
        for piece in pieces0:
            item = piece.to_dict()
            item["speaker"] = "SPEAKER_00"
            result.append(item)
        for piece in pieces1:
            item = piece.to_dict()
            item["speaker"] = "SPEAKER_01"
            result.append(item)

        # время у всех общее, поэтому сортировка даёт правильный порядок
        result.sort(key=lambda x: x["start"])
        print(f"[оркестратор] готово: {len(result)} реплик")
        return result

    def _handle_one_channel(self, audio):
        """Один канал: распознаём и определяем спикеров по голосу."""
        print("[оркестратор] моно: определяю спикеров по голосу")
        mono = audio.to_mono() if audio.channel_count > 1 else audio.get_channel(0)
        pieces = self.recognizer.transcribe(mono, audio.is_phone)

        if self.diarizer.ready and pieces:
            return self.diarizer.assign_speakers(mono, pieces)

        # диаризация недоступна — всё на одного спикера
        result = []
        for piece in pieces:
            item = piece.to_dict()
            item["speaker"] = "SPEAKER_00"
            result.append(item)
        return result

    def _collect_speakers(self, pieces):
        """Собираем весь текст каждого спикера в одну строку."""
        speakers = {}
        for piece in pieces:
            name = piece["speaker"]
            old = speakers.get(name, "")
            speakers[name] = (old + " " + piece["text"]).strip()
        return speakers

    def _identify_speaker_names(self, pieces):
        """Определяем реальные имена спикеров через LLM."""
        if not self.summarizer.ready:
            print("[оркестратор] LLM не загружена, имена спикеров не определяются")
            return {}
        try:
            print("[оркестратор] определяю имена спикеров...")
            names = self.summarizer.identify_speakers(pieces)
            print(f"[оркестратор] имена спикеров: {names}")
            return names
        except Exception as error:
            print(f"[оркестратор] ошибка определения имён: {error}")
            return {}

    def _enrich_segments(self, pieces, speaker_names):
        """Добавляем поле speaker_name к каждому сегменту."""
        if not speaker_names:
            return pieces
        for piece in pieces:
            piece["speaker_name"] = speaker_names.get(piece.get("speaker"), None)
        return pieces

    # --- пересказ ---
    def make_summary(self, pieces, speakers):
        return self.summarizer.summarize(pieces, speakers)
