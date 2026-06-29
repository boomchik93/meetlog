import librosa
import numpy as np
from scipy import signal


# частота, с которой работает Whisper
TARGET_RATE = 16000
# частота телефонной записи (узкая полоса)
PHONE_RATE = 8000


class AudioData:
    """Готовое аудио: каналы, их количество и флаг телефонии."""

    def __init__(self, channels, is_phone):
        self.channels = channels        # массив [каналы x отсчёты]
        self.is_phone = is_phone        # это телефонная запись?

    @property
    def channel_count(self):
        return self.channels.shape[0]

    def get_channel(self, index):
        return self.channels[index]

    def to_mono(self):
        """Смешать все каналы в один."""
        return self.channels.mean(axis=0)


class AudioLoader:
    """Открывает аудиофайл и готовит его к распознаванию."""

    def load(self, path):
        # librosa отдаёт numpy в нативной частоте, все каналы
        waveform, rate = librosa.load(path, sr=None, mono=False)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        rate = int(rate)
        is_phone = rate == PHONE_RATE
        print(f"[аудио] открыл файл: {rate} Гц, каналов: {waveform.shape[0]}")

        if is_phone:
            print("[аудио] это телефон, чищу звук")
            channels = self._prepare_phone(waveform)
        else:
            channels = self._prepare_normal(waveform, rate)

        channels = self._maybe_denoise(channels)
        return AudioData(channels, is_phone)

    def _maybe_denoise(self, channels):
        """Шумоподавление каждого канала (noisereduce). Включается в конфиге."""
        from transcriber.config.settings import settings
        if not settings.audio_denoise:
            return channels
        try:
            import noisereduce as nr
        except ImportError:
            print("[аудио] noisereduce не установлен, денойз пропущен")
            return channels

        print("[аудио] шумоподавление каналов...")
        cleaned = []
        for i in range(channels.shape[0]):
            try:
                ch = nr.reduce_noise(y=channels[i], sr=TARGET_RATE, stationary=False)
            except Exception as error:
                print(f"[аудио] денойз канала {i} не удался: {error}")
                ch = channels[i]
            cleaned.append(ch.astype(np.float32))
        return np.stack(cleaned)

    def _prepare_normal(self, waveform, rate):
        """Обычное аудио: просто меняем частоту на 16 кГц, если надо."""
        if rate != TARGET_RATE:
            waveform = librosa.resample(waveform, orig_sr=rate, target_sr=TARGET_RATE)
        return waveform.astype(np.float32)

    def _prepare_phone(self, waveform):
        """Телефонное аудио: чистим каждый канал и поднимаем до 16 кГц."""
        cleaned = []
        for i in range(waveform.shape[0]):
            one_channel = waveform[i]
            cleaned.append(self._upsample_phone(one_channel))
        return np.stack(cleaned)

    def _upsample_phone(self, audio):
        """Чистим телефонный канал и поднимаем частоту с 8 до 16 кГц."""
        clean = self._clean_phone_noise(audio, PHONE_RATE)
        upsampled = signal.resample_poly(
            clean,
            2, 1,                       # 8000 * 2 = 16000
            window=("kaiser", 5.0),
            padtype="line",
        )
        return upsampled.astype(np.float32)

    def _clean_phone_noise(self, audio, rate):
        """Оставляем только голосовую полосу (300-3400 Гц) и убираем тихий шум."""
        nyquist = rate / 2
        low = 300 / nyquist
        high = min(3400 / nyquist, 0.95)

        # полосовой фильтр: пропускаем только частоты голоса
        sos = signal.butter(4, [low, high], btype="band", output="sos")
        filtered = signal.sosfilt(sos, audio)

        # глушим тихие звуки (шум), оставляем голос
        rms = np.sqrt(np.mean(filtered ** 2))
        threshold = rms * 0.1
        gate = np.tanh((np.abs(filtered) - threshold) / (threshold + 1e-8))
        gate = np.maximum(gate, 0)
        result = filtered * gate

        return self._normalize(result)

    def _normalize(self, audio):
        """Приводим громкость к нормальному уровню."""
        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = audio / peak * 0.95
        return audio.astype(np.float32)


def normalize_for_encoder(audio):
    """
    Готовим кусок аудио для модели голоса (resemblyzer).
    Ей нужен float64 и нормальная громкость.
    """
    wav = audio.astype(np.float64)
    peak = np.abs(wav).max()
    if peak > 1e-6:
        wav = wav / peak * 0.95
    return wav
