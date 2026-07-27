import librosa
import numpy as np
from scipy import signal


TARGET_RATE = 16000
PHONE_RATE = 8000


class AudioData:
    def __init__(self, channels, is_phone):
        self.channels = channels        # [каналы x отсчёты]
        self.is_phone = is_phone

    @property
    def channel_count(self):
        return self.channels.shape[0]

    def get_channel(self, index):
        return self.channels[index]

    def to_mono(self):
        return self.channels.mean(axis=0)


class AudioLoader:
    def load(self, path):
        # sr=None — не трогаем частоту на этом шаге, mono=False — все каналы
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

        return AudioData(channels, is_phone)

    def _prepare_normal(self, waveform, rate):
        if rate != TARGET_RATE:
            waveform = librosa.resample(waveform, orig_sr=rate, target_sr=TARGET_RATE)
        return waveform.astype(np.float32)

    def _prepare_phone(self, waveform):
        cleaned = []
        for i in range(waveform.shape[0]):
            cleaned.append(self._upsample_phone(waveform[i]))
        return np.stack(cleaned)

    def _upsample_phone(self, audio):
        clean = self._clean_phone_noise(audio, PHONE_RATE)
        upsampled = signal.resample_poly(
            clean,
            2, 1,                       # 8000 -> 16000
            window=("kaiser", 5.0),
            padtype="line",
        )
        return upsampled.astype(np.float32)

    def _clean_phone_noise(self, audio, rate):
        nyquist = rate / 2
        low = 300 / nyquist
        high = min(3400 / nyquist, 0.95)

        # полоса телефонного голоса 300-3400 Гц, всё остальное режем
        sos = signal.butter(4, [low, high], btype="band", output="sos")
        filtered = signal.sosfilt(sos, audio)

        # мягкий гейт: давим то, что тише 10% от RMS
        rms = np.sqrt(np.mean(filtered ** 2))
        threshold = rms * 0.1
        gate = np.tanh((np.abs(filtered) - threshold) / (threshold + 1e-8))
        gate = np.maximum(gate, 0)
        result = filtered * gate

        return self._normalize(result)

    def _normalize(self, audio):
        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = audio / peak * 0.95
        return audio.astype(np.float32)


def normalize_for_encoder(audio):
    """resemblyzer ждёт float64 с нормализованной громкостью."""
    wav = audio.astype(np.float64)
    peak = np.abs(wav).max()
    if peak > 1e-6:
        wav = wav / peak * 0.95
    return wav
