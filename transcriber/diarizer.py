"""
Диаризация для моно-записи: кластеризация голосовых эмбеддингов (resemblyzer).
Стерео-телефон сюда не попадает — там канал сам по себе является спикером,
это разбирает Pipeline.
"""
import numpy as np
from collections import Counter
from scipy.spatial.distance import cosine

from transcriber.audio import normalize_for_encoder


RATE = 16000

# косинусное расстояние, выше которого заводим нового спикера.
# для телефона порог низкий — узкая полоса сближает голоса
NEW_SPEAKER_THRESHOLD = 0.50
MIN_PIECE_SECONDS = 1.0
MAX_PIECE_SECONDS = 3.0
SMOOTH_WINDOW = 5


class SpeakerDiarizer:
    def __init__(self):
        self.encoder = None
        self.ready = False

    def load(self):
        print("[спикеры] гружу модель голоса")
        from resemblyzer import VoiceEncoder
        # модель крошечная, а torch тут всё равно без поддержки Pascal — держим на CPU
        self.encoder = VoiceEncoder(device="cpu")
        self.ready = True
        print("[спикеры] модель голоса готова")

    def assign_speakers(self, audio, pieces):
        """Аудио + куски текста -> те же куски с метками SPEAKER_XX."""
        fingerprints = self._make_fingerprints(audio, pieces)
        labels = self._group_voices(fingerprints)
        labels = self._smooth(labels, SMOOTH_WINDOW)
        labels = self._fix_tiny_groups(labels, fingerprints)
        labels = self._renumber(labels)
        return self._attach_labels(pieces, labels)

    def _make_fingerprints(self, audio, pieces):
        result = []
        for piece in pieces:
            result.append(self._one_fingerprint(audio, piece.start, piece.end))
        return result

    def _one_fingerprint(self, audio, start, end):
        length = end - start
        if length < 0.1:
            return None

        # короткий кусок расширяем — на нём эмбеддинг нестабильный
        if length < MIN_PIECE_SECONDS:
            pad = (MIN_PIECE_SECONDS - length) / 2
            start = max(0.0, start - pad)
            end = min(len(audio) / RATE, end + pad)

        if end - start > MAX_PIECE_SECONDS:
            middle = (start + end) / 2
            start = middle - MAX_PIECE_SECONDS / 2
            end = middle + MAX_PIECE_SECONDS / 2

        chunk = audio[int(start * RATE):int(end * RATE)]
        if len(chunk) < int(RATE * 0.4):
            return None

        try:
            return self.encoder.embed_utterance(normalize_for_encoder(chunk))
        except Exception:
            return None

    def _group_voices(self, fingerprints):
        """Онлайн-кластеризация: сравниваем с центрами групп, иначе новая группа."""
        group_members = []
        group_centers = []
        labels = []

        for fp in fingerprints:
            if fp is None:
                labels.append(labels[-1] if labels else 0)
                continue

            if not group_centers:
                self._start_new_group(group_members, group_centers, fp)
                labels.append(0)
                continue

            distances = [cosine(fp, center) for center in group_centers]
            nearest = int(np.argmin(distances))

            if distances[nearest] < NEW_SPEAKER_THRESHOLD:
                self._add_to_group(group_members, group_centers, nearest, fp)
                labels.append(nearest)
            elif min(distances) < NEW_SPEAKER_THRESHOLD * 1.3:
                # в серой зоне лучше приклеить к ближайшему, чем плодить спикеров из шума
                self._add_to_group(group_members, group_centers, nearest, fp)
                labels.append(nearest)
            else:
                self._start_new_group(group_members, group_centers, fp)
                labels.append(len(group_centers) - 1)

        self.centers = group_centers
        return labels

    def _start_new_group(self, members, centers, fp):
        members.append([fp.copy()])
        centers.append(fp.copy())

    def _add_to_group(self, members, centers, index, fp):
        members[index].append(fp.copy())
        centers[index] = np.mean(members[index], axis=0)

    def _smooth(self, labels, window):
        """Медианный фильтр по меткам — гасит одиночные перескоки."""
        if window < 2 or len(labels) < window:
            return labels
        result = list(labels)
        half = window // 2
        for i in range(len(labels)):
            low = max(0, i - half)
            high = min(len(labels), i + half + 1)
            neighbours = labels[low:high]
            result[i] = max(set(neighbours), key=neighbours.count)
        return result

    def _fix_tiny_groups(self, labels, fingerprints):
        """Группы на 1-2 реплики приклеиваем к ближайшей большой."""
        counts = Counter(labels)
        tiny = {label for label, count in counts.items() if count < 3}
        if not tiny or len(counts) <= 1:
            return labels

        big_groups = [i for i in range(len(self.centers)) if i not in tiny]
        if not big_groups:
            return labels

        result = list(labels)
        for i, label in enumerate(labels):
            if label not in tiny:
                continue
            if fingerprints[i] is not None:
                distances = [cosine(fingerprints[i], self.centers[g]) for g in big_groups]
                result[i] = big_groups[int(np.argmin(distances))]
            else:
                result[i] = big_groups[0]
        return result

    def _renumber(self, labels):
        """Перенумеровываем группы по порядку появления."""
        mapping = {}
        next_id = 0
        result = []
        for label in labels:
            if label not in mapping:
                mapping[label] = next_id
                next_id += 1
            result.append(mapping[label])
        return result

    def _attach_labels(self, pieces, labels):
        result = []
        for piece, label in zip(pieces, labels):
            item = piece.to_dict()
            item["speaker"] = f"SPEAKER_{label:02d}"
            result.append(item)
        return result
