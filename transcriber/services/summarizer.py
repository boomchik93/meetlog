import json
import os
import re

from transcriber.config.settings import settings


# пустой результат отдаём когда что то пошло не так
EMPTY_RESULT = {
    "title": "",
    "summary": "",
    "topics": [],
    "decisions": [],
    "action_items": [],
    "risks": [],
}

RESULT_FIELDS = ["title", "summary", "topics", "decisions", "action_items", "risks"]

# у кириллицы с метками SPEAKER ~2.8 символа на токен, а не 4
CHARS_PER_TOKEN = 2.8
RESERVE_PROMPT_TOKENS = 1200   # системный промпт + обёртка
RESERVE_ANSWER_TOKENS = 4096   # ответ модели (подняли, чтобы влезал длинный конспект)

# потолок ответа модели за один вызов
ANSWER_MAX_TOKENS = 4096
# на длинной записи один вызов не тянет: режем транскрипт на куски по контексту,
# каждый пересказываем отдельно (map), потом сводим частичные конспекты (reduce)
CHUNK_OVERLAP_CHARS = 400   # нахлёст между кусками, чтобы не рвать мысль на границе


def _max_transcript_chars():
    budget_tokens = settings.llm_context - RESERVE_PROMPT_TOKENS - RESERVE_ANSWER_TOKENS
    if budget_tokens < 1000:
        budget_tokens = 1000
    return int(budget_tokens * CHARS_PER_TOKEN)


CORRECTION_PROMPT = """Ты — редактор автоматических расшифровок телефонных разговоров.
Тебе дают фрагмент расшифровки (распознавание речи) с ошибками: искажённые слова, термины, опечатки, бессмысленные обрывки.

Задача: исправить явные ошибки распознавания, опираясь на смысл и контекст разговора. Восстанови правильные термины и слова там, где модель ослышалась.

Правила:
- Сохрани разметку спикеров (SPEAKER_00:, SPEAKER_01: и т.д.) и порядок реплик.
- Не добавляй, не выкидывай и не объединяй реплики. Не пересказывай.
- Чини только слова. Очевидно мусорные обрывки можно убрать.
- Верни ТОЛЬКО исправленный текст, без пояснений и без markdown."""


# на сколько символов резать транскрипт для коррекции (чтобы влезал в контекст)
CORRECTION_CHUNK_CHARS = 5000


SPEAKER_NAMES_PROMPT = """Проанализируй транскрипцию переговоров и определи реальные имена участников.

В транскрипции каждая реплика помечена меткой SPEAKER_XX. Твоя задача — по контексту разговора (обращениям по имени, представлениям, самопредставлениям, упоминаниям) определить, как зовут каждого спикера.

Верни СТРОГО JSON без каких-либо пояснений, только объект вида:
{"SPEAKER_00": "Полное имя как прозвучало в разговоре", "SPEAKER_01": null}

Правила:
- Если спикер называет другого по имени — это имя того другого спикера, а не его собственное.
- Если имя спикера нигде не упоминается — верни null.
- Не придумывай имена. Только то, что явно прозвучало."""


SYSTEM_PROMPT = """Ты — аналитик деловых переговоров. Разбери транскрибацию обсуждения и верни структурированный конспект.

Цель: зафиксировать ВСЕ обсуждённые темы без потерь. Тем может быть много — лучше лишняя тема, чем потерянная. Каждое решение, задачу, риск — с конкретикой и именами из разговора.

Объём конспекта должен отражать объём разговора: длинное обсуждение — развёрнутый разбор с множеством тем и тезисов, а не пара общих фраз. Не сжимай и не обобщай в ущерб деталям.

ФОРМАТ ОТВЕТА (СТРОГО JSON):
{
  "title": "Короткая тема всего обсуждения, 3-6 слов, для названия папки",
  "summary": "Связное и подробное описание сути обсуждения: по абзацу на каждый крупный блок разговора, со всеми ключевыми деталями",
  "topics": [
    {
      "title": "Название темы",
      "category": "technical|organizational|financial|other",
      "points": ["Ключевой тезис с конкретикой", "Ещё тезис"]
    }
  ],
  "decisions": [
    {"decision": "Что решено", "context": "Почему / при каком условии", "responsible": "Имя ответственного"}
  ],
  "action_items": [
    {"action": "Что нужно сделать", "responsible": "Имя", "deadline": "дд.мм.гггг или 'Не указано'"}
  ],
  "risks": [
    {"risk": "Описание риска", "impact": "Последствие, если риск реализуется"}
  ]
}

ПРАВИЛА:
- Сохраняй ВСЕ темы, даже второстепенные.
- category — строго одно из: technical, organizational, financial, other.
- Имена ответственных — как звучат в разговоре, без обобщений вроде «команда».
- Нет данных для поля — пиши "Не указано", не выдумывай.
- Пустая секция (нет решений / задач / рисков) — оставляй [].
- Только JSON. Без markdown, пояснений и вводного текста.
- Язык ответа — русский. Тон деловой и безличный."""


# промпт свода: на входе несколько частичных конспктов подряд идущих кусков
# одного разговора, на выходе — один цельный конспект по всему разговору
REDUCE_PROMPT = """Ты — аналитик деловых переговоров. Тебе дают несколько частичных конспектов — это разборы идущих подряд кусков ОДНОГО длинного разговора.

Задача: свести их в один цельный конспект по всему разговору.

Правила свода:
- Объедини одинаковые и близкие темы из разных кусков в одну, не теряя ни одного тезиса.
- Сохрани ВСЕ решения, задачи и риски из всех кусков. Дубли — схлопни, разное — сохрани всё.
- summary собери заново по всему разговору: подробно, по абзацу на крупный блок. Не сокращай до пары фраз.
- title — общая тема всего разговора.
- Имена, цифры, сроки — как в исходных конспектах, ничего не выдумывай.

ФОРМАТ ОТВЕТА — тот же строгий JSON, что и у частичных конспектов (title, summary, topics, decisions, action_items, risks). Только JSON, без markdown и пояснений. Язык — русский."""


class Summarizer:
    def __init__(self):
        self.llm = None
        self.ready = False

    def load(self):
        if not settings.llm_enabled:
            print("[пересказ] выключен в конфиге")
            return

        try:
            from llama_cpp import Llama
        except ImportError:
            print("[пересказ] llama-cpp-python не установлен, пересказ выключен")
            return

        import os
        if not os.path.exists(settings.llm_path):
            print(f"[пересказ] модель не найдена {settings.llm_path}")
            return

        # на CPU 7B-модель не уложится в бюджет времени — предупреждаем
        if settings.llm_gpu_layers == 0:
            print("[пересказ] ВНИМАНИЕ: LLM на CPU (llm_gpu_layers=0), "
                  "пересказ будет медленным. Проверь сборку llama-cpp с CUDA")

        print(f"[пересказ] гружу модель из {settings.llm_path}, "
              f"gpu_layers={settings.llm_gpu_layers}, ctx={settings.llm_context}")
        self.llm = Llama(
            model_path=settings.llm_path,
            n_ctx=settings.llm_context,
            n_threads=settings.llm_threads,
            n_gpu_layers=settings.llm_gpu_layers,
            verbose=False,
        )
        self.ready = True
        print("[пересказ] модель готова")

    def summarize(self, pieces, speakers=None, corrected_text=None):
        if not self.ready:
            return dict(EMPTY_RESULT, error="LLM не загружена")

        # если есть исправленный транскрипт — саммари строим по нему
        transcript = corrected_text or self._build_transcript(pieces)
        if len(transcript) < 50:
            return dict(EMPTY_RESULT, error="Транскрипт слишком короткий")

        limit = _max_transcript_chars()
        # короткий разговор влезает в контекст целиком — один проход
        if len(transcript) <= limit:
            answer = self._ask_model(transcript)
            return self._parse_answer(answer)

        # длинный разговор не влезает: режем на куски, пересказываем каждый
        # (map), затем сводим частичные конспекты в один (reduce). Так объём
        # итога растёт с длиной записи и ничего не теряется за обрезкой.
        return self._map_reduce(transcript, limit)

    # --- map-reduce для длинных записей ---
    def _map_reduce(self, transcript, limit):
        chunks = self._split_transcript(transcript, limit)
        print(f"[пересказ] длинный транскрипт, кусков для свода: {len(chunks)}")

        partials = []
        for i, chunk in enumerate(chunks):
            answer = self._ask_model(chunk)
            parsed = self._parse_answer(answer)
            if not parsed.get("error"):
                partials.append(parsed)
            else:
                print(f"[пересказ] кусок {i + 1} не разобран: {parsed.get('error')}")

        if not partials:
            return dict(EMPTY_RESULT, error="Ни один кусок не удалось пересказать")
        if len(partials) == 1:
            return partials[0]

        return self._reduce_partials(partials)

    # режем транскрипт на куски ~limit символов, не разрывая реплики, с нахлёстом
    def _split_transcript(self, transcript, limit):
        # запас под нахлёст, чтобы кусок с нахлёстом всё равно влезал в контекст
        target = max(1000, limit - CHUNK_OVERLAP_CHARS)
        lines = transcript.split("\n")
        chunks, cur = [], ""
        for line in lines:
            if cur and len(cur) + len(line) > target:
                chunks.append(cur.strip())
                # начинаем следующий кусок с хвоста предыдущего (нахлёст)
                cur = cur[-CHUNK_OVERLAP_CHARS:]
            cur += line + "\n"
        if cur.strip():
            chunks.append(cur.strip())
        return chunks

    # сводим частичные конспекты в один финальный
    def _reduce_partials(self, partials):
        blocks = []
        for i, part in enumerate(partials):
            blocks.append(f"=== Конспект куска {i + 1} ===\n"
                          + json.dumps(part, ensure_ascii=False, indent=2))
        joined = "\n\n".join(blocks)

        # сам свод тоже может не влезть в контекст — тогда сводим по частям
        limit = _max_transcript_chars()
        if len(joined) > limit:
            mid = len(partials) // 2
            left = self._reduce_partials(partials[:mid])
            right = self._reduce_partials(partials[mid:])
            return self._reduce_partials([left, right])

        user_message = ("Своди частичные конспекты одного разговора в один:\n\n"
                        + joined)
        prompt = self._build_prompt(REDUCE_PROMPT, user_message)
        answer = self._run_llm(prompt)
        parsed = self._parse_answer(answer)
        # если свод сорвался — отдаём хотя бы первый кусок, не пустоту
        return parsed if not parsed.get("error") else partials[0]

    # --- LLM-коррекция транскрипта (правит ошибки распознавания) ---
    def correct_transcript(self, pieces):
        if not self.ready:
            return None
        transcript = self._build_transcript_with_labels(pieces)
        if len(transcript) < 50:
            return None

        chunks = self._split_for_correction(transcript)
        print(f"[коррекция] чиню транскрипт LLM, кусков: {len(chunks)}")
        fixed = []
        for i, chunk in enumerate(chunks):
            fixed.append(self._correct_chunk(chunk))
        return "\n".join(p for p in fixed if p).strip() or None

    def _split_for_correction(self, text):
        # режем по переводам строк, не разрывая реплики, ~CHUNK_CHARS на кусок
        lines = text.split("\n")
        chunks, cur = [], ""
        for line in lines:
            if cur and len(cur) + len(line) > CORRECTION_CHUNK_CHARS:
                chunks.append(cur)
                cur = ""
            cur += (line + "\n")
        if cur.strip():
            chunks.append(cur)
        return chunks

    def _correct_chunk(self, chunk):
        prompt = self._build_prompt(CORRECTION_PROMPT, f"Фрагмент расшифровки:\n\n{chunk}")
        try:
            output = self.llm(
                prompt,
                max_tokens=2048,
                temperature=0.0,
                top_p=0.9,
                stop=["<|im_end|>", "<|endoftext|>"],
                echo=False,
            )
            return self._strip_markdown(output["choices"][0]["text"].strip())
        except Exception as error:
            print(f"[коррекция] ошибка куска: {error}")
            return chunk

    def identify_speakers(self, pieces):
        if not self.ready:
            return {}

        transcript = self._build_transcript_with_labels(pieces)
        if len(transcript) < 50:
            return {}

        transcript = self._clip(transcript)
        answer = self._ask_speaker_names(transcript)
        return self._parse_speaker_names(answer, pieces)

    def _clip(self, transcript):
        limit = _max_transcript_chars()
        if len(transcript) > limit:
            return transcript[:limit] + "\n[...текст обрезан...]"
        return transcript

    # собираем реплики в один текст помечая смену говорящего
    def _build_transcript(self, pieces):
        lines = []
        last_speaker = None
        for piece in pieces:
            speaker = piece.get("speaker", "Спикер")
            text = piece.get("text", "").strip()
            if not text:
                continue
            if speaker != last_speaker:
                lines.append(f"\n{speaker}: {text}")
                last_speaker = speaker
            else:
                lines.append(text)
        return " ".join(lines).strip()

    # тот же транскрипт но с метками SPEAKER_XX для определения имён
    def _build_transcript_with_labels(self, pieces):
        lines = []
        last_speaker = None
        for piece in pieces:
            speaker = piece.get("speaker", "SPEAKER_00")
            text = piece.get("text", "").strip()
            if not text:
                continue
            if speaker != last_speaker:
                lines.append(f"\n{speaker}: {text}")
                last_speaker = speaker
            else:
                lines[-1] = lines[-1] + " " + text
        return "\n".join(lines).strip()

    def _ask_speaker_names(self, transcript):
        user_message = f"Определи имена участников разговора:\n\n{transcript}"
        prompt = self._build_prompt(SPEAKER_NAMES_PROMPT, user_message)
        try:
            output = self.llm(
                prompt,
                max_tokens=256,
                temperature=0.0,
                top_p=0.9,
                stop=["<|im_end|>", "<|endoftext|>"],
                echo=False,
            )
            return output["choices"][0]["text"].strip()
        except Exception as error:
            print(f"[пересказ] ошибка определения имён {error}")
            return "{}"

    def _parse_speaker_names(self, raw, pieces):
        raw = self._strip_markdown(raw)
        match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if match:
            raw = match.group()
        data = self._try_load_json(raw)
        if not isinstance(data, dict):
            print(f"[пересказ] не удалось распознать имена спикеров {raw[:200]}")
            return {}
        known = {p["speaker"] for p in pieces if "speaker" in p}
        return {k: (v if v and isinstance(v, str) else None) for k, v in data.items() if k in known}

    def _ask_model(self, transcript):
        user_message = f"Составь структурированный пересказ следующего разговора:\n\n{transcript}"
        prompt = self._build_prompt(SYSTEM_PROMPT, user_message)
        return self._run_llm(prompt)

    # один вызов модели для пересказа/свода (общий код для map и reduce)
    def _run_llm(self, prompt):
        try:
            output = self.llm(
                prompt,
                max_tokens=ANSWER_MAX_TOKENS,
                temperature=0.1,
                top_p=0.9,
                stop=["<|im_end|>", "<|endoftext|>"],
                echo=False,
            )
            return output["choices"][0]["text"].strip()
        except Exception as error:
            return f'{{"error": "{error}"}}'

    # формат сообщений который понимает Qwen
    def _build_prompt(self, system, user):
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{user}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def _parse_answer(self, raw):
        raw = self._strip_markdown(raw)

        data = self._try_load_json(raw)
        if data is None:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = self._try_load_json(match.group())

        if data is None:
            return dict(EMPTY_RESULT, error="Не удалось разобрать JSON", raw=raw[:500])

        if not any(field in data for field in RESULT_FIELDS):
            return dict(EMPTY_RESULT, error="Неожиданная структура JSON", raw=raw[:500])

        for field in RESULT_FIELDS:
            data.setdefault(field, "" if field in ("summary", "title") else [])
        return data

    def _strip_markdown(self, text):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        return text.strip()

    def _try_load_json(self, text):
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
