"""OpenAI LLM-backed translator and Japanese sentence segmenter."""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-5.4-mini"
# Lines per API call: small enough to keep output bounded and alignment reliable,
# large enough to amortise request overhead.
_BATCH_SIZE = 40
# Concurrent batches: throughput without tripping rate limits (retry handles the rest).
_DEFAULT_CONCURRENCY = 5
# sentence_starts windowing (cues per request, and context cues re-read from the last one)
_SENTENCE_WINDOW = 80
_SENTENCE_OVERLAP = 10


def _batch_size_for_model(model: str) -> int:
    """Keep Groq-hosted structured responses below its JSON completion limit."""
    return 20 if model.startswith(("qwen/", "openai/gpt-oss")) else _BATCH_SIZE


def _model_kwargs(model: str) -> dict:
    """Extra request options per model family.

    gpt-oss is a reasoning model: its reasoning counts against the completion budget, so with
    the default budget a 20-line JSON answer gets truncated. Low effort and a larger budget fix it.
    """
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low", "max_completion_tokens": 16000}
    return {}


@dataclass
class _Translated:
    """Result wrapper: callers only ever read ``.text``."""

    text: str


def _lang_name(code: str | None) -> str:
    """Map a language code (EN-US, ES, JA) to an English language name for the prompt."""
    c = (code or "").upper()
    if c.startswith("EN"):
        return "English"
    if c.startswith("ES"):
        return "Spanish"
    if c.startswith("JA"):
        return "Japanese"
    return code or "the target language"


def _target_variety(tgt: str) -> str:
    """Spanish defaults to neutral Latin American; other targets keep their plain name."""
    return "neutral Latin American Spanish" if tgt == "Spanish" else tgt


def _is_failed_generation(error: Exception) -> bool:
    """The provider rejected the model's own output as invalid JSON (Groq json_validate_failed)."""
    return "json_validate_failed" in str(error)


def _translation_system_prompt(src: str, tgt: str) -> str:
    variety = _target_variety(tgt)
    return (
        f"You are a professional subtitle translator. Translate each {src} subtitle "
        f"line into natural, colloquial {variety} that matches spoken register. Preserve "
        "the speaker, addressee, and level of formality. You receive a JSON object mapping "
        "an index to a line. Return a JSON object with the SAME keys, each mapped to the "
        "translation of that line only; neighbouring lines may be read as context to "
        "resolve an incomplete thought. Do NOT merge, split, drop, or add keys, and do not "
        "add notes or romanization."
    )


def _chunk_schema(n: int) -> dict:
    """Strict JSON schema forcing exactly the keys '0'..'n-1', each a string."""
    keys = [str(i) for i in range(n)]
    return {
        "name": "subtitle_translations",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {k: {"type": "string"} for k in keys},
            "required": keys,
            "additionalProperties": False,
        },
    }


class OpenAITranslator:
    """Translator backed by the OpenAI chat API."""

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        base_url: str | None = None,
        concurrency: int = _DEFAULT_CONCURRENCY,
    ):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._concurrency = max(1, concurrency)
        # Structured Outputs (json_schema) is the primary alignment guarantee. Some
        # OpenAI-compatible proxies don't support it; the first rejection flips this
        # off for the rest of the run and we fall back to json_object + the key-guard.
        self._supports_schema = True
        # Tokens the provider reported, so a run can say what it actually used.
        self.usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _count_usage(self, resp) -> None:
        totals = self.__dict__.setdefault(
            "usage", {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
        )
        usage = getattr(resp, "usage", None)
        totals["requests"] += 1
        if usage is not None:
            totals["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            totals["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0

    def translate_text(self, text, source_lang=None, target_lang=None, **_):
        """Translate a string or list of strings. Returns _Translated or list thereof.

        A str yields one result; a list yields a list of the same length and order.
        """
        single = isinstance(text, str)
        items = [text] if single else list(text)
        if not items:
            return []
        out = self._translate(items, source_lang, target_lang)
        results = [_Translated(t) for t in out]
        return results[0] if single else results

    def _translate(self, items: list[str], source_lang, target_lang) -> list[str]:
        src = _lang_name(source_lang) if source_lang else "Japanese"
        tgt = _lang_name(target_lang)
        batch_size = _batch_size_for_model(self._model)
        chunks = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
        if len(chunks) == 1:
            return self._translate_chunk(chunks[0], src, tgt)

        # Translate batches concurrently, then reassemble in original order.
        done: list[list[str] | None] = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=self._concurrency) as ex:
            futures = {
                ex.submit(self._translate_chunk, chunk, src, tgt): idx
                for idx, chunk in enumerate(chunks)
            }
            for fut in as_completed(futures):
                done[futures[fut]] = fut.result()
        return [line for chunk in done for line in chunk]  # type: ignore[union-attr]

    def _translate_chunk(self, chunk: list[str], src: str, tgt: str) -> list[str]:
        import openai

        payload = {str(i): line for i, line in enumerate(chunk)}
        system = _translation_system_prompt(src, tgt)
        try:
            content = self._complete(
                system, json.dumps(payload, ensure_ascii=False), schema=_chunk_schema(len(chunk))
            )
        except openai.BadRequestError as e:
            if not _is_failed_generation(e):
                raise
            # The model emitted invalid JSON for this batch (Groq's json_validate_failed, which
            # is stochastic). Losing one batch must not lose a whole chapter: halve and retry,
            # down to single lines.
            if len(chunk) == 1:
                return [self._translate_one(chunk[0], src, tgt)]
            mid = len(chunk) // 2
            logger.warning(
                f"Provider failed to generate JSON for a {len(chunk)}-line batch — "
                f"retrying as {mid} + {len(chunk) - mid}"
            )
            return self._translate_chunk(chunk[:mid], src, tgt) + self._translate_chunk(
                chunk[mid:], src, tgt
            )
        try:
            out = json.loads(content)
        except json.JSONDecodeError:
            out = {}
        if not isinstance(out, dict):
            out = {}

        expected = {str(i) for i in range(len(chunk))}
        if set(out.keys()) != expected:
            logger.warning(
                f"LLM returned {len(out)} keys for a {len(chunk)}-line chunk "
                "(key-set mismatch) — retranslating the whole chunk line-by-line"
            )
            return [self._translate_one(line, src, tgt) for line in chunk]

        result: list[str] = [""] * len(chunk)
        blanks = []
        for i in range(len(chunk)):
            val = out.get(str(i))
            if isinstance(val, str) and val.strip():
                result[i] = val
            else:
                blanks.append(i)

        if blanks:
            logger.warning(
                f"LLM left {len(blanks)}/{len(chunk)} line(s) blank — "
                "retranslating those individually"
            )
            for i in blanks:
                result[i] = self._translate_one(chunk[i], src, tgt)
        return result

    def _translate_one(self, line: str, src: str, tgt: str, attempts: int = 3) -> str:
        """Translate one line; "" when the provider can't, so callers drop it as untranslated.

        Returning the source text instead would store Japanese as the "translation".
        """
        import openai

        system = (
            f"Translate this single {src} subtitle line into natural, colloquial "
            f"{_target_variety(tgt)}. "
            'Return JSON {"0": "..."} mapping the key "0" to the translation.'
        )
        for _ in range(attempts):
            try:
                content = self._complete(
                    system, json.dumps({"0": line}, ensure_ascii=False), schema=_chunk_schema(1)
                )
            except openai.BadRequestError as e:
                if not _is_failed_generation(e):
                    raise
                continue
            try:
                val = json.loads(content).get("0")
                if isinstance(val, str) and val.strip():
                    return val
            except (json.JSONDecodeError, AttributeError):
                pass
        logger.error("LLM per-line translation failed — leaving the line untranslated")
        return ""

    def sentence_starts(self, ja_texts: list[str]) -> list[int]:
        """Return the cue indices that begin a new sentence.

        Groups unpunctuated Japanese subtitle cues into sentence-level segments using the
        Japanese alone — no English is consulted, so precision never depends on a
        translation. The model only emits cut positions — never text — so a mistake can at
        worst mis-place a boundary, never corrupt content, timing, or language.
        """
        n = len(ja_texts)
        if n <= 1:
            return [0] if n else []

        system = (
            "You group fragments of spoken Japanese into sentences. You receive an array of "
            "numbered cues, each with Japanese text 'ja'. The cues are consecutive fragments "
            "of continuous speech with no punctuation. Return the indices of the cues that "
            "START a new sentence — i.e. the previous cue ended one.\n"
            "A cue ENDS a sentence when its final form is sentence-final: a predicate in "
            "plain or polite form (e.g. 〜だ/〜です/〜ます/〜た/〜だった/〜ない), the explanatory "
            "〜んだ/〜んです, or any of these followed by a sentence-final particle "
            "(ね/よ/な/わ/か/の/さ/ぞ/もんね/かな/でしょ). The NEXT cue then starts a "
            "new sentence.\n"
            "A cue does NOT end a sentence when it trails off on a connective or particle: "
            "〜て/〜で/〜けど/〜が/〜から/〜ので/〜し/〜たり, or a dangling は/を/に/の/と/へ — "
            "the next cue continues it. Use meaning for ambiguous cases (〜けど, 〜もんね can "
            "be either). Index 0 always starts a sentence. Return JSON "
            '{"sentence_starts": [0, ...]} with strictly increasing indices.'
        )
        schema = {
            "name": "sentence_starts",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"sentence_starts": {"type": "array", "items": {"type": "integer"}}},
                "required": ["sentence_starts"],
                "additionalProperties": False,
            },
        }
        # Some providers (e.g. Qwen on Groq) can't return a whole episode's analysis in one
        # structured response, so cues go in bounded windows. Each window after the first
        # re-reads the previous _SENTENCE_OVERLAP cues as context and only decides cuts in
        # its own region: the model always marks a window's first cue as a start, and
        # without the overlap every window boundary would split a sentence.
        starts: set[int] = set()
        region_start = 0
        while region_start < n:
            win_start = max(0, region_start - _SENTENCE_OVERLAP)
            win_end = min(n, win_start + _SENTENCE_WINDOW)
            chunk = [{"i": i - win_start, "ja": ja_texts[i]} for i in range(win_start, win_end)]
            content = self._complete(
                system, json.dumps({"cues": chunk}, ensure_ascii=False), schema=schema
            )
            try:
                raw = json.loads(content).get("sentence_starts", [])
            except (json.JSONDecodeError, AttributeError):
                raw = []
            starts.update(
                win_start + i
                for i in raw
                if isinstance(i, int)
                and not isinstance(i, bool)
                and region_start <= win_start + i < win_end
            )
            region_start = win_end

        ordered_starts = sorted(starts)
        if not ordered_starts or ordered_starts[0] != 0:
            ordered_starts = [0, *ordered_starts]
        return ordered_starts

    def _complete(
        self,
        system: str,
        user: str,
        *,
        schema: dict | None = None,
        attempts: int = 4,
        base_delay: float = 2.0,
    ) -> str:
        import openai

        def _response_format() -> dict:
            if schema is not None and self._supports_schema:
                return {"type": "json_schema", "json_schema": schema}
            return {"type": "json_object"}

        for attempt in range(1, attempts + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format=_response_format(),
                    temperature=0.2,
                    **_model_kwargs(self._model),
                )
                self._count_usage(resp)
                return resp.choices[0].message.content or "{}"
            except openai.BadRequestError:
                # Any 400 while requesting json_schema (unsupported format, or a
                # provider's structured-output token limit) falls back to json_object.
                if self._supports_schema and schema is not None:
                    logger.warning(
                        "Endpoint rejected json_schema response_format — falling back "
                        "to json_object for the rest of this run"
                    )
                    self._supports_schema = False
                    continue
                raise
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ) as e:
                if attempt == attempts:
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"OpenAI transient error ({type(e).__name__}, attempt {attempt}/{attempts}) — "
                    f"retrying in {delay:.0f}s"
                )
                time.sleep(delay)
        return "{}"  # unreachable: the final attempt either returns or raises


def get_openai_translator() -> OpenAITranslator | None:
    """Build an OpenAITranslator from env (OPENAI_API_KEY required), or None if unset."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set — translation disabled")
        return None
    model = os.getenv("OPENAI_TRANSLATE_MODEL", _DEFAULT_MODEL)
    base_url = os.getenv("OPENAI_BASE_URL")  # for proxies / local OpenAI-compatible servers
    try:
        concurrency = int(os.getenv("OPENAI_TRANSLATE_CONCURRENCY", _DEFAULT_CONCURRENCY))
    except ValueError:
        concurrency = _DEFAULT_CONCURRENCY
    logger.info(f"Using OpenAI translator (model: {model}, concurrency: {concurrency})")
    return OpenAITranslator(api_key, model, base_url=base_url, concurrency=concurrency)
