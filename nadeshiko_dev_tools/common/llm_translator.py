"""LLM-backed translator with a DeepL-compatible interface."""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"
# Lines per API call: small enough to keep output bounded and alignment reliable,
# large enough to amortise request overhead.
_BATCH_SIZE = 40
# Concurrent batches: throughput without tripping rate limits (retry handles the rest).
_DEFAULT_CONCURRENCY = 5


@dataclass
class _Translated:
    """Mimics DeepL's TextResult: callers only ever read ``.text``."""

    text: str


def _lang_name(code: str | None) -> str:
    """Map a DeepL-style code (EN-US, ES, JA) to an English language name for the prompt."""
    c = (code or "").upper()
    if c.startswith("EN"):
        return "English"
    if c.startswith("ES"):
        return "Spanish"
    if c.startswith("JA"):
        return "Japanese"
    return code or "the target language"


class OpenAITranslator:
    """DeepL-compatible translator backed by the OpenAI chat API."""

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

    def translate_text(self, text, source_lang=None, target_lang=None, **_):
        """Translate a string or list of strings. Returns _Translated or list thereof.

        Matches deepl.Translator.translate_text: a str yields one result, a list yields a
        list of the same length and order.
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
        chunks = [items[i : i + _BATCH_SIZE] for i in range(0, len(items), _BATCH_SIZE)]
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
        payload = {str(i): line for i, line in enumerate(chunk)}
        system = (
            f"You are a professional subtitle translator. Translate each {src} subtitle "
            f"line into natural, colloquial {tgt} that matches spoken register. You receive "
            "a JSON object mapping an index to a line. Return a JSON object with the SAME "
            "keys, each mapped to the translation of that line. Translate each line "
            "independently; do NOT merge, split, drop, or add keys, and do not add notes "
            "or romanization."
        )
        content = self._complete(system, json.dumps(payload, ensure_ascii=False))
        try:
            out = json.loads(content)
        except json.JSONDecodeError:
            out = {}
        if not isinstance(out, dict):
            out = {}

        result: list[str | None] = [None] * len(chunk)
        missing = []
        for i in range(len(chunk)):
            val = out.get(str(i))
            if isinstance(val, str) and val.strip():
                result[i] = val
            else:
                missing.append(i)

        if missing:
            logger.warning(
                f"LLM left {len(missing)}/{len(chunk)} line(s) unmapped — "
                "retranslating only those individually"
            )
            for i in missing:
                result[i] = self._translate_one(chunk[i], src, tgt)
        return result  # type: ignore[return-value]

    def _translate_one(self, line: str, src: str, tgt: str) -> str:
        system = (
            f"Translate this single {src} subtitle line into natural, colloquial {tgt}. "
            'Return JSON {"0": "..."} mapping the key "0" to the translation.'
        )
        content = self._complete(system, json.dumps({"0": line}, ensure_ascii=False))
        try:
            val = json.loads(content).get("0")
            if isinstance(val, str) and val.strip():
                return val
        except (json.JSONDecodeError, AttributeError):
            pass
        logger.error("LLM per-line translation failed — keeping source text unchanged")
        return line

    def _complete(
        self, system: str, user: str, *, attempts: int = 4, base_delay: float = 2.0
    ) -> str:
        import openai

        for attempt in range(1, attempts + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
                return resp.choices[0].message.content or "{}"
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
        logger.warning("TRANSLATOR=openai but OPENAI_API_KEY is not set — translation disabled")
        return None
    model = os.getenv("OPENAI_TRANSLATE_MODEL", _DEFAULT_MODEL)
    base_url = os.getenv("OPENAI_BASE_URL")  # for proxies / local OpenAI-compatible servers
    try:
        concurrency = int(os.getenv("OPENAI_TRANSLATE_CONCURRENCY", _DEFAULT_CONCURRENCY))
    except ValueError:
        concurrency = _DEFAULT_CONCURRENCY
    logger.info(f"Using OpenAI translator (model: {model}, concurrency: {concurrency})")
    return OpenAITranslator(api_key, model, base_url=base_url, concurrency=concurrency)
