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
        content = self._complete(
            system, json.dumps(payload, ensure_ascii=False), schema=_chunk_schema(len(chunk))
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

    def _translate_one(self, line: str, src: str, tgt: str) -> str:
        system = (
            f"Translate this single {src} subtitle line into natural, colloquial {tgt}. "
            'Return JSON {"0": "..."} mapping the key "0" to the translation.'
        )
        content = self._complete(
            system, json.dumps({"0": line}, ensure_ascii=False), schema=_chunk_schema(1)
        )
        try:
            val = json.loads(content).get("0")
            if isinstance(val, str) and val.strip():
                return val
        except (json.JSONDecodeError, AttributeError):
            pass
        logger.error("LLM per-line translation failed — keeping source text unchanged")
        return line

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

        cues = [{"i": i, "ja": ja_texts[i]} for i in range(n)]

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
                "properties": {
                    "sentence_starts": {"type": "array", "items": {"type": "integer"}}
                },
                "required": ["sentence_starts"],
                "additionalProperties": False,
            },
        }
        content = self._complete(
            system, json.dumps({"cues": cues}, ensure_ascii=False), schema=schema
        )
        try:
            raw = json.loads(content).get("sentence_starts", [])
        except (json.JSONDecodeError, AttributeError):
            raw = []

        # Sanitise: keep in-range ints, drop bools, dedupe, sort, force index 0 present.
        starts = sorted(
            {i for i in raw if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n}
        )
        if not starts or starts[0] != 0:
            starts = [0, *starts]
        return starts

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
                )
                return resp.choices[0].message.content or "{}"
            except openai.BadRequestError:
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
