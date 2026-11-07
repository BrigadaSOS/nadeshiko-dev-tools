"""Book-aware EN/ES translation for audiobooks: glossary, echo check, detection and repair.

The generic subtitle translator translates each language in its own call, keyed by line index. On a
light novel that went wrong in ways no key check catches: a line got its neighbour's translation
(the batch shifted by one), names were invented, characters changed gender, and JSON debris such as
``"}`` or ``// Note: ...`` ended up in the text. This translator, used by process-audiobook:

* asks for EN and ES of each line in one call, with the book's glossary (names, genders, terms);
* has the model copy the first characters of the Japanese line it translated (``src``), so an
  answer that belongs to another line is rejected instead of stored (the echo check);
* runs a detector over every result (``problems``) and retranslates the flagged lines with their
  neighbours as context, keeping a new translation only when it has fewer problems;
* reports what is still flagged, for a human to look at.
"""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from nadeshiko_dev_tools.common.llm_translator import _model_kwargs

logger = logging.getLogger(__name__)

BATCH_SIZE = 20
CONTEXT_LINES = 2
REPAIR_PASSES = 2
DEFAULT_MODEL = "openai/gpt-oss-120b"

_CJK = re.compile(r"[぀-ヿ一-鿿]")
_DEBRIS = re.compile(r"//|\bNote:|[{}]|\\n|```|['\"]\]\s*$|\[\s*$|Translation:|Traducción:")
_REPEAT = re.compile(r"([^\s.…ー─-])\1{7,}|(\b\w+\b)(?:\W+\2\b){3,}", re.IGNORECASE)
_QUOTED = re.compile(r"[«“\"][^«»“”\"]+[»”\"]")


@dataclass
class Glossary:
    """From book.json ``glossary``: ``names`` maps a Japanese name to its romaji (checked in every
    translation that contains it); ``notes`` are free lines for the prompt (genders, roles, terms).
    """

    names: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_book(cls, book: dict) -> "Glossary":
        raw = book.get("glossary") or {}
        names = {str(k): str(v) for k, v in (raw.get("names") or {}).items() if k and v}
        notes = [str(n) for n in (raw.get("notes") or []) if n]
        return cls(names=names, notes=notes)

    def prompt(self) -> str:
        if not self.names and not self.notes:
            return ""
        lines = ["Glossary (always use these romanizations and genders):"]
        lines += [f"- {ja} = {romaji}" for ja, romaji in self.names.items()]
        lines += [f"- {note}" for note in self.notes]
        return "\n".join(lines)


def problems(ja: str, text: str, glossary: Glossary | None = None) -> list[str]:
    """What looks wrong with one translation of one Japanese line (empty list: nothing found)."""
    if not text or not text.strip():
        return ["empty"]
    found = []
    if _DEBRIS.search(text):
        found.append("debris")
    if _CJK.search(text):
        found.append("japanese-left")
    if _REPEAT.search(text):
        found.append("repeat")
    ratio = len(text) / max(1, len(ja))
    if len(ja) >= 4 and ratio > 6:
        found.append("too-long")
    if len(ja) >= 20 and ratio < 0.7:
        found.append("too-short")
    if text.strip() == ja.strip() and _CJK.search(ja):
        found.append("untranslated")
    if len(_QUOTED.findall(text)) >= 2 and ja.count("「") <= 1:
        found.append("merged")
    if glossary:
        for name_ja, romaji in glossary.names.items():
            if name_ja in ja and romaji.lower() not in text.lower():
                found.append(f"name:{romaji}")
    return found


def _echo_ok(ja: str, src: str) -> bool:
    """The model's copy of the line start must match the line it was given."""
    src = (src or "").strip()
    if not src:
        return False
    n = min(3, len(src), len(ja.strip()))
    return ja.strip()[:n] == src[:n]


def _schema(n: int) -> dict:
    item = {
        "type": "object",
        "properties": {
            "src": {"type": "string"},
            "en": {"type": "string"},
            "es": {"type": "string"},
        },
        "required": ["src", "en", "es"],
        "additionalProperties": False,
    }
    keys = [str(i) for i in range(n)]
    return {
        "name": "book_lines",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": dict.fromkeys(keys, item),
            "required": keys,
            "additionalProperties": False,
        },
    }


class BookTranslator:
    """Translates one book's lines to EN and ES; ``translate_book`` is the entry point."""

    def __init__(
        self,
        client,
        model: str = DEFAULT_MODEL,
        glossary: Glossary | None = None,
        title: str = "",
        workers: int = 5,
    ):
        self._client = client
        self._model = model
        self.glossary = glossary or Glossary()
        self._title = title
        self._workers = max(1, workers)
        self.usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
        # Lines still flagged after repair: `flagged` for the last call, `review` for the whole run.
        self.flagged: list[dict] = []
        self.review: list[dict] = []

    def _system(self) -> str:
        book = f" ({self._title})" if self._title else ""
        return (
            f"You translate a Japanese novel{book} line by line for a sentence-mining site. You "
            "receive a JSON object mapping an index to one line; neighbouring lines are context "
            "only. For EVERY index return an object with: 'src' = the first 5 characters of THAT "
            "line copied exactly, 'en' = natural English translation of the WHOLE line and nothing "
            "else, 'es' = natural neutral Latin American Spanish translation of the WHOLE line and "
            "nothing else. Never merge, split, move, shorten or skip lines; an unfinished line "
            "stays unfinished. Write names exactly as the line writes them: a family name stays a "
            "family name, a given name only where the Japanese uses it. Keep honorifics (-kun, "
            "-san). No notes, no romanization.\n\n" + self.glossary.prompt()
        ).strip()

    def _request(self, lines: list[str]) -> dict:
        payload = json.dumps({str(i): t for i, t in enumerate(lines)}, ensure_ascii=False)
        for attempt in range(1, 4):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": self._system()},
                        {"role": "user", "content": payload},
                    ],
                    response_format={"type": "json_schema", "json_schema": _schema(len(lines))},
                    temperature=0.2,
                    **_model_kwargs(self._model),
                )
                usage = getattr(resp, "usage", None)
                self.usage["requests"] += 1
                self.usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
                self.usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
                out = json.loads(resp.choices[0].message.content or "{}")
                return out if isinstance(out, dict) else {}
            except Exception as e:  # noqa: BLE001 - a failed batch is retried, then repaired
                logger.warning(f"Translation request failed ({type(e).__name__}), try {attempt}/3")
                time.sleep(2 * attempt)
        return {}

    def _translate_window(self, lines: list[str]) -> list[tuple[str, str] | None]:
        """EN/ES for each line, or None where the answer is missing or belongs to another line."""
        out = self._request(lines)
        result: list[tuple[str, str] | None] = []
        for i, ja in enumerate(lines):
            item = out.get(str(i))
            if not isinstance(item, dict) or not _echo_ok(ja, item.get("src", "")):
                result.append(None)
                continue
            result.append(((item.get("en") or "").strip(), (item.get("es") or "").strip()))
        return result

    def _score(self, ja: str, pair: tuple[str, str] | None) -> int:
        if pair is None:
            return 99
        return len(problems(ja, pair[0], self.glossary)) + len(problems(ja, pair[1], self.glossary))

    def _flag_indices(self, lines: list[str], pairs: list[tuple[str, str] | None]) -> set[int]:
        bad = {i for i, (ja, p) in enumerate(zip(lines, pairs, strict=True)) if self._score(ja, p)}
        # A shifted batch shows up as two different lines with the same translation.
        for i in range(len(lines) - 1):
            a, b = pairs[i], pairs[i + 1]
            if a and b and lines[i] != lines[i + 1] and (a[0] == b[0] or a[1] == b[1]):
                bad.update((i, i + 1))
        return bad

    def repair(
        self, lines: list[str], pairs: list[tuple[str, str] | None]
    ) -> list[tuple[str, str] | None]:
        """Retranslate flagged lines with their neighbours as context; keep the better answer."""
        pairs = list(pairs)
        for _ in range(REPAIR_PASSES):
            bad = sorted(self._flag_indices(lines, pairs))
            if not bad:
                break

            def redo(i: int) -> tuple[int, tuple[str, str] | None]:
                lo, hi = max(0, i - CONTEXT_LINES), min(len(lines), i + CONTEXT_LINES + 1)
                return i, self._translate_window(lines[lo:hi])[i - lo]

            with ThreadPoolExecutor(max_workers=self._workers) as pool:
                for i, new in pool.map(redo, bad):
                    old_score, new_score = (
                        self._score(lines[i], pairs[i]),
                        self._score(lines[i], new),
                    )
                    # A line flagged only for sharing its neighbour's translation has no problem
                    # of its own, so a clean retranslation wins on equal score.
                    if new_score < old_score or (new is not None and new_score == old_score == 0):
                        pairs[i] = new
        self.flagged = [
            {
                "ja": lines[i],
                "en": (pairs[i] or ("", ""))[0],
                "es": (pairs[i] or ("", ""))[1],
                "problems": sorted(
                    set(problems(lines[i], (pairs[i] or ("", ""))[0], self.glossary))
                    | set(problems(lines[i], (pairs[i] or ("", ""))[1], self.glossary))
                )
                or ["shifted-neighbour"],
            }
            for i in sorted(self._flag_indices(lines, pairs))
        ]
        self.review.extend(self.flagged)
        return pairs

    def translate_book(self, lines: list[str]) -> list[tuple[str, str] | None]:
        """Translate lines (in reading order), then repair the flagged ones."""
        batches = [lines[i : i + BATCH_SIZE] for i in range(0, len(lines), BATCH_SIZE)]
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            done = list(pool.map(self._translate_window, batches))
        pairs = [p for batch in done for p in batch]
        return self.repair(lines, pairs)


def get_book_translator(book: dict) -> BookTranslator | None:
    """A BookTranslator from env (OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_TRANSLATE_MODEL)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL"))
    model = os.getenv("OPENAI_TRANSLATE_MODEL", DEFAULT_MODEL)
    try:
        workers = int(os.getenv("OPENAI_TRANSLATE_CONCURRENCY", "5"))
    except ValueError:
        workers = 5
    title = book.get("japanese_name", "")
    return BookTranslator(client, model, Glossary.from_book(book), title=title, workers=workers)
