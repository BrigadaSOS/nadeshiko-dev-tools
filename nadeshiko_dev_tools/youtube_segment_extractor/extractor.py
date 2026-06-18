"""Convert downloaded YouTube subtitles into segment data.

Consecutive JA cues are grouped into sentence-level segments (YouTube cues
often split mid-sentence), then each segment becomes one entry; segments
reference YouTube timestamps. EN/ES content is taken from overlapping cues in
their respective VTTs; gaps are filled per-segment via the LLM translator when
one is available.

Output schema mirrors the anime _data.json contract so downstream tools
(`tokenize-media`, `assets-uploader`) can consume both pipelines.
"""

import glob
import hashlib
import json
import logging
import os
import re

from nadeshiko_dev_tools.common.file_utils import atomic_write_json
from nadeshiko_dev_tools.common.translator import LANG_CODES
from nadeshiko_dev_tools.segment_extractor.utils.media import generate_segment_media
from nadeshiko_dev_tools.segment_extractor.utils.subtitle_utils import load_subtitle_file
from nadeshiko_dev_tools.segment_extractor.utils.text_utils import process_subtitle_line

logger = logging.getLogger(__name__)

_TARGET_LANGS = ("en", "es")

# Sentence-final punctuation that closes a grouped segment.
_SENTENCE_ENDERS = "。．！？!?…」』）)】〕"
# These vlog subtitles are largely unpunctuated, so we also close a group when the
# Japanese itself ends a sentence: a polite/plain predicate or a sentence-final
# particle. This is a deterministic, Japanese-only signal (no reliance on the English
# translation). Continuative tails (〜て/〜けど/〜が/〜ので/dangling particles) deliberately
# don't match, so clauses that run on are kept together.
# Only high-confidence enders: plain past forms (〜た/〜った/〜だった) are deliberately
# excluded because they're also attributive (modifying a noun in the next cue), so
# cutting there could split a relative clause from its noun. The LLM handles those.
_JA_SENTENCE_END_RE = re.compile(
    r"(?:"
    r"です|ます|ました|でした|ません(?:でした)?|でしょう?|ましょう?"  # polite predicates
    r"|もんね?|かな|っけ|じゃん"  # casual sentence enders
    r"|[ねよわぞか]"  # sentence-final particles
    r")$"
)
# Soft break: split at a comma once the group is already substantial, so long
# run-on speech that never reaches sentence-final punctuation still gets cut at a
# natural clause boundary.
_SOFT_BREAK_AFTER = "、，"
_SOFT_BREAK_MIN_CHARS = 40
# Hard caps (checked before adding a cue, so a group never exceeds them) bound
# runaway groups when a creator rarely punctuates.
_GROUP_MAX_CHARS = 80
_GROUP_MAX_DURATION_MS = 12_000
_GROUP_MAX_CUES = 4
# Per-segment clip cap. Grouping can't split a single cue, so a lone cue left on
# screen far longer than its speech (e.g. an outtake caption over a visual gag) would
# otherwise yield a very long clip. The segment's end is trimmed to this bound.
_SEGMENT_MAX_DURATION_MS = 12_000


def _group_sentences(ja_lines: list[dict]) -> list[dict]:
    """Merge consecutive JA cues into sentence-level segments.

    YouTube cues frequently split a sentence across several cues. Accumulate cues
    into a group, closing it when the joined text ends on sentence-final
    punctuation (or a comma once it's already long). Hard caps (chars/duration/cue
    count) are checked before adding the next cue, so a group never overflows them.
    Each returned group has start_ms/end_ms/text like a single cue, so the rest of
    the pipeline is unchanged.
    """
    groups: list[dict] = []
    cur: dict | None = None

    def close() -> None:
        nonlocal cur
        if cur is not None:
            del cur["_cues"]
            groups.append(cur)
            cur = None

    for line in ja_lines:
        # Close the current group before it would overflow a hard cap.
        if cur is not None and (
            len(cur["text"]) + len(line["text"]) > _GROUP_MAX_CHARS
            or cur["_cues"] + 1 > _GROUP_MAX_CUES
            or line["end_ms"] - cur["start_ms"] > _GROUP_MAX_DURATION_MS
        ):
            close()

        if cur is None:
            cur = {**line, "_cues": 1}
        else:
            cur["text"] += line["text"]
            cur["end_ms"] = line["end_ms"]
            cur["_cues"] += 1

        text = cur["text"].rstrip()
        ends_segment = (
            text.endswith(tuple(_SENTENCE_ENDERS))
            or _JA_SENTENCE_END_RE.search(text) is not None
            or (len(text) >= _SOFT_BREAK_MIN_CHARS and text.endswith(tuple(_SOFT_BREAK_AFTER)))
        )
        if ends_segment:
            close()

    close()
    return groups


def _load_lines(vtt_path: str) -> list[dict]:
    if not os.path.exists(vtt_path):
        return []
    subs = load_subtitle_file(vtt_path)
    lines = []
    for ev in subs:
        text = process_subtitle_line(ev)
        if not text:
            continue
        lines.append({"start_ms": int(ev.start), "end_ms": int(ev.end), "text": text})
    lines.sort(key=lambda x: x["start_ms"])
    return lines


def _join_overlapping(lines: list[dict], start_ms: int, end_ms: int) -> str:
    matched = [
        line["text"] for line in lines if line["start_ms"] < end_ms and start_ms < line["end_ms"]
    ]
    return " ".join(matched).strip()


def _strip_telop(ja: str, en: str, es: str) -> tuple[str, str, str] | None:
    """Clean on-screen-caption (telop) markers; return cleaned text or None to drop.

    Many vloggers wrap text shown on screen but NOT spoken aloud in annotation marks:
    ※ for asides and ＊ (NFKC-normalised to *) for 「おまけ」 blooper/outtake captions, both
    rendered as *...* in the EN/ES. That text has no matching speech (often just BGM), so:

    - JA beginning with ※, or containing a ＊/* outtake marker → entirely caption → dropped.
    - Otherwise a trailing ※ aside is trimmed (spoken head kept) and the matching *...* is
      removed from EN/ES.

    Markers are always stripped from kept text. Segments with no telop pass through
    unchanged (the *...* removal is a no-op when there are no markers).
    """
    ja = (ja or "").strip()
    if ja.startswith("※") or "*" in ja:
        return None

    ja = ja.split("※", 1)[0].strip()

    def _clean(text: str) -> str:
        text = re.sub(r"\*[^*]*\*", "", text or "")  # drop fully-wrapped telop spans
        text = text.replace("*", "").replace("※", "")  # strip any stray markers
        return re.sub(r"\s{2,}", " ", text).strip()

    return ja, _clean(en), _clean(es)


def _segment_hash(salt: str, video_id: str, start_ms: int, content_ja: str) -> str:
    payload = f"{salt}:{video_id}:{start_ms}:{content_ja}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def _ignored_entry(idx, start_ms, end_ms, ja_text, en_text, es_text, reason) -> dict:
    """Build one ignored_segments entry."""
    return {
        "segment_index": idx,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": end_ms - start_ms,
        "content_ja": ja_text,
        "content_en": en_text or None,
        "content_es": es_text or None,
        "reason": reason,
    }


def _batch_translate(translator, ja_texts: list[str], target_lang: str) -> list[str]:
    """Translate a list of JA strings to target_lang in one call."""
    if not ja_texts:
        return []
    results = translator.translate_text(
        ja_texts, source_lang="JA", target_lang=LANG_CODES[target_lang]
    )
    return [r.text for r in results]


def _group_sentences_llm(ja_lines: list[dict], segmenter) -> list[dict]:
    """Group cues into sentences using LLM-decided boundaries (cut positions only).

    Segmentation is driven purely by the Japanese: the model is shown the JA cues and
    returns only the indices where a new sentence starts; we slice ``ja_lines`` there
    and keep the original text/timestamps. English is never used — not as a boundary
    signal nor as context — so precision does not depend on a translation that may be
    absent or machine-generated. Because the LLM only emits cut positions, a wrong
    boundary can at most mis-place a cut, never corrupt content or timing.

    The LLM tends to under-split, so we union its boundaries with the deterministic
    Japanese sentence-end signal: cue i starts a new sentence if cue i-1's text ends
    one. Falls back to the punctuation heuristic if neither yields anything usable.
    """
    starts = set(segmenter.sentence_starts([line["text"] for line in ja_lines]))
    starts.update(
        i for i in range(1, len(ja_lines)) if _JA_SENTENCE_END_RE.search(ja_lines[i - 1]["text"])
    )
    starts = sorted(starts)
    if not starts:
        return _group_sentences(ja_lines)

    bounds = [*starts, len(ja_lines)]
    groups: list[dict] = []
    for a, b in zip(bounds, bounds[1:], strict=False):
        chunk = ja_lines[a:b]
        if not chunk:
            continue
        text = "".join(c["text"] for c in chunk)
        duration = chunk[-1]["end_ms"] - chunk[0]["start_ms"]
        # Safety net: the LLM occasionally under-splits a long run-on into one giant
        # group. Re-split any group that overflows the caps with the heuristic, so no
        # segment is pathologically long while keeping the LLM boundaries everywhere else.
        if len(text) > _GROUP_MAX_CHARS or duration > _GROUP_MAX_DURATION_MS:
            groups.extend(_group_sentences(chunk))
        else:
            groups.append(
                {"start_ms": chunk[0]["start_ms"], "end_ms": chunk[-1]["end_ms"], "text": text}
            )
    return groups


def process_video_folder(
    video_folder: str,
    hash_salt: str,
    translator=None,
    group_sentences: bool = True,
    llm_grouper=None,
) -> int:
    """Build _data.json for one video folder. Returns the segment count."""
    meta_path = os.path.join(video_folder, "_meta.json")
    data_path = os.path.join(video_folder, "_data.json")

    if not os.path.exists(meta_path):
        logger.warning(f"{video_folder}: no _meta.json, skipping")
        return 0

    if os.path.exists(data_path):
        with open(data_path) as f:
            return len(json.load(f).get("segments", []))

    with open(meta_path) as f:
        meta = json.load(f)

    video_id = meta["video_id"]
    track_mt = set(meta.get("translated_langs", []))

    ja_lines = _load_lines(os.path.join(video_folder, "subs.ja.vtt"))
    lang_lines = {
        lang: _load_lines(os.path.join(video_folder, f"subs.{lang}.vtt")) for lang in _TARGET_LANGS
    }

    if not ja_lines:
        logger.warning(f"[{video_id}] no JA lines after filtering")
        return 0

    # Group consecutive cues into sentence-level segments before aligning EN/ES.
    if group_sentences:
        cue_count = len(ja_lines)
        if llm_grouper is not None:
            try:
                ja_lines = _group_sentences_llm(ja_lines, llm_grouper)
                method = "LLM"
            except Exception:
                logger.warning(
                    f"[{video_id}] LLM grouping failed — falling back to heuristic",
                    exc_info=True,
                )
                ja_lines = _group_sentences(ja_lines)
                method = "heuristic (fallback)"
        else:
            ja_lines = _group_sentences(ja_lines)
            method = "heuristic"
        logger.info(
            f"[{video_id}] grouped {cue_count} cues into {len(ja_lines)} segments [{method}]"
        )

    # Pull manual translations from overlapping cues
    manual = {
        lang: [_join_overlapping(lang_lines[lang], ja["start_ms"], ja["end_ms"]) for ja in ja_lines]
        for lang in _TARGET_LANGS
    }

    # Fill gaps with one translation batch call per language
    final = {lang: list(manual[lang]) for lang in _TARGET_LANGS}
    if translator:
        for lang in _TARGET_LANGS:
            gap_indices = [i for i, t in enumerate(manual[lang]) if not t]
            if not gap_indices:
                continue
            translated = _batch_translate(
                translator, [ja_lines[i]["text"] for i in gap_indices], lang
            )
            for i, text in zip(gap_indices, translated, strict=True):
                final[lang][i] = text

    # fetch-youtube always saves the source video as video.<ext>; a missing one
    # means the fetch was incomplete, so skip rather than emit media-less segments.
    video_file = next(iter(sorted(glob.glob(os.path.join(video_folder, "video.*")))), None)
    if not video_file:
        logger.warning(f"[{video_id}] no video file found — fetch incomplete, skipping")
        return 0

    segments: list[dict] = []
    ignored: list[dict] = []

    for idx, ja in enumerate(ja_lines):
        start_ms, end_ms, ja_text = ja["start_ms"], ja["end_ms"], ja["text"]
        # A lone cue can linger on screen far longer than its speech; grouping can't
        # split it, so bound the clip length here.
        end_ms = min(end_ms, start_ms + _SEGMENT_MAX_DURATION_MS)
        en_text, es_text = final["en"][idx], final["es"][idx]

        # Drop on-screen-caption (telop) segments that have no spoken audio, and trim
        # telop asides off otherwise-spoken segments.
        cleaned = _strip_telop(ja_text, en_text, es_text)
        if cleaned is None:
            ignored.append(
                _ignored_entry(
                    idx, start_ms, end_ms, ja_text, en_text, es_text, "telop (non-spoken text)"
                )
            )
            continue
        ja_text, en_text, es_text = cleaned

        missing = [lang for lang, t in (("en", en_text), ("es", es_text)) if not t]
        if missing:
            ignored.append(
                _ignored_entry(
                    idx,
                    start_ms,
                    end_ms,
                    ja_text,
                    en_text,
                    es_text,
                    f"missing translation: {','.join(missing)}",
                )
            )
            continue

        segment_hash = _segment_hash(hash_salt, video_id, start_ms, ja_text)

        try:
            files = generate_segment_media(
                video_file,
                video_folder,
                segment_hash,
                start_ms / 1000,
                end_ms / 1000,
            )
        except Exception:
            logger.error(f"[{video_id}] media extraction failed for segment {idx}", exc_info=True)
            ignored.append(
                _ignored_entry(
                    idx,
                    start_ms,
                    end_ms,
                    ja_text,
                    en_text,
                    es_text,
                    "media extraction failed",
                )
            )
            continue

        segments.append(
            {
                "segment_hash": segment_hash,
                "segment_index": idx,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": end_ms - start_ms,
                "content_ja": ja_text,
                "content_en": en_text,
                "content_es": es_text,
                "is_mt_en": "en" in track_mt or not manual["en"][idx],
                "is_mt_es": "es" in track_mt or not manual["es"][idx],
                "actor_ja": None,
                "actor_en": None,
                "actor_es": None,
                "files": files,
                # Filled by the NSFW tagger (tag-media) once screenshots exist.
                "content_rating": None,
                "content_analysis": None,
                "pos_analysis": None,
            }
        )

    data = {
        "metadata": {
            "version": "6",
            "video_id": video_id,
            "title": meta.get("title"),
            "duration_ms": meta.get("duration_ms"),
            "published_at": meta.get("published_at"),
            "total_segments": len(segments),
        },
        "media": {
            "channel_id": meta.get("channel_id"),
            "media_source": "youtube",
        },
        "segments": segments,
        "ignored_segments": ignored,
    }

    atomic_write_json(data_path, data)
    return len(segments)


def load_channel_info(channel_folder: str) -> dict:
    """Load a channel folder's _info.json (raises FileNotFoundError if missing)."""
    info_path = os.path.join(channel_folder, "_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"No _info.json in {channel_folder}")
    with open(info_path) as f:
        return json.load(f)


def list_video_folders(channel_folder: str) -> list[tuple[str, str]]:
    """Return (video_id, path) for each video folder produced by fetch-youtube."""
    return sorted(
        (d, os.path.join(channel_folder, d))
        for d in os.listdir(channel_folder)
        if os.path.isdir(os.path.join(channel_folder, d))
        and os.path.exists(os.path.join(channel_folder, d, "_meta.json"))
    )
