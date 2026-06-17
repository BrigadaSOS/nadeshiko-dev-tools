"""Convert downloaded YouTube subtitles into segment data.

Consecutive JA cues are grouped into sentence-level segments (YouTube cues
often split mid-sentence), then each segment becomes one entry; segments
reference YouTube timestamps. EN/ES content is taken from overlapping cues in
their respective VTTs; gaps are filled per-segment via DeepL when a translator
is available.

Output schema mirrors the anime _data.json contract so downstream tools
(`tokenize-media`, `assets-uploader`) can consume both pipelines.
"""

import glob
import hashlib
import json
import logging
import os

from nadeshiko_dev_tools.common.deepl import DEEPL_LANG
from nadeshiko_dev_tools.common.file_utils import atomic_write_json
from nadeshiko_dev_tools.segment_extractor.utils.media import generate_segment_media
from nadeshiko_dev_tools.segment_extractor.utils.subtitle_utils import load_subtitle_file
from nadeshiko_dev_tools.segment_extractor.utils.text_utils import process_subtitle_line

logger = logging.getLogger(__name__)

_TARGET_LANGS = ("en", "es")

# Sentence-final punctuation that closes a grouped segment.
_SENTENCE_ENDERS = "。．！？!?…」』）)】〕"
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
        ends_segment = text.endswith(tuple(_SENTENCE_ENDERS)) or (
            len(text) >= _SOFT_BREAK_MIN_CHARS and text.endswith(tuple(_SOFT_BREAK_AFTER))
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
    """Translate a list of JA strings to target_lang in one DeepL call."""
    if not ja_texts:
        return []
    results = translator.translate_text(
        ja_texts, source_lang="JA", target_lang=DEEPL_LANG[target_lang]
    )
    return [r.text for r in results]


def process_video_folder(
    video_folder: str,
    hash_salt: str,
    translator=None,
    group_sentences: bool = True,
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
        ja_lines = _group_sentences(ja_lines)
        logger.info(f"[{video_id}] grouped {cue_count} cues into {len(ja_lines)} segments")

    # Pull manual translations from overlapping cues
    manual = {
        lang: [_join_overlapping(lang_lines[lang], ja["start_ms"], ja["end_ms"]) for ja in ja_lines]
        for lang in _TARGET_LANGS
    }

    # Fill gaps with one DeepL batch call per language
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
        en_text, es_text = final["en"][idx], final["es"][idx]

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
