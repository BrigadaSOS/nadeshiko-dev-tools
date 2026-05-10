"""Convert downloaded YouTube subtitles into segment data.

Each manual JA subtitle line becomes one segment (YouGlish-style: no clip
extraction, segments reference YouTube timestamps). EN/ES content is taken
from overlapping cues in their respective VTTs; gaps are filled per-segment
via DeepL when a translator is available.

Output schema mirrors the anime _data.json contract so downstream tools
(`tokenize-media`, `assets-uploader`) can consume both pipelines.
"""

import hashlib
import json
import logging
import os

from nadeshiko_dev_tools.common.file_utils import atomic_write_json
from nadeshiko_dev_tools.segment_extractor.utils.subtitle_utils import load_subtitle_file
from nadeshiko_dev_tools.segment_extractor.utils.text_utils import process_subtitle_line
from nadeshiko_dev_tools.youtube_extractor.fetch import DEEPL_LANG

logger = logging.getLogger(__name__)

_TARGET_LANGS = ("en", "es")


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
        line["text"]
        for line in lines
        if line["start_ms"] < end_ms and start_ms < line["end_ms"]
    ]
    return " ".join(matched).strip()


def _segment_hash(salt: str, video_id: str, start_ms: int, content_ja: str) -> str:
    payload = f"{salt}:{video_id}:{start_ms}:{content_ja}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


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
        lang: _load_lines(os.path.join(video_folder, f"subs.{lang}.vtt"))
        for lang in _TARGET_LANGS
    }

    if not ja_lines:
        logger.warning(f"[{video_id}] no JA lines after filtering")
        return 0

    # Pull manual translations from overlapping cues
    manual = {
        lang: [
            _join_overlapping(lang_lines[lang], ja["start_ms"], ja["end_ms"])
            for ja in ja_lines
        ]
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

    segments: list[dict] = []
    ignored: list[dict] = []

    for idx, ja in enumerate(ja_lines):
        start_ms, end_ms, ja_text = ja["start_ms"], ja["end_ms"], ja["text"]
        en_text, es_text = final["en"][idx], final["es"][idx]

        missing = [lang for lang, t in (("en", en_text), ("es", es_text)) if not t]
        if missing:
            ignored.append(
                {
                    "segment_index": idx,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": end_ms - start_ms,
                    "content_ja": ja_text,
                    "content_en": en_text or None,
                    "content_es": es_text or None,
                    "reason": f"missing translation: {','.join(missing)}",
                }
            )
            continue

        segments.append(
            {
                "segment_hash": _segment_hash(hash_salt, video_id, start_ms, ja_text),
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


def process_channel(channel_folder: str, translator=None) -> tuple[int, int, int]:
    """Process every video folder in a channel folder.

    Returns (videos_processed, total_segments, videos_failed).
    """
    info_path = os.path.join(channel_folder, "_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"No _info.json in {channel_folder}")
    with open(info_path) as f:
        info = json.load(f)
    hash_salt = info["hash_salt"]

    video_dirs = sorted(
        d
        for d in os.listdir(channel_folder)
        if os.path.isdir(os.path.join(channel_folder, d))
        and os.path.exists(os.path.join(channel_folder, d, "_meta.json"))
    )

    processed = 0
    failed = 0
    total_segments = 0
    for vid in video_dirs:
        try:
            n = process_video_folder(
                os.path.join(channel_folder, vid), hash_salt, translator
            )
            total_segments += n
            processed += 1
        except Exception:
            logger.error(f"Failed to process {vid}", exc_info=True)
            failed += 1

    return processed, total_segments, failed
