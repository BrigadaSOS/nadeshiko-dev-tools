"""Batch mora alignment processing.

Walks a media folder, processes each episode's segments by combining
_data.json (text + timing) with _pitch.json (F0 contours),
and produces _mora_pitch.json files.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from rich.console import Console

from nadeshiko_dev_tools.common.archive import discover_episodes
from nadeshiko_dev_tools.common.progress import create_progress

console = Console()

MORA_PITCH_FILE = "_mora_pitch.json"
DATA_FILE = "_data.json"
PITCH_FILE = "_pitch.json"

# Quick check: contains at least one CJK/hiragana/katakana character
_JA_RE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")


def _is_japanese(text: str) -> bool:
    return bool(_JA_RE.search(text))


def process_episode(
    episode_dir: Path,
    progress,
    task_id,
) -> dict | None:
    """Process all segments in an episode, producing mora alignment data.

    Returns the mora pitch data dict, or None if no data produced.
    """
    from .analyzer import analyze_text
    from .timing import assign_mora_timing, slice_f0

    data_file = episode_dir / DATA_FILE
    pitch_file = episode_dir / PITCH_FILE

    if not data_file.exists() or not pitch_file.exists():
        return None

    with open(data_file) as f:
        data = json.load(f)
    with open(pitch_file) as f:
        pitch_data = json.load(f)

    segments_list = data.get("segments", [])
    pitch_segments = pitch_data.get("segments", {})
    pitch_meta = pitch_data.get("metadata", {})
    sample_ms = pitch_meta.get("sample_ms", 10)

    result_segments: dict = {}

    for seg in segments_list:
        seg_hash = seg.get("segment_hash", "")
        if not seg_hash:
            progress.advance(task_id)
            continue

        # Check pitch data exists for this segment
        pitch_entry = pitch_segments.get(seg_hash)
        if not pitch_entry:
            progress.advance(task_id)
            continue

        f0 = pitch_entry.get("f0", [])
        seg_sample_ms = pitch_entry.get("sample_ms", sample_ms)

        # Process each Japanese subtitle line independently
        ja_subs = seg.get("subtitles", {}).get("ja", [])
        if not ja_subs:
            progress.advance(task_id)
            continue

        seg_start_ms = seg.get("start_ms", 0)
        all_words: list[dict] = []

        for sub in ja_subs:
            text = sub.get("text", "").strip()
            if not text or not _is_japanese(text):
                continue

            sub_start = sub.get("start_ms", seg_start_ms)
            sub_end = sub.get("end_ms", seg.get("end_ms", seg_start_ms))

            try:
                words = analyze_text(text)
            except Exception:
                continue

            if not words:
                continue

            # Slice F0 for this subtitle's time range (relative to segment)
            rel_start = sub_start - seg_start_ms
            rel_end = sub_end - seg_start_ms
            frame_start = max(0, rel_start // seg_sample_ms)
            frame_end = min(len(f0), rel_end // seg_sample_ms + 1)
            sub_f0 = f0[frame_start:frame_end] if frame_start < frame_end else []

            # Assign timing using voiced regions from F0
            assign_mora_timing(
                words, rel_start, rel_end,
                f0=sub_f0, sample_ms=seg_sample_ms,
            )

            # Serialize words with F0 slicing
            # mora _start_ms/_end_ms are already relative to segment (0-based)
            for word in words:
                mora_out = []
                for m in word.mora:
                    m_start = getattr(m, "_start_ms", 0)
                    m_end = getattr(m, "_end_ms", 0)

                    f0_slice, f0_mean = slice_f0(
                        m_start, m_end, f0, seg_sample_ms,
                        segment_start_ms=0,
                    )

                    mora_out.append({
                        "kana": m.kana,
                        "accent": m.accent,
                        "start_ms": m_start,
                        "end_ms": m_end,
                        "f0": f0_slice,
                        "f0_mean": f0_mean,
                    })

                if mora_out:
                    all_words.append({
                        "surface": word.surface,
                        "reading": word.reading,
                        "pos": word.pos,
                        "mora": mora_out,
                    })

        if all_words:
            result_segments[seg_hash] = {"words": all_words}

        progress.advance(task_id)

    return result_segments if result_segments else None


def run_batch(
    media_folder: Path,
    resume: bool = True,
) -> None:
    """Run batch mora alignment on a single media folder."""
    console.print("[bold]Mora Pitch Aligner[/bold]")
    console.print(f"Media: {media_folder}")
    console.print()

    all_episodes = discover_episodes(media_folder)
    if not all_episodes:
        console.print("[yellow]No episode folders found.[/yellow]")
        return

    # Filter episodes: need both _data.json and _pitch.json
    pending: list[Path] = []
    skipped = 0
    no_data = 0

    for ep_dir in all_episodes:
        if resume and (ep_dir / MORA_PITCH_FILE).exists():
            skipped += 1
            continue
        if (ep_dir / DATA_FILE).exists() and (ep_dir / PITCH_FILE).exists():
            pending.append(ep_dir)
        else:
            no_data += 1

    console.print(
        f"Episodes: {len(all_episodes)} total,"
        f" {len(pending)} to process"
        + (f", {skipped} done" if skipped else "")
        + (f", {no_data} missing data/pitch" if no_data else "")
    )

    if not pending:
        console.print("[green]All episodes already processed![/green]")
        return

    # Count total segments across pending episodes
    total_segments = 0
    for ep_dir in pending:
        with open(ep_dir / DATA_FILE) as f:
            d = json.load(f)
        total_segments += len(d.get("segments", []))

    console.print(f"Total segments: {total_segments:,}")
    console.print()

    stats = {
        "episodes_processed": 0,
        "segments_with_mora": 0,
        "elapsed_seconds": 0,
    }
    start_time = time.time()

    with create_progress(console) as progress:
        overall_task = progress.add_task("Overall", total=total_segments)

        for ep_dir in pending:
            ep_num = ep_dir.name
            progress.update(overall_task, description=f"ep {ep_num}")

            ep_data = process_episode(ep_dir, progress, overall_task)

            if ep_data:
                mora_result = {
                    "metadata": {"version": "1"},
                    "segments": ep_data,
                }

                output_file = ep_dir / MORA_PITCH_FILE
                with open(output_file, "w") as f:
                    json.dump(mora_result, f, ensure_ascii=False, separators=(",", ":"))

                seg_count = len(ep_data)
                stats["segments_with_mora"] += seg_count
                stats["episodes_processed"] += 1
                console.print(f"  ep {ep_num}: {seg_count} segments with mora data")
            else:
                console.print(
                    f"  ep {ep_num}: [yellow]no mora data produced[/yellow]"
                )

    stats["elapsed_seconds"] = round(time.time() - start_time, 1)

    console.print()
    console.print("[bold green]Mora alignment complete![/bold green]")
    console.print(f"  Episodes processed: {stats['episodes_processed']}")
    console.print(f"  Segments with mora: {stats['segments_with_mora']:,}")
    console.print(f"  Time: {stats['elapsed_seconds']:.0f}s")
