import hashlib
import json
import logging
import os
import subprocess
import warnings
from collections import namedtuple
from datetime import timedelta

import ffmpeg
from rich.console import Console

from nadeshiko_dev_tools.common.file_utils import (
    write_data_json,
)
from nadeshiko_dev_tools.segment_extractor.utils.media import (
    build_clip,
    extract_audio,
    extract_screenshot,
)
from nadeshiko_dev_tools.segment_extractor.utils.subtitle_utils import (
    load_subtitle_file,
)
from nadeshiko_dev_tools.segment_extractor.utils.text_utils import (
    join_sentences_to_segment,
    process_subtitle_line,
)

warnings.filterwarnings("ignore", message="Subtitle stream parsing is not supported")

console = Console()
logger = logging.getLogger(__name__)
logger.propagate = 0
if not logger.handlers:
    from rich.logging import RichHandler

    handler = RichHandler(console=console, show_time=True, show_path=False, markup=True)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

MatchingSubtitle = namedtuple("MatchingSubtitle", ["origin", "data", "filepath"])
MAX_SEGMENT_CONTENT_LENGTH = 500
MAX_SEGMENT_JP_LINES = 4
# Subtitle lines from JA/EN/ES are sorted by time and grouped into segments
# when they overlap. This threshold (ms) controls how much a new line must
# overlap the current segment to be considered part of the same dialogue moment.
# A JA line and its EN/ES translations typically overlap by 1000ms+, while
# consecutive dialogue turns overlap by < 300ms. 500ms cleanly separates them.
MIN_OVERLAP_TO_MERGE_MS = 500
# When matching EN/ES lines to JA groups, allow this much gap (ms) between the
# line and the group's time range. Different subtitle sources (BD vs fansub vs
# streaming) often have timing offsets of 100-500ms. Without tolerance, lines
# that start slightly after the group ends are lost.
# Increased from 500ms to 1000ms to handle multi-line subtitle grouping mismatches
# where EN combines lines that are split in JA with small gaps.
MATCH_TOLERANCE_MS = 1000

_OP_ED_KEYWORDS = {"opening", "ending", "op", "ed"}


def _probe_chapters(filepath: str) -> list[tuple[int, int]]:
    """Extract OP/ED time ranges (ms) from a single file's chapter metadata."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_chapters", "-print_format", "json", filepath],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return []
        chapters = json.loads(result.stdout).get("chapters", [])
    except Exception:
        return []

    ranges = []
    for ch in chapters:
        title = ch.get("tags", {}).get("title", "").strip().lower()
        if title and any(kw in title for kw in _OP_ED_KEYWORDS):
            start_ms = int(float(ch.get("start_time", 0)) * 1000)
            end_ms = int(float(ch.get("end_time", 0)) * 1000)
            ranges.append((start_ms, end_ms))
    return ranges


def _get_op_ed_ranges(
    video_file: str, episode_number: int, input_folder: str | None = None
) -> list[tuple[int, int]]:
    """Get OP/ED time ranges from chapter metadata.

    Tries the video file first. If it has no named chapters and input_folder
    is provided, checks other MKVs in the same folder for matching episodes
    (e.g., a different release that has named chapters).
    """
    if not video_file:
        return []

    # Try the primary video file
    ranges = _probe_chapters(video_file)
    if ranges:
        labels = ", ".join(f"{s // 1000}-{e // 1000}s" for s, e in ranges)
        logger.info(f"[E{episode_number}] OP/ED from chapters: {labels}")
        return ranges

    # Try other MKVs in the input folder (different release may have named chapters)
    if input_folder:
        from guessit import guessit

        video_basename = os.path.basename(video_file)
        for f in os.listdir(input_folder):
            if not f.endswith(".mkv") or f == video_basename:
                continue
            info = guessit(f)
            ep = info.get("episode")
            if isinstance(ep, list):
                ep = ep[0]
            if ep == episode_number:
                alt_path = os.path.join(input_folder, f)
                ranges = _probe_chapters(alt_path)
                if ranges:
                    labels = ", ".join(f"{s // 1000}-{e // 1000}s" for s, e in ranges)
                    logger.info(f"[E{episode_number}] OP/ED from alternate MKV ({f}): {labels}")
                    return ranges

    return []


def _filter_dialogue_only(subs):
    """Return a copy of the subtitle file with only dialogue lines (no lyrics/signs).

    Uses process_subtitle_line to identify dialogue — same filter used later
    for segment extraction, so sync and extraction see the same lines.
    """
    import copy

    filtered = copy.deepcopy(subs)
    filtered.events = [line for line in filtered.events if process_subtitle_line(line)]
    return filtered


def _in_op_ed(start_ms: int, end_ms: int, op_ed_ranges: list[tuple[int, int]]) -> bool:
    """Check if a subtitle line's midpoint falls within an OP/ED range."""
    midpoint = (start_ms + end_ms) // 2
    return any(r_start <= midpoint <= r_end for r_start, r_end in op_ed_ranges)


def process_episode_segments(
    main_mkv_filepath: str,
    anime_folder_fullpath: str,
    episode_number: int,
    matching_subtitles: dict,
    translator,
    anime_data,
    hash_salt: str,
    sync_external_subs: bool,
    dryrun: bool = False,
    audio_index: int | None = None,
    input_folder: str | None = None,
) -> int:
    """Process episode into segments using discovered subtitles.

    Returns:
        int: Number of segments generated
    """
    logger.info("Start file segmentation...")

    # Get video duration for metadata
    file_probe = ffmpeg.probe(main_mkv_filepath)
    duration_seconds = float(file_probe["format"]["duration"])
    duration_ms = int(duration_seconds * 1000)

    episode_folder_output_path = os.path.join(anime_folder_fullpath, str(episode_number))
    os.makedirs(episode_folder_output_path, exist_ok=True)

    segment_count = split_video_by_subtitles(
        translator,
        main_mkv_filepath,
        matching_subtitles,
        episode_folder_output_path,
        anime_data,
        episode_number,
        duration_ms,
        hash_salt,
        sync_external_subs,
        dryrun,
        audio_index,
        input_folder,
    )
    logger.info(f"[green][E{episode_number}] Completed processing[/green]")
    return segment_count


def _compute_overlap_score(sub_a, sub_b, sample_size: int = 50) -> tuple[float, float, float]:
    """Measure alignment between two subtitle tracks.

    Samples lines from sub_a, finds closest-in-time match in sub_b, and computes:
      - overlap_ratio: fraction of sampled lines that overlap with a sub_b line
      - mean_offset_ms: average absolute start-time offset for overlapping pairs
      - signed_offset_ms: median signed offset (a_start - b_start), positive means
        sub_a is ahead of sub_b

    Returns:
        (overlap_ratio, mean_offset_ms, signed_offset_ms)
    """
    lines_a = [e for e in sub_a if e.type == "Dialogue"]
    lines_b = [e for e in sub_b if e.type == "Dialogue"]

    if not lines_a or not lines_b:
        return 0.0, 0.0, 0.0

    # Sample evenly across the track
    step = max(1, len(lines_a) // sample_size)
    sampled = lines_a[::step][:sample_size]

    overlaps = 0
    offsets = []
    signed_offsets = []

    for a_line in sampled:
        a_start, a_end = a_line.start, a_line.end
        best_offset = None
        best_signed = None

        for b_line in lines_b:
            b_start, b_end = b_line.start, b_line.end
            # Check temporal overlap
            if a_start < b_end and b_start < a_end:
                offset = abs(a_start - b_start)
                if best_offset is None or offset < best_offset:
                    best_offset = offset
                    best_signed = a_start - b_start

        if best_offset is not None:
            overlaps += 1
            offsets.append(best_offset)
            signed_offsets.append(best_signed)

    overlap_ratio = overlaps / len(sampled) if sampled else 0.0
    mean_offset = sum(offsets) / len(offsets) if offsets else 0.0
    # Use median for signed offset — more robust to outliers
    signed_offset = sorted(signed_offsets)[len(signed_offsets) // 2] if signed_offsets else 0.0
    return overlap_ratio, mean_offset, signed_offset


def split_video_by_subtitles(
    translator,
    video_file,
    subtitles,
    episode_folder_output_path,
    anime_data,
    episode_number,
    duration_ms,
    hash_salt: str = "",
    sync_external_subs: bool = True,
    dryrun: bool = False,
    audio_index: int | None = None,
    input_folder: str | None = None,
) -> int:
    """Split a video file into segments based on subtitles.

    Returns:
        int: Number of segments generated
    """
    logger.info(f"[cyan][E{episode_number}] Starting segmentation...[/cyan]")

    # Sync external subtitles with internal reference if requested
    if sync_external_subs:
        # Find first internal subtitle track as sync reference
        internal_ref = None
        internal_ref_lang = None
        for lang, sub in subtitles.items():
            if sub.origin == "internal":
                internal_ref = sub
                internal_ref_lang = lang
                break

        if internal_ref:
            # Create a dialogue-only version of the reference for alignment scoring.
            # Lyrics/signs have different timing and would skew overlap calculations.
            ref_dialogue = _filter_dialogue_only(internal_ref.data)

            for lang, sub in subtitles.items():
                if sub.origin == "external":
                    sub_dialogue = _filter_dialogue_only(sub.data)

                    # Compute pre-sync alignment score (dialogue-only)
                    pre_overlap, pre_offset, signed_offset = _compute_overlap_score(
                        sub_dialogue, ref_dialogue
                    )
                    logger.info(
                        f"[E{episode_number}] Pre-sync {lang}: "
                        f"overlap={pre_overlap:.1%}, mean_offset={pre_offset:.0f}ms, "
                        f"signed={signed_offset:+.0f}ms"
                    )

                    # Only skip ffsubsync for same-language subs that are
                    # already well-aligned. Cross-language overlap is unreliable
                    # because different languages have different line splitting,
                    # so nearly everything "overlaps" in a 24-min episode.
                    same_language = lang == internal_ref_lang
                    if same_language and pre_overlap >= 0.5 and pre_offset < 3000:
                        logger.info(
                            f"[E{episode_number}] {lang} subs already well-aligned "
                            f"(overlap={pre_overlap:.1%}, offset={pre_offset:.0f}ms), "
                            f"skipping ffsubsync"
                        )
                        continue

                    try:
                        tmp_output_folder = os.path.join(
                            os.path.dirname(episode_folder_output_path), "tmp"
                        )
                        os.makedirs(tmp_output_folder, exist_ok=True)

                        # Save dialogue-only versions for sync (lyrics/signs would
                        # confuse ffsubsync, especially when one track has OP lyrics
                        # and the other doesn't)
                        ref_clean_path = os.path.join(
                            tmp_output_folder,
                            f"clean_ref_{os.path.basename(internal_ref.filepath)}",
                        )
                        ref_dialogue.save(ref_clean_path)

                        sub_clean_path = os.path.join(
                            tmp_output_folder,
                            f"clean_{lang}_{os.path.basename(sub.filepath)}",
                        )
                        sub_dialogue.save(sub_clean_path)

                        # Pre-shift: apply bulk offset correction before ffsubsync
                        input_for_sync = sub_clean_path
                        if abs(signed_offset) >= 200:  # Only shift if offset > 200ms
                            shifted_filepath = os.path.join(
                                tmp_output_folder,
                                f"shifted_{lang}_{os.path.basename(sub.filepath)}",
                            )
                            shifted_data = load_subtitle_file(sub_clean_path)
                            shift_ms = -int(signed_offset)
                            for line in shifted_data:
                                line.start = max(0, line.start + shift_ms)
                                line.end = max(0, line.end + shift_ms)
                            shifted_data.save(shifted_filepath)
                            input_for_sync = shifted_filepath
                            logger.info(
                                f"[E{episode_number}] Pre-shifted {lang} by {shift_ms:+d}ms"
                            )

                        synced_filepath = os.path.join(
                            tmp_output_folder,
                            f"synced_{lang}_{os.path.basename(sub.filepath)}",
                        )

                        # Run ffsubsync with dialogue-only files
                        subprocess.run(
                            [
                                "ffsubsync",
                                ref_clean_path,
                                "-i",
                                input_for_sync,
                                "-o",
                                synced_filepath,
                            ],
                            check=True,
                            capture_output=True,
                        )

                        # Load synced subtitle and check post-sync quality
                        synced_data = load_subtitle_file(synced_filepath)
                        post_overlap, post_offset, _ = _compute_overlap_score(
                            synced_data, ref_dialogue
                        )
                        logger.info(
                            f"[E{episode_number}] Post-sync {lang}: "
                            f"overlap={post_overlap:.1%}, mean_offset={post_offset:.0f}ms"
                        )

                        # Fall back if sync degraded alignment
                        overlap_dropped = pre_overlap - post_overlap > 0.10
                        offset_increased = post_offset - pre_offset > 2000
                        if overlap_dropped or offset_increased:
                            logger.warning(
                                f"[E{episode_number}] ffsubsync degraded {lang} alignment "
                                f"(overlap {pre_overlap:.1%}->{post_overlap:.1%}, "
                                f"offset {pre_offset:.0f}ms->{post_offset:.0f}ms). "
                            )
                            # Use the pre-shifted version if available (already
                            # bulk-corrected), otherwise keep the cleaned version
                            if input_for_sync != sub_clean_path:
                                shifted_data = load_subtitle_file(input_for_sync)
                                subtitles[lang] = MatchingSubtitle(
                                    origin="external",
                                    filepath=input_for_sync,
                                    data=shifted_data,
                                )
                                logger.info(f"Falling back to pre-shifted {lang} subs")
                            else:
                                subtitles[lang] = MatchingSubtitle(
                                    origin="external",
                                    filepath=sub_clean_path,
                                    data=sub_dialogue,
                                )
                                logger.info(f"Falling back to cleaned {lang} subs")
                        else:
                            subtitles[lang] = MatchingSubtitle(
                                origin="external", filepath=synced_filepath, data=synced_data
                            )
                            logger.info(f"Synced {lang} subtitles against internal reference")
                    except Exception as e:
                        logger.warning(f"Failed to sync {lang} subtitles: {e}")
        else:
            logger.info("No internal subtitle found for sync reference, skipping sync")

    # ── Detect OP/ED time ranges from MKV chapters ──
    op_ed_ranges = _get_op_ed_ranges(video_file, episode_number, input_folder)

    # ── Extract and deduplicate subtitle lines ──
    all_lines = []
    for language, subs in subtitles.items():
        for line in subs.data:
            sentence = process_subtitle_line(line)
            if sentence:
                all_lines.append(
                    {
                        "start": line.start,
                        "end": line.end,
                        "language": language,
                        "sentence": sentence,
                        "actor": line.name,
                    }
                )

    all_lines.sort(key=lambda x: x["start"])
    for i, line in enumerate(all_lines):
        line["sub_id"] = i

    seen = set()
    deduped = []
    for line in all_lines:
        key = (line["start"], line["end"], line["language"], line["sentence"])
        if key not in seen:
            seen.add(key)
            deduped.append(line)
    all_lines = deduped

    # Filter out lines that fall within OP/ED time ranges
    if op_ed_ranges:
        before_count = len(all_lines)
        all_lines = [
            line for line in all_lines if not _in_op_ed(line["start"], line["end"], op_ed_ranges)
        ]
        dropped = before_count - len(all_lines)
        if dropped:
            logger.info(f"[E{episode_number}] Dropped {dropped} OP/ED lines")

    # Separate by language
    lines_by_lang = {"ja": [], "en": [], "es": []}
    for line in all_lines:
        if line["language"] in lines_by_lang:
            lines_by_lang[line["language"]].append(line)

    lang_summary = ", ".join(f"{k}={len(v)}" for k, v in sorted(lines_by_lang.items()))
    logger.info(f"[E{episode_number}] Lines after filtering: {lang_summary}")

    ja_lines = lines_by_lang["ja"]
    if not ja_lines:
        logger.warning(f"[yellow][E{episode_number}] No JA lines after filtering[/yellow]")
        return 0

    # ── Step 1: Group JA lines into segments by JA-only overlap ──
    groups = []
    cur = {
        "ja": [ja_lines[0]],
        "en": [],
        "es": [],
        "start": ja_lines[0]["start"],
        "end": ja_lines[0]["end"],
    }

    for line in ja_lines[1:]:
        if cur["end"] - line["start"] >= MIN_OVERLAP_TO_MERGE_MS:
            cur["ja"].append(line)
            cur["end"] = max(cur["end"], line["end"])
        else:
            groups.append(cur)
            cur = {
                "ja": [line],
                "en": [],
                "es": [],
                "start": line["start"],
                "end": line["end"],
            }
    groups.append(cur)

    # ── Step 2: Match EN/ES lines to JA groups ──
    # Each EN/ES line is assigned to the JA group it overlaps most with.
    # Matching uses a tolerance window (MATCH_TOLERANCE_MS) to handle timing
    # offsets between subtitle sources. If a line genuinely spans multiple
    # groups (>40% of its duration in each), the groups are merged.
    #
    # After each language pass, group time ranges expand to include matched
    # lines, so ES (matched second) benefits from EN widening the window.
    for lang in ["en", "es"]:
        for line in lines_by_lang[lang]:
            line_dur = line["end"] - line["start"]
            matches = []
            for i, g in enumerate(groups):
                # Overlap with tolerance: negative = gap within tolerance
                ov = min(line["end"], g["end"]) - max(line["start"], g["start"])
                if ov > -MATCH_TOLERANCE_MS:
                    matches.append((i, max(ov, 0)))

            if not matches:
                continue

            if len(matches) == 1:
                groups[matches[0][0]][lang].append(line)
                continue

            # Multiple matches — assign to best, or merge if genuinely spanning
            matches.sort(key=lambda x: -x[1])
            # Lowered threshold from 0.4 to 0.15 to handle cases where EN/ES
            # combines multiple short JA lines. A 7.5s EN line spanning three
            # 2s JA lines gives ~25% overlap each - still significant for merging.
            significant = [(i, ov) for i, ov in matches if ov / line_dur > 0.15]

            if len(significant) > 1:
                # Merge into the lowest-indexed group, pop higher indices descending
                sig_indices = sorted([i for i, _ in significant])
                target = sig_indices[0]
                for i in reversed(sig_indices[1:]):
                    groups[target]["ja"].extend(groups[i]["ja"])
                    groups[target]["en"].extend(groups[i]["en"])
                    groups[target]["es"].extend(groups[i]["es"])
                    groups[target]["end"] = max(groups[target]["end"], groups[i]["end"])
                    groups[target]["start"] = min(groups[target]["start"], groups[i]["start"])
                    groups.pop(i)
                groups[target][lang].append(line)
            else:
                groups[matches[0][0]][lang].append(line)

        # Expand group time ranges to include matched lines
        for g in groups:
            for matched_line in g[lang]:
                g["start"] = min(g["start"], matched_line["start"])
                g["end"] = max(g["end"], matched_line["end"])

    logger.info(f"[E{episode_number}] JA groups: {len(groups)} (from {len(ja_lines)} JA lines)")

    # ── Step 3: Generate segments from groups ──
    segments_data = []
    ignored_segments = []
    failed_segments = []

    for group in groups:
        segment_start = group["start"]
        segment_end = group["end"]

        if not group["en"] and not group["es"]:
            # JA only — no translation matched this time range
            segment_index = group["ja"][0]["sub_id"]
            sentence_ja, actor_ja, subs_jp = join_sentences_to_segment(group["ja"], "ja")
            ignored_segments.append(
                {
                    "segment_index": segment_index,
                    "start_ms": segment_start,
                    "end_ms": segment_end,
                    "duration_ms": segment_end - segment_start,
                    "content_ja": sentence_ja,
                    "content_es": None,
                    "content_en": None,
                    "actor_ja": actor_ja or None,
                    "actor_es": None,
                    "actor_en": None,
                    "reason": "no en/es subtitle match",
                    "files": None,
                    "subtitles": {"ja": subs_jp, "es": [], "en": []},
                }
            )
            continue

        segment_sentences = {"ja": group["ja"]}
        if group["en"]:
            segment_sentences["en"] = group["en"]
        if group["es"]:
            segment_sentences["es"] = group["es"]

        segment_index = group["ja"][0]["sub_id"]
        _, segment_dict, failure_reason = generate_segment(
            segment_index,
            episode_number,
            segment_sentences,
            segment_start,
            segment_end,
            episode_folder_output_path,
            video_file,
            translator,
            dryrun,
            anime_data,
            hash_salt,
            audio_index,
        )
        if segment_dict:
            if failure_reason == "ignored":
                ignored_segments.append(segment_dict)
            else:
                segments_data.append(segment_dict)
        elif failure_reason:
            failed_segments.append(
                {
                    "segment_index": segment_index,
                    "start_ms": segment_start,
                    "reason": failure_reason,
                }
            )

    if segments_data or ignored_segments:
        write_data_json(
            episode_folder_output_path,
            segments_data,
            episode_number,
            duration_ms,
            anime_data,
            ignored_segments,
        )
        ignored_msg = f", {len(ignored_segments)} ignored" if ignored_segments else ""
        logger.info(
            f"[green][E{episode_number}] Created _data.json with "
            f"{len(segments_data)} segments{ignored_msg}[/green]"
        )
    else:
        logger.warning(f"[yellow][E{episode_number}] No segments generated[/yellow]")

    if failed_segments:
        logger.error(f"[red][E{episode_number}] {len(failed_segments)} segment(s) failed:[/red]")
        for failed in failed_segments:
            start_td = timedelta(milliseconds=failed["start_ms"])
            logger.error(
                f"[red]  - Segment #{failed['segment_index']} "
                f"at {start_td} ({failed['reason']})[/red]"
            )

    # Diagnostic: segment ratio summary
    total_segments = len(segments_data) + len(ignored_segments) + len(failed_segments)
    if total_segments > 0:
        valid_pct = len(segments_data) / total_segments * 100
        logger.info(
            f"[E{episode_number}] Segment summary: "
            f"{len(segments_data)} valid, {len(ignored_segments)} ignored, "
            f"{len(failed_segments)} failed "
            f"(total={total_segments}, valid={valid_pct:.0f}%)"
        )
        if valid_pct < 30:
            logger.warning(
                f"[yellow][E{episode_number}] Low valid segment ratio ({valid_pct:.0f}%)! "
                f"This may indicate sync or filter problems.[/yellow]"
            )

    return len(segments_data)


def generate_segment_hash(
    anilist_id: int, episode_number: int, subtitle_id: int, subs_jp_ids: list, salt: str
) -> str:
    """Generate a salted hash for a segment.

    The salt prevents reverse engineering the hash to extract Anilist IDs
    and other internal structure.
    """
    subs_str = ",".join(map(str, subs_jp_ids))
    hash_input = f"{salt}:{anilist_id}:{episode_number}:{subtitle_id}:{subs_str}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:10]


def generate_segment(
    segment_index,
    episode_number,
    segment_sentences,
    segment_start,
    segment_end,
    output_path,
    video_file,
    translator,
    dryrun,
    anime_data,
    hash_salt: str,
    audio_index: int | None = None,
):
    """Generate a single segment with audio, screenshot, and video."""
    logs = []
    sentence_japanese, actor_japanese, subs_jp = join_sentences_to_segment(
        segment_sentences["ja"], "ja"
    )
    sentence_english, actor_english, subs_en = (
        join_sentences_to_segment(segment_sentences["en"], "en")
        if "en" in segment_sentences
        else (None, None, [])
    )
    sentence_spanish, actor_spanish, subs_es = (
        join_sentences_to_segment(segment_sentences["es"], "es")
        if "es" in segment_sentences
        else (None, None, [])
    )
    # Generate salted hash for segment identification
    subs_jp_ids = [s["id"] for s in subs_jp]
    segment_hash = generate_segment_hash(
        anime_data.id, episode_number, subs_jp_ids[0], subs_jp_ids, hash_salt
    )

    sentence_spanish_is_mt = False
    sentence_english_is_mt = False

    if translator and not sentence_spanish:
        sentence_spanish = translator.translate_text(
            sentence_japanese, source_lang="JA", target_lang="ES"
        ).text
        sentence_spanish_is_mt = True
        logs.append(f"[MT - SPANISH]: {sentence_spanish}")

    if translator and not sentence_english:
        sentence_english = translator.translate_text(
            sentence_japanese, source_lang="JA", target_lang="EN-US"
        ).text
        sentence_english_is_mt = True
        logs.append(f"[MT - ENGLISH]: {sentence_english}")

    duration_ms = segment_end - segment_start
    start_time_delta = timedelta(milliseconds=segment_start)
    start_time_seconds = start_time_delta.total_seconds()
    end_time_delta = timedelta(milliseconds=segment_end)
    end_time_seconds = end_time_delta.total_seconds()

    def build_ignored_segment(reason: str) -> dict:
        return {
            "segment_index": segment_index,
            "start_ms": segment_start,
            "end_ms": segment_end,
            "duration_ms": duration_ms,
            "content_ja": sentence_japanese,
            "content_es": sentence_spanish,
            "content_en": sentence_english,
            "actor_ja": actor_japanese or None,
            "actor_es": actor_spanish or None,
            "actor_en": actor_english or None,
            "reason": reason,
            "files": None,
            "subtitles": {
                "ja": subs_jp,
                "es": subs_es,
                "en": subs_en,
            },
        }

    logs.append(f"({segment_hash}) {start_time_delta} - {end_time_delta}")
    logs.append(f"[JA] {sentence_japanese}")
    logs.append(f"[ES] {sentence_spanish}")
    logs.append(f"[EN] {sentence_english}")

    missing_languages = []
    if not sentence_japanese:
        missing_languages.append("ja")
    if not sentence_spanish:
        missing_languages.append("es")
    if not sentence_english:
        missing_languages.append("en")

    if missing_languages:
        reason = f"missing required languages: {','.join(missing_languages)}"
        logs.append(f"[yellow]Skipping segment: {reason}[/yellow]")
        return logs, build_ignored_segment(reason), "ignored"

    if len(sentence_japanese) > MAX_SEGMENT_CONTENT_LENGTH:
        reason = f"content too long ({len(sentence_japanese)} > {MAX_SEGMENT_CONTENT_LENGTH})"
        logs.append(f"[yellow]Skipping segment: {reason}[/yellow]")
        return logs, build_ignored_segment(reason), "ignored"

    if len(subs_jp) > MAX_SEGMENT_JP_LINES:
        reason = f"too many JP lines joined ({len(subs_jp)} > {MAX_SEGMENT_JP_LINES})"
        logs.append(f"[yellow]Skipping segment: {reason}[/yellow]")
        return logs, build_ignored_segment(reason), "ignored"

    audio_filename = f"{segment_hash}.mp3"
    screenshot_filename = f"{segment_hash}.webp"
    video_filename = f"{segment_hash}.mp4"
    content_rating = "SAFE"
    content_analysis = None

    if video_file and not dryrun:
        audio_path = os.path.join(output_path, audio_filename)
        screenshot_path = os.path.join(output_path, screenshot_filename)
        video_path = os.path.join(output_path, video_filename)

        try:
            extract_audio(
                video_file,
                audio_path,
                start_time_seconds,
                end_time_seconds,
                audio_index,
            )
            logs.append(f"> Saved audio in {audio_path}")
        except Exception as err:
            logger.error(f"[red]Error creating audio '{audio_filename}': {err}[/red]")
            return logs, None, "audio"

        try:
            screenshot_time = (start_time_seconds + end_time_seconds) / 2
            extract_screenshot(video_file, screenshot_path, screenshot_time)
            logs.append(f"> Saved screenshot in {screenshot_path}")

            # Content rating is handled by batch_tagger() in pipeline.py after extraction
            content_rating = None
            content_analysis = None
        except Exception as err:
            logger.error(f"[red]Error creating screenshot '{screenshot_filename}': {err}[/red]")
            return logs, None, "screenshot"

        try:
            build_clip(screenshot_path, audio_path, video_path)
            logs.append(f"> Saved video in {video_path}")
        except Exception as err:
            logger.error(f"[red]Error creating video '{video_filename}': {err}[/red]")
            return logs, None, "video"

    segment_dict = {
        "segment_hash": segment_hash,
        "segment_index": segment_index,
        "start_ms": segment_start,
        "end_ms": segment_end,
        "duration_ms": duration_ms,
        "content_ja": sentence_japanese,
        "content_es": sentence_spanish,
        "content_en": sentence_english,
        "is_mt_es": sentence_spanish_is_mt,
        "is_mt_en": sentence_english_is_mt,
        "actor_ja": actor_japanese or None,
        "actor_es": actor_spanish or None,
        "actor_en": actor_english or None,
        "files": {
            "audio": audio_filename,
            "screenshot": screenshot_filename,
            "video": video_filename,
        },
        "subtitles": {
            "ja": subs_jp,
            "es": subs_es,
            "en": subs_en,
        },
        "content_rating": content_rating,
        "content_analysis": content_analysis,
        "pos_analysis": None,
    }

    logs.append("[green]Segment saved![/green]")
    return logs, segment_dict, None
