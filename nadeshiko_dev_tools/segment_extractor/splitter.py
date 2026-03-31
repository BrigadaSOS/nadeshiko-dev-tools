import hashlib
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


def process_episode_segments(
    pool,
    main_mkv_filepath: str,
    anime_folder_fullpath: str,
    episode_number: int,
    episode_number_pretty: str,
    matching_subtitles: dict,
    translator,
    anime_data,
    config,
    hash_salt: str,
    sync_external_subs: bool,
    audio_index: int | None = None,
) -> int | None:
    """Process episode into segments using discovered subtitles.

    Returns:
        int: Number of segments generated, or None if processing in parallel
    """
    logger.info("Start file segmentation...")

    # Get video duration for metadata
    file_probe = ffmpeg.probe(main_mkv_filepath)
    duration_seconds = float(file_probe["format"]["duration"])
    duration_ms = int(duration_seconds * 1000)

    # Create episode folder (no season subfolder, just E01, E02, etc.)
    episode_folder_output_path = os.path.join(anime_folder_fullpath, episode_number_pretty)
    os.makedirs(episode_folder_output_path, exist_ok=True)

    if config.parallel:
        logger.info(f"[blue][E{episode_number}] Queued for parallel processing[/blue]")
        pool.apply_async(
            split_video_by_subtitles,
            (
                translator,
                main_mkv_filepath,
                matching_subtitles,
                episode_folder_output_path,
                config,
                anime_data,
                episode_number,
                duration_ms,
                hash_salt,
                sync_external_subs,
                audio_index,
            ),
            callback=lambda _: logger.info(
                f"[green][E{episode_number}] Completed processing[/green]"
            ),
            error_callback=lambda e: logger.error(f"[red][E{episode_number}] Error: {e}[/red]"),
        )
        return None  # Can't track segment count in parallel mode
    else:
        segment_count = split_video_by_subtitles(
            translator,
            main_mkv_filepath,
            matching_subtitles,
            episode_folder_output_path,
            config,
            anime_data,
            episode_number,
            duration_ms,
            hash_salt,
            sync_external_subs,
            audio_index,
        )
        logger.info(f"[green][E{episode_number}] Completed processing[/green]")
        return segment_count


def _compute_overlap_score(
    sub_a, sub_b, sample_size: int = 50
) -> tuple[float, float, float]:
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
    config,
    anime_data,
    episode_number,
    duration_ms,
    hash_salt: str = "",
    sync_external_subs: bool = True,
    audio_index: int | None = None,
) -> int:
    """Split a video file into segments based on subtitles.

    Returns:
        int: Number of segments generated
    """
    logger.info(f"[cyan][E{episode_number}] Starting segmentation...[/cyan]")

    # Sync external subtitles with internal reference if requested
    if sync_external_subs:
        # Find first internal subtitle track as reference
        internal_ref = None
        for _lang, sub in subtitles.items():
            if sub.origin == "internal":
                internal_ref = sub
                break

        if internal_ref:
            for lang, sub in subtitles.items():
                if sub.origin == "external":
                    # Compute pre-sync alignment score
                    pre_overlap, pre_offset, signed_offset = _compute_overlap_score(
                        sub.data, internal_ref.data
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
                    internal_lang = None
                    for ref_lang, ref_sub in subtitles.items():
                        if ref_sub is internal_ref:
                            internal_lang = ref_lang
                            break
                    same_language = lang == internal_lang
                    if same_language and pre_overlap >= 0.5 and pre_offset < 3000:
                        logger.info(
                            f"[E{episode_number}] {lang} subs already well-aligned "
                            f"(overlap={pre_overlap:.1%}, offset={pre_offset:.0f}ms), "
                            f"skipping ffsubsync"
                        )
                        continue

                    try:
                        # Create output filepath for synced subtitle in output tmp dir
                        tmp_output_folder = os.path.join(
                            os.path.dirname(episode_folder_output_path), "tmp"
                        )
                        os.makedirs(tmp_output_folder, exist_ok=True)

                        # Pre-shift: apply bulk offset correction before ffsubsync
                        # This corrects consistent offsets (e.g., Netflix ~1s ahead)
                        # so ffsubsync only needs to handle fine-grained alignment.
                        input_for_sync = sub.filepath
                        if abs(signed_offset) >= 200:  # Only shift if offset > 200ms
                            shifted_filepath = os.path.join(
                                tmp_output_folder,
                                f"shifted_{lang}_{os.path.basename(sub.filepath)}",
                            )
                            shifted_data = load_subtitle_file(sub.filepath)
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

                        # Run ffsubsync on the (possibly pre-shifted) subs
                        subprocess.run(
                            [
                                "ffsubsync",
                                internal_ref.filepath,
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
                            synced_data, internal_ref.data
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
                            # bulk-corrected), otherwise keep the original
                            if input_for_sync != sub.filepath:
                                shifted_data = load_subtitle_file(input_for_sync)
                                subtitles[lang] = MatchingSubtitle(
                                    origin="external",
                                    filepath=input_for_sync,
                                    data=shifted_data,
                                )
                                logger.info(
                                    f"Falling back to pre-shifted {lang} subs"
                                )
                            else:
                                logger.info(
                                    f"Falling back to original {lang} subs"
                                )
                        else:
                            subtitles[lang] = MatchingSubtitle(
                                origin="external", filepath=synced_filepath, data=synced_data
                            )
                            logger.info(f"Synced {lang} subtitles against internal reference")
                    except (subprocess.CalledProcessError, Exception) as e:
                        logger.warning(f"Failed to sync {lang} subtitles: {e}")
        else:
            logger.info("No internal subtitle found for sync reference, skipping sync")

    # > From here on just assume all subtitles are perfectly synced
    synced_subtitles = subtitles

    # Extract all subtitles lines from all subtitle files passed
    sorted_lines = []
    for language, subs in synced_subtitles.items():
        for line in subs.data:
            sentence = process_subtitle_line(line)
            sorted_lines.append(
                {
                    "start": line.start,
                    "end": line.end,
                    "language": language,
                    "sentence": sentence,
                    "actor": line.name,
                }
            )

    # Sort all subtitle lines by start timestamp
    sorted_lines.sort(key=lambda x: x["start"])

    # Give an id to each line
    for i, line in enumerate(sorted_lines):
        line["sub_id"] = i
        sorted_lines[i] = line

    # Remove empty lines
    sorted_lines = list(filter(lambda x: x["sentence"], sorted_lines))

    # Remove duplicate lines (with same start, end, sentence and language)
    duplicates_set = set()
    for line in list(sorted_lines):
        # Ignore the attribute `sub_id` so we can detect duplicates
        line_hashkey = (line["start"], line["end"], line["language"], line["sentence"])

        if line_hashkey not in duplicates_set:
            duplicates_set.add(line_hashkey)
        else:
            sorted_lines.remove(line)

    # Diagnostic: per-language line counts after filtering
    lang_counts = {}
    for line in sorted_lines:
        lang_counts[line["language"]] = lang_counts.get(line["language"], 0) + 1
    lang_summary = ", ".join(f"{lang}={count}" for lang, count in sorted(lang_counts.items()))
    logger.info(f"[E{episode_number}] Lines after filtering: {lang_summary}")

    segments_data = []
    ignored_segments = []
    failed_segments = []

    segment_start = sorted_lines[0]["start"] - 1
    segment_end = sorted_lines[0]["end"] + 1
    segment_sentences = {}
    line_logs = [episode_folder_output_path, ""]
    for line in sorted_lines:
        ln = line["language"]

        # New line when:
        #   * No overlap
        #   * Overlap, but gap is smaller than 500
        if not (segment_start < line["end"] and line["start"] < segment_end) or (
            (segment_start < line["end"] and line["start"] < segment_end)
            and abs(segment_end - line["start"]) < 500
        ):
            if "ja" in segment_sentences and (
                "en" in segment_sentences or "es" in segment_sentences
            ):
                # Use first Japanese subtitle ID as segment_index
                segment_index = segment_sentences["ja"][0]["sub_id"]
                segment_logs, segment_dict, failure_reason = generate_segment(
                    segment_index,
                    episode_number,
                    segment_sentences,
                    segment_start,
                    segment_end,
                    episode_folder_output_path,
                    video_file,
                    translator,
                    config,
                    anime_data,
                    hash_salt,
                    audio_index,
                )
                if segment_logs:
                    line_logs = line_logs + segment_logs
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

            else:
                if "ja" in segment_sentences:
                    # Use first Japanese subtitle ID as segment_index
                    segment_index = segment_sentences["ja"][0]["sub_id"]
                    sentence_ja, actor_ja, subs_jp = join_sentences_to_segment(
                        segment_sentences["ja"], "ja"
                    )
                    sentence_en, actor_en, subs_en = (
                        join_sentences_to_segment(segment_sentences["en"], "en")
                        if "en" in segment_sentences
                        else (None, None, [])
                    )
                    sentence_es, actor_es, subs_es = (
                        join_sentences_to_segment(segment_sentences["es"], "es")
                        if "es" in segment_sentences
                        else (None, None, [])
                    )
                    ignored_segment = {
                        "segment_index": segment_index,
                        "start_ms": segment_start,
                        "end_ms": segment_end,
                        "duration_ms": segment_end - segment_start,
                        "content_ja": sentence_ja,
                        "content_es": sentence_es,
                        "content_en": sentence_en,
                        "actor_ja": actor_ja or None,
                        "actor_es": actor_es or None,
                        "actor_en": actor_en or None,
                        "reason": "no en/es subtitle match",
                        "files": None,
                        "subtitles": {
                            "ja": subs_jp,
                            "es": subs_es,
                            "en": subs_en,
                        },
                    }
                    ignored_segments.append(ignored_segment)
                line_logs.append("[yellow]No en/es subtitle match. Ignoring...[/yellow]")

            line_logs.append("-------------------------------------------------")
            logger.info("\n".join(line_logs))
            line_logs = [episode_folder_output_path, ""]
            line_logs.append(f"[{ln}] Line: {line}")

            segment_sentences = {ln: [line]}
            segment_start = line["start"]
            segment_end = line["end"]

        else:
            line_logs.append(f"[{ln}] Line: {line}")
            segment_sentences[ln] = segment_sentences.get(ln, [])

            # Sometimes when two characters are speaking the same line is repeated.
            # Detect that to avoid duplicating the same sentence
            eq_match = False
            for saved_line in segment_sentences[ln]:
                if (
                    saved_line["sentence"] == line["sentence"]
                    and segment_sentences[ln][-1]["end"] == line["start"]
                ):
                    eq_match = True

            if not eq_match:
                segment_sentences[ln].append(line)

            segment_start = min(segment_start, line["start"])
            segment_end = max(segment_end, line["end"])

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
    config,
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
    # Extract IDs for hashing
    subs_jp_ids = [s["id"] for s in subs_jp]
    subs_en_ids = [s["id"] for s in subs_en]
    subs_es_ids = [s["id"] for s in subs_es]

    # Generate salted hash for segment identification
    original_subtitle_id = subs_jp_ids[0]
    segment_hash = generate_segment_hash(
        anime_data.id, episode_number, original_subtitle_id, subs_jp_ids, hash_salt
    )

    sentence_spanish_is_mt = False if sentence_spanish else None
    sentence_english_is_mt = False if sentence_english else None

    if translator and not sentence_spanish:
        sentence_spanish = translator.translate_text(
            sentence_japanese, source_lang="JA", target_lang="ES"
        ).text
        sentence_spanish_is_mt = True
        logs.append(f"[DEEPL - SPANISH]: {sentence_spanish}")

    if translator and not sentence_english:
        sentence_english = translator.translate_text(
            sentence_japanese, source_lang="JA", target_lang="EN-US"
        ).text
        sentence_english_is_mt = True
        logs.append(f"[DEEPL - ENGLISH]: {sentence_english}")

    start_ms = segment_start
    end_ms = segment_end
    duration_ms = segment_end - segment_start
    start_time_delta = timedelta(milliseconds=segment_start)
    start_time_seconds = start_time_delta.total_seconds()
    end_time_delta = timedelta(milliseconds=segment_end)
    end_time_seconds = end_time_delta.total_seconds()

    def build_ignored_segment(reason: str) -> dict:
        return {
            "segment_index": segment_index,
            "start_ms": start_ms,
            "end_ms": end_ms,
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

    subs_jp_ids_str = ",".join(list(map(str, subs_jp_ids)))
    subs_es_ids_str = ",".join(list(map(str, subs_es_ids)))
    subs_en_ids_str = ",".join(list(map(str, subs_en_ids)))
    logs.append(f"({segment_hash}) {start_time_delta} - {end_time_delta}")
    logs.append(f"[JA] ({subs_jp_ids_str}) {sentence_japanese}")
    logs.append(f"[ES] ({subs_es_ids_str}) {sentence_spanish}")
    logs.append(f"[EN] ({subs_en_ids_str}) {sentence_english}")

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

    if video_file and not config.dryrun:
        try:
            audio_path = os.path.join(output_path, audio_filename)

            # Build ffmpeg command for audio extraction
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_time_seconds),
                "-i",
                video_file,
                "-t",
                str(end_time_seconds - start_time_seconds),
            ]

            # Add -map option if specific audio track is selected
            if audio_index is not None:
                ffmpeg_cmd.extend(["-map", f"0:{audio_index}"])

            ffmpeg_cmd.extend(
                [
                    "-vn",
                    "-af",
                    "loudnorm=I=-16:LRA=11:TP=-2",
                    "-c:a",
                    "libmp3lame",
                    "-q:a",
                    "5",
                    audio_path,
                ]
            )

            subprocess.call(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            logs.append(f"> Saved audio in {audio_path}")

        except Exception as err:
            logger.error(f"[red]Error creating audio '{audio_filename}': {err}[/red]")
            return logs, None, "audio"

        try:
            screenshot_path = os.path.join(output_path, screenshot_filename)
            screenshot_time = (start_time_seconds + end_time_seconds) / 2

            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(screenshot_time),
                    "-i",
                    video_file,
                    "-vframes",
                    "1",
                    "-vf",
                    "scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:black",
                    "-c:v",
                    "libwebp",
                    "-quality",
                    "85",
                    "-method",
                    "6",
                    screenshot_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if result.returncode != 0:
                logger.error(f"[red]ffmpeg screenshot failed: {result.stdout}[/red]")
                raise RuntimeError(f"ffmpeg screenshot failed with code {result.returncode}")

            logs.append(f"> Saved screenshot in {screenshot_path}")

            # Content rating is handled by batch_tagger() in pipeline.py after extraction
            content_rating = None
            content_analysis = None

        except Exception as err:
            logger.error(f"[red]Error creating screenshot '{screenshot_filename}': {err}[/red]")
            return logs, None, "screenshot"

        video_path = os.path.join(output_path, video_filename)

        try:
            # Web-optimized: baseline profile, level 3.0, fastdecode tune for browser compatibility
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loop",
                    "1",
                    "-framerate",
                    "24",
                    "-i",
                    screenshot_path,
                    "-i",
                    audio_path,
                    "-vf",
                    "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,setsar=1",
                    "-c:v",
                    "libx264",
                    "-profile:v",
                    "baseline",
                    "-level",
                    "3.0",
                    "-preset",
                    "faster",
                    "-tune",
                    "fastdecode",
                    "-crf",
                    "35",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-aac_coder",
                    "twoloop",
                    "-b:a",
                    "96k",
                    "-movflags",
                    "+faststart",
                    "-shortest",
                    video_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if result.returncode != 0:
                logger.error(f"[red]ffmpeg video failed: {result.stdout}[/red]")
                raise RuntimeError(f"ffmpeg video failed with code {result.returncode}")
            logs.append(f"> Saved video in {video_path}")

        except Exception as err:
            logger.error(f"[red]Error creating video '{video_filename}': {err}[/red]")
            return logs, None, "video"

    # Tokenization is handled by batch_tokenizer() in pipeline.py after extraction
    pos_analysis = None

    segment_dict = {
        "segment_hash": segment_hash,
        "segment_index": segment_index,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": duration_ms,
        "content_ja": sentence_japanese,
        "content_es": sentence_spanish,
        "content_en": sentence_english,
        "is_mt_es": sentence_spanish_is_mt or False,
        "is_mt_en": sentence_english_is_mt or False,
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
        "pos_analysis": pos_analysis,
    }

    logs.append("[green]Segment saved![/green]")
    return logs, segment_dict, None


