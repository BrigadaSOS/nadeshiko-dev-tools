import hashlib
import logging
import os
import re
import subprocess
import warnings
from collections import namedtuple
from datetime import timedelta
from multiprocessing.pool import ThreadPool as Pool
from pathlib import Path

import babelfish
import deepl
import ffmpeg
from dotenv import load_dotenv
from guessit import guessit
from langdetect import detect
from rich.console import Console

from nadeshiko_dev_tools.common.anilist import CachedAnilist
from nadeshiko_dev_tools.common.config import (
    ProcessingConfig,
    load_subtitle_config,
    save_subtitle_config,
)
from nadeshiko_dev_tools.common.file_utils import (
    discover_input_folders,
    save_info_json,
    write_data_json,
)
from nadeshiko_dev_tools.media_sub_splitter.cli import command_args
from nadeshiko_dev_tools.media_sub_splitter.utils.display_utils import (
    display_file_details,
    display_folder_mappings,
)
from nadeshiko_dev_tools.media_sub_splitter.utils.ffmpeg_utils import probe_files
from nadeshiko_dev_tools.media_sub_splitter.utils.prompts import (
    confirm_processing,
    map_folder_to_anilist,
    restore_terminal,
    select_audio_tracks,
    select_subtitle_tracks,
    setup_signal_handlers,
)
from nadeshiko_dev_tools.media_sub_splitter.utils.subtitle_utils import (
    SUPPORTED_LANGUAGES,
    load_subtitle_file,
)
from nadeshiko_dev_tools.media_sub_splitter.utils.text_utils import (
    extract_anime_title_for_guessit,
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


def main():
    """Main entry point for the media-sub-splitter CLI.

    This function orchestrates the entire workflow:
    1. Discover folders with .mkv files
    2. Map folders to Anilist media
    3. Probe files for audio/subtitle streams
    4. Select audio and subtitle tracks
    5. Process episodes into segments
    """
    setup_signal_handlers()

    try:
        args = command_args()
        # Parse episodes if provided
        episodes_filter = None
        if args.episodes:
            try:
                episodes_filter = {int(e.strip()) for e in args.episodes.split(",")}
                console.print(f"[cyan]Filtering to episodes: {sorted(episodes_filter)}[/cyan]")
            except ValueError:
                console.print("[red]Invalid episodes format. Use comma-separated numbers like '1,3,5'[/red]")
                return

        config = ProcessingConfig(
            input_folder=args.input,
            output_folder=args.output,
            deepl_token=args.token,
            verbose=args.verbose,
            dryrun=args.dryrun,
            extra_punctuation=args.extra_punctuation,
            parallel=args.parallel,
            episodes=episodes_filter,
            sync_external_subs=False if args.no_sync else None,
        )

        # 1. Discover folders with .mkv files
        media_folders = discover_input_folders(config.input_folder)

        if not media_folders:
            msg = (
                f"[red]No folders with .mkv files found in {config.input_folder}! "
                "Nothing else to do.[/red]"
            )
            console.print(msg)
            return

        console.print(
            f"[green]Found {len(media_folders)} folder(s) to process in "
            f"{config.input_folder}[/green]"
        )

        console.print("\n[green bold]Discovered folders:[/green bold]")
        for folder in media_folders:
            console.print(f"  [cyan]{folder['name']}[/cyan] ({folder['file_count']} files)")

        # 2. Map each folder to Anilist media
        anilist = CachedAnilist()
        folder_mappings = {}

        console.print("\n[cyan]Mapping folders to Anilist media...[/cyan]")
        for folder in media_folders:
            anime_data = map_folder_to_anilist(folder, anilist)
            if anime_data:
                folder_mappings[folder["name"]] = {
                    "anime": anime_data,
                    "path": folder["path"],
                    "files": folder["files"],
                }

        if not folder_mappings:
            console.print(
                "[yellow]No folders were mapped to Anilist media. Nothing to process.[/yellow]"
            )
            return

        folder_mappings = filter_folder_mappings_by_episodes(folder_mappings, config.episodes)
        if not folder_mappings:
            console.print("[yellow]No files matched the selected episode filter.[/yellow]")
            return

        display_folder_mappings(folder_mappings)

        # Load existing config
        config_data = load_subtitle_config()

        # Probe files for streams
        all_file_details = probe_files(folder_mappings)
        display_file_details(all_file_details)

        # Select audio and subtitle tracks (prompts)
        audio_config = select_audio_tracks(
            folder_mappings, all_file_details, config_data.get("audio", {})
        )
        subtitle_config = select_subtitle_tracks(
            folder_mappings, all_file_details, config_data.get("subtitles", {})
        )

        # Save config if anything changed
        if audio_config != config_data.get("audio", {}) or subtitle_config != config_data.get(
            "subtitles", {}
        ):
            config_data["audio"] = audio_config
            config_data["subtitles"] = subtitle_config
            save_subtitle_config(config_data)
            console.print("\n[green]Configuration saved![/green]")

        # Confirm and process
        total_episodes = len({(d["folder_name"], d["episode"]) for d in all_file_details})
        if confirm_processing(total_episodes, len(folder_mappings)):
            episode_stats = process_episodes(config, folder_mappings, audio_config=audio_config)
            display_episode_summary_report(episode_stats)

    except KeyboardInterrupt:
        restore_terminal()
        import sys

        sys.exit(130)
    except Exception:
        restore_terminal()
        raise


# ----------------------------------------------------------------------------
# Processing Functions
# ----------------------------------------------------------------------------


def filter_folder_mappings_by_episodes(
    folder_mappings: dict,
    selected_episodes: set[int] | None,
) -> dict:
    """Filter folder mappings by selected episode numbers."""
    if not selected_episodes:
        return folder_mappings

    unmatched_episodes = set(selected_episodes)
    filtered_mappings = {}

    for folder_name, mapping in folder_mappings.items():
        matched_files = []

        for filename in mapping["files"]:
            filepath = os.path.join(mapping["path"], filename)
            guessit_query = extract_anime_title_for_guessit(filepath)

            try:
                episode_info = guessit(guessit_query)
                episode_number = episode_info.get("episode")
            except Exception as e:
                logger.warning(f"Could not determine episode number for {filepath}: {e}")
                continue

            if episode_number is None:
                logger.warning(f"Could not determine episode number for {filepath}")
                continue

            if episode_number in selected_episodes:
                matched_files.append(filename)
                unmatched_episodes.discard(episode_number)

        if matched_files:
            filtered_mappings[folder_name] = {
                **mapping,
                "files": matched_files,
            }

    if unmatched_episodes:
        console.print(
            "[yellow]No matching files found for episode(s): "
            f"{', '.join(str(ep) for ep in sorted(unmatched_episodes))}[/yellow]"
        )

    return filtered_mappings


def process_episodes(
    config: ProcessingConfig,
    folder_mappings: dict,
    subtitles_dict_remembered: dict | None = None,
    audio_config: dict | None = None,
) -> dict:
    """Process episodes in folder mappings.

    Returns:
        dict: Episode statistics with format {folder_name: {episode_number: segment_count}}
    """
    load_dotenv()
    logger.setLevel(logging.DEBUG if config.verbose else logging.INFO)

    deepl_token = os.getenv("TOKEN") or config.deepl_token
    if not deepl_token:
        logger.warning(
            " > IMPORTANT < DEEPL TOKEN has not been detected. "
            "Subtitles won't be translated to all supported languages"
        )

    translator = deepl.Translator(deepl_token) if deepl_token else None
    output_folder = config.output_folder

    if subtitles_dict_remembered is None:
        subtitles_dict_remembered = {}

    if audio_config is None:
        audio_config = {}

    # Track episode statistics
    episode_stats = {}

    # Process each folder
    pool = Pool(config.pool_size)

    for folder_name, mapping in folder_mappings.items():
        console.print(f"\n[cyan]Processing folder: {folder_name}[/cyan]")
        episode_stats[folder_name] = {}

        # Create output folder for this media
        anime_data = mapping["anime"]
        anime_folder_name = str(anime_data.id)
        anime_folder_fullpath = os.path.join(output_folder, anime_folder_name)
        os.makedirs(anime_folder_fullpath, exist_ok=True)

        # Save info.json for this media (also generates/loads the hash salt)
        info_json_fullpath = os.path.join(anime_folder_fullpath, "info.json")
        hash_salt = save_info_json(info_json_fullpath, anime_data, anime_folder_name)

        # Process each file in the folder
        for filename in mapping["files"]:
            filepath = os.path.join(mapping["path"], filename)

            # Filter by episode number if specified
            if config.episodes:
                # Extract episode number from filename using guessit
                from nadeshiko_dev_tools.media_sub_splitter.utils.text_utils import (
                    extract_anime_title_for_guessit,
                )

                guessit_query = extract_anime_title_for_guessit(filepath)
                try:
                    from guessit import guessit

                    episode_info = guessit(guessit_query)
                    episode_number = episode_info.get("episode", 1)

                    if episode_number not in config.episodes:
                        logger.info(f"Skipping episode {episode_number} (not in filter)")
                        continue
                except Exception as e:
                    logger.warning(f"Could not determine episode number for {filepath}: {e}")
                    if config.episodes:
                        # If we can't determine the episode and a filter is set, skip it
                        continue

            pool, subtitles_dict_remembered, segment_count = extract_segments_from_episode(
                pool,
                filepath,
                anime_folder_fullpath,
                translator,
                anime_data,
                subtitles_dict_remembered,
                config,
                hash_salt,
                mapping["path"],  # folder_path
                audio_config,  # audio_config
            )

            # Track segment count for this episode
            if segment_count is not None:
                # Extract episode number for stats
                from nadeshiko_dev_tools.media_sub_splitter.utils.text_utils import (
                    extract_anime_title_for_guessit,
                )
                from guessit import guessit

                guessit_query = extract_anime_title_for_guessit(filepath)
                episode_info = guessit(guessit_query)
                episode_number = episode_info.get("episode", 1)
                episode_stats[folder_name][episode_number] = segment_count

    pool.close()
    pool.join()

    console.print("[green]Processing complete![/green]")
    return episode_stats


def extract_segments_from_episode(
    pool,
    episode_filepath,
    anime_folder_fullpath,
    translator,
    anime_data,
    subtitles_dict_remembered,
    config,
    hash_salt: str,
    folder_path: str,
    audio_config: dict,
) -> tuple:
    """Extract segments from a single episode file.

    Returns:
        tuple: (pool, subtitles_dict_remembered, segment_count or None)
    """
    try:
        logger.info(f"Anime: {anime_data.title.romaji}")

        # Discover and match subtitles
        (
            episode_number,
            episode_number_pretty,
            matching_subtitles,
            subtitles_dict_remembered,
            sync_external_subs,
            main_mkv,
            audio_index,
        ) = discover_episode_subtitles(
            episode_filepath,
            anime_folder_fullpath,
            subtitles_dict_remembered,
            folder_path,
            audio_config,
            config,
        )

        # Process episode into segments
        segment_count = process_episode_segments(
            pool,
            main_mkv,
            anime_folder_fullpath,
            episode_number,
            episode_number_pretty,
            matching_subtitles,
            translator,
            anime_data,
            config,
            hash_salt,
            sync_external_subs,
            audio_index,
        )

        return pool, subtitles_dict_remembered, segment_count

    except Exception:
        logger.error("Error processing episode. Skipping...", exc_info=True)
        return pool, subtitles_dict_remembered, None


def discover_episode_subtitles(
    episode_filepath: str,
    anime_folder_fullpath: str,
    subtitles_dict_remembered: dict,
    folder_path: str,
    audio_config: dict,
    config,
) -> tuple:
    """Discover and match subtitles for an episode file."""
    from nadeshiko_dev_tools.common.file_utils import discover_matching_mkv_files
    from nadeshiko_dev_tools.media_sub_splitter.utils.prompts import (
        _get_config_key,
        select_mkv_sources_and_tracks,
        select_subtitle_streams,
    )

    logger.info(f"Filepath: {episode_filepath}\n")

    # Get episode info from guessit
    guessit_query = extract_anime_title_for_guessit(episode_filepath)
    logger.info(f"> Query for Guessit: {guessit_query}")
    episode_info = guessit(guessit_query)

    episode_number = episode_info.get("episode", 1)
    episode_number_pretty = str(episode_number)
    logger.info(f"Episode: {episode_number_pretty}")

    # Get subtitles
    logger.info("> Finding matching subtitles...")
    matching_subtitles = {}

    # Part 1: Find subtitle files on same directory as episode, with same episode number
    input_episode_parent_folder = Path(episode_filepath).parent
    subtitle_filepaths = [
        os.path.join(input_episode_parent_folder, filename)
        for filename in os.listdir(input_episode_parent_folder)
        if filename.endswith(".ass") or filename.endswith(".srt")
    ]
    logger.debug(f"Subtitle filepaths: {subtitle_filepaths}")

    for subtitle_filepath in subtitle_filepaths:
        subtitle_filename = re.sub(r"\[.*?\]|\(.*?\)", "", os.path.basename(subtitle_filepath))
        guessed_subtitle_info = guessit(subtitle_filename)
        if "episode" in guessed_subtitle_info:
            subtitle_episode = guessed_subtitle_info["episode"]
        else:
            episode_matches = re.search(r"(?!S)(\D\d\d|\D\d)\D", subtitle_filename)
            if episode_matches:
                subtitle_episode = episode_matches.group(1)
            else:
                logger.info(f"> Could not guess Episode number for subtitle: {subtitle_filepath}")
                continue

        if int(subtitle_episode) == episode_number:
            logger.info(f"> (E{subtitle_episode}) Found external subtitle: {subtitle_filepath}")

            subtitle_language = None
            if "subtitle_language" in guessed_subtitle_info:
                subtitle_language = guessed_subtitle_info["subtitle_language"].alpha2
            else:
                try:
                    subtitle_data = load_subtitle_file(subtitle_filepath)
                    subtitle_text = " ".join([event.text for event in subtitle_data])
                    subtitle_language = detect(subtitle_text)
                    logger.info(f"> External subtitle detected language: {subtitle_language}")
                except Exception as e:
                    logger.error(f"Failed to detect language for subtitle: {e}")
                    continue

            if not subtitle_language:
                logger.error("Impossible to guess the language of the subtitle. Skipping...")
                continue

            if subtitle_language not in SUPPORTED_LANGUAGES:
                logger.info(f"Language {subtitle_language} is not supported. Skipping...")
                continue

            subtitle_data = load_subtitle_file(subtitle_filepath)
            logger.info(f">Found [{subtitle_language}] subtitles: {subtitle_data}")

            if subtitle_language in matching_subtitles and len(subtitle_data) < len(
                matching_subtitles[subtitle_language]
            ):
                logger.info("Already found better matching subtitles. Skipping...")
                continue

            logger.info(f"Saving subtitles: {subtitle_data}\n")
            matching_subtitles[subtitle_language] = MatchingSubtitle(
                origin="external",
                filepath=subtitle_filepath,
                data=subtitle_data,
            )

    # Part 2: Discover and process MKV files (including the original episode file)
    folder_name = Path(episode_filepath).parent.name
    tmp_output_folder = os.path.join(anime_folder_fullpath, "tmp")
    os.makedirs(tmp_output_folder, exist_ok=True)

    # Find all matching MKV files for this episode
    matching_mkv_sources = discover_matching_mkv_files(episode_filepath, episode_number)
    logger.info(
        f"Found {len(matching_mkv_sources)} matching MKV file(s) for episode {episode_number}"
    )

    # Determine main MKV and subtitle sources
    main_mkv = episode_filepath
    audio_index = None
    subtitle_selections = {}

    if len(matching_mkv_sources) > 1:
        # Multiple MKV files - let user select main file, audio track, and subtitle sources
        logger.info("Multiple MKV files found for this episode")
        main_mkv, audio_index, subtitle_selections, subtitles_dict_remembered = (
            select_mkv_sources_and_tracks(
                matching_mkv_sources,
                folder_name,
                subtitles_dict_remembered,
                audio_config,
                folder_path,
            )
        )
    else:
        # Single MKV file - use existing behavior
        file_probe = ffmpeg.probe(episode_filepath)

        # Try to use folder audio_config if available
        config_key = _get_config_key(folder_name, folder_path)
        if audio_config and config_key in audio_config:
            audio_index = audio_config[config_key].get("index")
            # Verify the audio track exists
            if audio_index is not None:
                audio_streams = [s for s in file_probe["streams"] if s["codec_type"] == "audio"]
                audio_stream = next((s for s in audio_streams if s["index"] == audio_index), None)
                if audio_stream is None:
                    audio_index = None

        # Auto-select audio if not configured
        if audio_index is None:
            japanese_stream = None
            for stream in file_probe["streams"]:
                if stream["codec_type"] == "audio":
                    lang = stream.get("tags", {}).get("language", "").lower()
                    if lang in ("jpn", "ja", "japanese"):
                        japanese_stream = stream
                        break

            if japanese_stream:
                audio_index = japanese_stream["index"]
            else:
                audio_streams = [s for s in file_probe["streams"] if s["codec_type"] == "audio"]
                if audio_streams:
                    audio_index = audio_streams[0]["index"]

        # Generate the list of available subs from the single file
        subtitles_dict = {}
        for stream in file_probe["streams"]:
            if stream["codec_type"] == "subtitle":
                index = stream["index"]
                title = stream.get("tags", {}).get("title")
                language = stream.get("tags", {}).get("language")
                title = title if title else language
                if title and language:
                    subtitles_dict[index] = {"title": title, "language": language}

        # Get subtitle selection from prompts module
        selected_indices, subtitles_dict_remembered, sync_external_subs = select_subtitle_streams(
            subtitles_dict, folder_name, subtitles_dict_remembered
        )
        subtitle_selections = {episode_filepath: selected_indices}

    # Part 3: Extract subtitles from all selected MKV sources
    for mkv_filepath, stream_indices in subtitle_selections.items():
        logger.info(f"Processing subtitles from: {os.path.basename(mkv_filepath)}")

        # Probe the specific file
        file_probe = ffmpeg.probe(mkv_filepath)

        # Get the subtitle streams for this file
        subtitle_streams = [
            stream
            for stream in file_probe["streams"]
            if stream["codec_type"] == "subtitle" and stream["index"] in stream_indices
        ]

        for subtitle_stream in subtitle_streams:
            index = subtitle_stream["index"]
            codec = subtitle_stream["codec_name"]
            tag_language = subtitle_stream["tags"]["language"]

            # Support for non-ISO 639-3 language tags
            tag_language_normalizer = {"fre": "fra", "ger": "deu"}
            if tag_language_normalizer.get(tag_language):
                tag_language = tag_language_normalizer.get(tag_language)

            subtitle_language = babelfish.Language(tag_language).alpha2
            logger.info(
                f"Found internal subtitle stream. Index: {index}. "
                f"Codec: {codec}. Language: {subtitle_language}"
            )

            if subtitle_language not in SUPPORTED_LANGUAGES:
                logger.info(f"Language {subtitle_language} is not supported. Skipping...")
                continue

            output_sub_tmp_filepath = os.path.join(tmp_output_folder, f"tmp.{codec}")

            subprocess.call(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    mkv_filepath,
                    "-map",
                    f"0:{index}",
                    "-c",
                    "copy",
                    output_sub_tmp_filepath,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            logger.info(f"Exported subtitle to: {output_sub_tmp_filepath}")

            subtitle_data = load_subtitle_file(output_sub_tmp_filepath)
            logger.info(f">Found [{subtitle_language}] subtitles: {subtitle_data}")

            if subtitle_language in matching_subtitles:
                logger.info(f"> Already matched subtitles for {subtitle_language}!!")

                if (
                    len(subtitle_data) > len(matching_subtitles[subtitle_language])
                    and matching_subtitles[subtitle_language].origin != "external"
                ):
                    logger.info(">> Internal subtitle is longer. Overriding...")
                else:
                    continue

            logger.info(f"Saving subtitles: {subtitle_data}\n")
            anime_folder_name = os.path.basename(anime_folder_fullpath)
            output_sub_final_filepath = os.path.join(
                tmp_output_folder,
                f"{anime_folder_name} {episode_number_pretty}.{subtitle_language}.{codec}",
            )
            subtitle_data.save(output_sub_final_filepath)
            matching_subtitles[subtitle_language] = MatchingSubtitle(
                origin="internal",
                filepath=output_sub_final_filepath,
                data=subtitle_data,
            )

    logger.info(f"Matching subtitles: {matching_subtitles}\n")

    # Having matching JP subtitles is required
    if "ja" not in matching_subtitles:
        raise Exception("Could not find Japanese subtitles. Skipping...")

    # Default sync_external_subs to True if not set
    if "sync_external_subs" not in locals():
        sync_external_subs = True

    # Override with config if explicitly set (e.g., --no-sync flag)
    if config.sync_external_subs is not None:
        sync_external_subs = config.sync_external_subs

    return (
        episode_number,
        episode_number_pretty,
        matching_subtitles,
        subtitles_dict_remembered,
        sync_external_subs,
        main_mkv,
        audio_index,
    )


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


def _compute_overlap_score(sub_a, sub_b, sample_size: int = 50) -> tuple[float, float]:
    """Measure alignment between two subtitle tracks.

    Samples lines from sub_a, finds closest-in-time match in sub_b, and computes:
      - overlap_ratio: fraction of sampled lines that overlap with a sub_b line
      - mean_offset_ms: average absolute start-time offset for overlapping pairs

    Returns:
        (overlap_ratio, mean_offset_ms)  — both 0.0 when either track is empty.
    """
    lines_a = [e for e in sub_a if e.type == "Dialogue"]
    lines_b = [e for e in sub_b if e.type == "Dialogue"]

    if not lines_a or not lines_b:
        return 0.0, 0.0

    # Sample evenly across the track
    step = max(1, len(lines_a) // sample_size)
    sampled = lines_a[::step][:sample_size]

    overlaps = 0
    offsets = []

    for a_line in sampled:
        a_start, a_end = a_line.start, a_line.end
        best_offset = None

        for b_line in lines_b:
            b_start, b_end = b_line.start, b_line.end
            # Check temporal overlap
            if a_start < b_end and b_start < a_end:
                offset = abs(a_start - b_start)
                if best_offset is None or offset < best_offset:
                    best_offset = offset

        if best_offset is not None:
            overlaps += 1
            offsets.append(best_offset)

    overlap_ratio = overlaps / len(sampled) if sampled else 0.0
    mean_offset = sum(offsets) / len(offsets) if offsets else 0.0
    return overlap_ratio, mean_offset


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
                    pre_overlap, pre_offset = _compute_overlap_score(
                        sub.data, internal_ref.data
                    )
                    logger.info(
                        f"[E{episode_number}] Pre-sync {lang}: "
                        f"overlap={pre_overlap:.1%}, mean_offset={pre_offset:.0f}ms"
                    )

                    # Skip ffsubsync if already well-aligned
                    if pre_overlap >= 0.5 and pre_offset < 3000:
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
                        synced_filepath = os.path.join(
                            tmp_output_folder,
                            f"synced_{lang}_{os.path.basename(sub.filepath)}",
                        )

                        # Run ffsubsync
                        subprocess.run(
                            [
                                "ffsubsync",
                                internal_ref.filepath,
                                "-i",
                                sub.filepath,
                                "-o",
                                synced_filepath,
                            ],
                            check=True,
                            capture_output=True,
                        )

                        # Load synced subtitle and check post-sync quality
                        synced_data = load_subtitle_file(synced_filepath)
                        post_overlap, post_offset = _compute_overlap_score(
                            synced_data, internal_ref.data
                        )
                        logger.info(
                            f"[E{episode_number}] Post-sync {lang}: "
                            f"overlap={post_overlap:.1%}, mean_offset={post_offset:.0f}ms"
                        )

                        # Fall back to original if sync degraded alignment
                        overlap_dropped = pre_overlap - post_overlap > 0.10
                        offset_increased = post_offset - pre_offset > 2000
                        if overlap_dropped or offset_increased:
                            logger.warning(
                                f"[E{episode_number}] ffsubsync degraded {lang} alignment "
                                f"(overlap {pre_overlap:.1%}->{post_overlap:.1%}, "
                                f"offset {pre_offset:.0f}ms->{post_offset:.0f}ms). "
                                f"Falling back to original subs."
                            )
                        else:
                            subtitles[lang] = MatchingSubtitle(
                                origin="external", filepath=synced_filepath, data=synced_data
                            )
                            logger.info(
                                f"Synced {lang} subtitles against internal reference"
                            )
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
            sentence = process_subtitle_line(line, config)
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
    segment_index = 0

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
                segment_index += 1
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
                    segment_index += 1
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
        reason = (
            f"content too long ({len(sentence_japanese)} > {MAX_SEGMENT_CONTENT_LENGTH})"
        )
        logs.append(f"[yellow]Skipping segment: {reason}[/yellow]")
        return logs, build_ignored_segment(reason), "ignored"

    audio_filename = f"{segment_hash}.mp3"
    screenshot_filename = f"{segment_hash}.webp"
    video_filename = f"{segment_hash}.mp4"

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
                    "scale=960:540",
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

        except Exception as err:
            logger.error(f"[red]Error creating screenshot '{screenshot_filename}': {err}[/red]")
            return logs, None, "screenshot"

        video_path = os.path.join(output_path, video_filename)
        video_length_delta = end_time_delta - start_time_delta

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
                    "scale=1280:720,setsar=1",
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
    }

    logs.append("[green]Segment saved![/green]")
    return logs, segment_dict, None


# ----------------------------------------------------------------------------
# Summary Report Functions
# ----------------------------------------------------------------------------


def display_episode_summary_report(episode_stats: dict) -> None:
    """Display a summary report of episode processing, flagging episodes with low segment counts.

    Args:
        episode_stats: Dict mapping {folder_name: {episode_number: segment_count}}
    """
    if not episode_stats:
        return

    console.print("\n[green bold]=== Episode Processing Summary ===[/green bold]")

    for folder_name, episodes in episode_stats.items():
        if not episodes:
            continue

        # Sort episodes by number
        sorted_episodes = sorted(episodes.items())
        episode_numbers = [ep for ep, _ in sorted_episodes]
        segment_counts = [count for _, count in sorted_episodes]

        # Calculate statistics
        avg_segments = sum(segment_counts) / len(segment_counts)
        min_segments = min(segment_counts)
        max_segments = max(segment_counts)

        console.print(f"\n[cyan bold]Folder: {folder_name}[/cyan bold]")
        console.print(f"  Episodes processed: {len(sorted_episodes)}")
        console.print(f"  Average segments: {avg_segments:.1f}")
        console.print(f"  Min segments: {min_segments}")
        console.print(f"  Max segments: {max_segments}")

        # Find episodes with significantly low segment counts (difference > 400)
        flagged_episodes = []
        for ep_num, count in sorted_episodes:
            if avg_segments - count > 400:
                flagged_episodes.append((ep_num, count, avg_segments - count))

        # Display individual episode counts
        console.print(f"\n  [bold]Episode segment counts:[/bold]")
        for ep_num, count in sorted_episodes:
            is_low = avg_segments - count > 400
            color = "red" if is_low else "green"
            console.print(f"    Episode {ep_num}: [{color}]{count}[/{color}]")

        # Display warning for low segment episodes
        if flagged_episodes:
            console.print(f"\n  [red bold]⚠ Warning: Episodes with abnormally low segment counts:[/red bold]")
            console.print("  [red]These episodes may have had issues during processing.[/red]")
            for ep_num, count, diff in flagged_episodes:
                console.print(f"    [red]Episode {ep_num}: {count} segments ({diff:.0f} below average)[/red]")
            console.print(f"\n  [yellow]Tip: Use -e {','.join(str(e) for e, _, _ in flagged_episodes)} to reprocess specific episodes[/yellow]")

    console.print("\n[green bold]=== Summary Complete ===[/green bold]\n")
