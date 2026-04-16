"""Segment extractor — extract segments from MKV files.

Extracts episode 1 first for validation, then remaining episodes.
Runs segment QC after extraction.

Usage:
    uv run process-media --anilist-id 21804 --input /mnt/storage/saiki-k \\
        --output /mnt/storage/saiki-k-output --subtitle-indices 2,4
"""

import argparse
import json
import logging
import os
import shutil
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

load_dotenv()

console = Console()
logger = logging.getLogger("process-media")
handler = RichHandler(console=console, show_time=True, show_path=False, markup=True)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def extract_episodes(
    episode_files: list[tuple[int, str]],
    anime_data,
    anime_folder: str,
    subtitle_indices: list[int],
    audio_index: int | None,
    input_folder: str,
    hash_salt: str,
    translator,
    dryrun: bool = False,
) -> dict[int, int | None]:
    """Extract segments from MKV files. Returns {episode_num: segment_count}."""
    import re
    import subprocess

    import babelfish
    import ffmpeg
    from guessit import guessit
    from langdetect import detect

    from nadeshiko_dev_tools.common import discord_audit
    from nadeshiko_dev_tools.segment_extractor.splitter import (
        MatchingSubtitle,
        process_episode_segments,
    )
    from nadeshiko_dev_tools.segment_extractor.utils.subtitle_utils import (
        SUPPORTED_LANGUAGES,
        load_subtitle_file,
    )

    episode_stats = {}
    tag_language_normalizer = {"fre": "fra", "ger": "deu"}

    for episode_number, filepath in episode_files:
        console.print(f"\n[cyan bold]{'=' * 60}[/cyan bold]")
        console.print(
            f"[cyan bold]Episode {episode_number}: {os.path.basename(filepath)}[/cyan bold]"
        )
        console.print(f"[cyan bold]{'=' * 60}[/cyan bold]")

        # Skip already-extracted episodes, or clean up partial extractions
        episode_folder = os.path.join(anime_folder, str(episode_number))
        existing_data_path = os.path.join(episode_folder, "_data.json")
        if os.path.isdir(episode_folder):
            if os.path.exists(existing_data_path):
                with open(existing_data_path) as f:
                    existing_data = json.load(f)
                expected_count = len(existing_data.get("segments", []))
                actual_count = len([f for f in os.listdir(episode_folder) if f.endswith(".mp4")])
                if expected_count > 0 and expected_count == actual_count:
                    console.print(
                        f"[yellow]Episode {episode_number}: already extracted "
                        f"({expected_count} segments), skipping.[/yellow]"
                    )
                    episode_stats[episode_number] = expected_count
                    continue
                logger.warning(
                    f"[yellow]Episode {episode_number}: partial extraction "
                    f"({actual_count} mp4 vs {expected_count} in _data.json), "
                    f"cleaning up and re-extracting[/yellow]"
                )
            else:
                orphan_count = len([f for f in os.listdir(episode_folder) if f.endswith(".mp4")])
                if orphan_count > 0:
                    logger.warning(
                        f"[yellow]Episode {episode_number}: {orphan_count} orphan mp4 files "
                        f"without _data.json, cleaning up and re-extracting[/yellow]"
                    )
            shutil.rmtree(episode_folder)

        try:
            tmp_folder = os.path.join(anime_folder, f"tmp_ep{episode_number}")
            os.makedirs(tmp_folder, exist_ok=True)

            matching_subtitles = {}
            file_probe = ffmpeg.probe(filepath)

            for stream in file_probe["streams"]:
                if stream["codec_type"] != "subtitle":
                    continue
                if stream["index"] not in subtitle_indices:
                    continue

                index = stream["index"]
                codec = stream["codec_name"]
                tag_language = stream.get("tags", {}).get("language", "jpn")
                tag_language = tag_language_normalizer.get(tag_language, tag_language)

                subtitle_language = babelfish.Language(tag_language).alpha2
                logger.info(f"Extracting subtitle stream #{index}: {subtitle_language} ({codec})")

                if subtitle_language not in SUPPORTED_LANGUAGES:
                    logger.info(f"Language {subtitle_language} not supported, skipping")
                    continue

                format_map = {"subrip": "srt", "ass": "ass", "ssa": "ssa"}
                ffmpeg_format = format_map.get(codec, codec)
                output_ext = "srt" if codec == "subrip" else codec
                output_sub_path = os.path.join(tmp_folder, f"tmp_{index}.{output_ext}")
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        filepath,
                        "-map",
                        f"0:{index}",
                        "-f",
                        ffmpeg_format,
                        output_sub_path,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                    check=True,
                )

                subtitle_data = load_subtitle_file(output_sub_path)
                logger.info(f"  -> {len(subtitle_data)} lines")

                final_sub_path = os.path.join(
                    tmp_folder,
                    f"{anime_data.id} {episode_number}.{subtitle_language}.{output_ext}",
                )
                subtitle_data.save(final_sub_path)

                if subtitle_language in matching_subtitles:
                    existing = matching_subtitles[subtitle_language]
                    if len(subtitle_data) <= len(existing.data):
                        logger.info(f"  -> Already have better {subtitle_language} subs, skipping")
                        continue

                matching_subtitles[subtitle_language] = MatchingSubtitle(
                    origin="internal",
                    filepath=final_sub_path,
                    data=subtitle_data,
                )

            # Discover external subtitle files
            for ext_file in os.listdir(input_folder):
                if not (ext_file.endswith(".ass") or ext_file.endswith(".srt")):
                    continue
                ext_info = guessit(ext_file)
                ext_ep = ext_info.get("episode")
                if isinstance(ext_ep, list):
                    ext_ep = ext_ep[0]
                if ext_ep is None:
                    ep_match = re.search(r"(?:E|-)[\s]*(\d{1,2})", ext_file)
                    if ep_match:
                        ext_ep = int(ep_match.group(1))
                if ext_ep is not None and int(ext_ep) == episode_number:
                    ext_path = os.path.join(input_folder, ext_file)
                    ext_data = load_subtitle_file(ext_path)

                    # Prefer language from filename, fall back to content detection
                    ext_lang = None
                    guessit_lang = ext_info.get("subtitle_language")
                    if guessit_lang:
                        ext_lang = (
                            str(guessit_lang.alpha2)
                            if hasattr(guessit_lang, "alpha2")
                            else str(guessit_lang)
                        )
                    if not ext_lang or ext_lang not in SUPPORTED_LANGUAGES:
                        ext_text = " ".join([e.text for e in ext_data if hasattr(e, "text")])
                        try:
                            ext_lang = detect(ext_text)
                        except Exception:
                            continue

                    if ext_lang not in SUPPORTED_LANGUAGES:
                        continue
                    if ext_lang in matching_subtitles and len(ext_data) <= len(
                        matching_subtitles[ext_lang].data
                    ):
                        continue
                    logger.info(
                        f"Found external {ext_lang} subtitle: {ext_file} ({len(ext_data)} lines)"
                    )
                    matching_subtitles[ext_lang] = MatchingSubtitle(
                        origin="external",
                        filepath=ext_path,
                        data=ext_data,
                    )

            if "ja" not in matching_subtitles:
                logger.error(
                    f"[red]No Japanese subtitles found for episode {episode_number}![/red]"
                )
                continue

            lang_summary = ", ".join(
                f"{lang}: {len(sub.data)} lines" for lang, sub in sorted(matching_subtitles.items())
            )
            console.print(f"[green]Subtitles: {lang_summary}[/green]")

            # Detect audio index
            ep_audio_index = audio_index
            if ep_audio_index is None:
                for stream in file_probe["streams"]:
                    if stream["codec_type"] == "audio":
                        lang = stream.get("tags", {}).get("language", "").lower()
                        if lang in ("jpn", "ja", "japanese"):
                            ep_audio_index = stream["index"]
                            break
                if ep_audio_index is None:
                    audio_streams = [s for s in file_probe["streams"] if s["codec_type"] == "audio"]
                    if audio_streams:
                        ep_audio_index = audio_streams[0]["index"]

            console.print(f"[cyan]Audio stream: #{ep_audio_index}[/cyan]")

            os.makedirs(episode_folder, exist_ok=True)

            segment_count = process_episode_segments(
                filepath,
                anime_folder,
                episode_number,
                matching_subtitles,
                translator,
                anime_data,
                hash_salt,
                sync_external_subs=True,
                dryrun=dryrun,
                audio_index=ep_audio_index,
                input_folder=input_folder,
            )

            # Self-verify: _data.json must exist and match files on disk
            data_path = os.path.join(episode_folder, "_data.json")
            if not dryrun and segment_count and segment_count > 0:
                if not os.path.exists(data_path):
                    logger.error(
                        f"[red]Episode {episode_number}: {segment_count} segments reported "
                        f"but _data.json is missing![/red]"
                    )
                    episode_stats[episode_number] = None
                    continue
                with open(data_path) as f:
                    ep_data = json.load(f)
                expected = {s["segment_hash"] for s in ep_data.get("segments", [])}
                actual = {f[:-4] for f in os.listdir(episode_folder) if f.endswith(".mp4")}
                missing = expected - actual
                if missing:
                    logger.error(
                        f"[red]Episode {episode_number}: {len(missing)}/{len(expected)} "
                        f"segments missing media files — extraction incomplete![/red]"
                    )
                    episode_stats[episode_number] = None
                    continue

            episode_stats[episode_number] = segment_count
            console.print(
                f"[green bold]Episode {episode_number}: "
                f"{segment_count} segments generated[/green bold]"
            )
            discord_audit.post(
                f"E{episode_number}: {segment_count or '?'} segments extracted",
                stage="extracting",
            )

            # Clean up tmp folder for this episode
            if os.path.isdir(tmp_folder):
                shutil.rmtree(tmp_folder)
                logger.info(f"Cleaned up {tmp_folder}")

        except Exception:
            logger.error(f"[red]Failed to process episode {episode_number}[/red]", exc_info=True)
            episode_stats[episode_number] = None

    return episode_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Extract segments from MKV files")
    parser.add_argument("--anilist-id", type=int, required=True, help="AniList media ID")
    parser.add_argument("--input", required=True, help="Folder with .mkv files")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument(
        "--subtitle-indices",
        required=True,
        help="Comma-separated subtitle stream indices (e.g. 2,4)",
    )
    parser.add_argument(
        "--audio-index", type=int, default=None, help="Audio stream index (default: auto-detect)"
    )
    parser.add_argument(
        "--episodes", default=None, help="Comma-separated episode numbers (default: all)"
    )
    parser.add_argument(
        "--episode-range",
        default=None,
        help="Episode range as START-END inclusive (e.g. 1-22). Combines with --episodes.",
    )
    parser.add_argument(
        "--source-pattern",
        default=None,
        help="Only use MKVs matching this substring as video source (e.g. '[Trix]'). "
        "Other MKVs in the folder are still used for chapter detection.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse only, no media generation")
    parser.add_argument(
        "--discord-audit", action="store_true", help="Send progress to DISCORD_AUDIT_WEBHOOK_URL"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure audit webhook
    from nadeshiko_dev_tools.common import discord_audit
    from nadeshiko_dev_tools.common.quality_check import run_qc

    # Parse episode filter
    episodes_filter = None
    if args.episodes:
        episodes_filter = {int(e.strip()) for e in args.episodes.split(",")}
    if args.episode_range:
        start, end = args.episode_range.split("-")
        range_set = set(range(int(start), int(end) + 1))
        episodes_filter = (episodes_filter & range_set) if episodes_filter else range_set
    if episodes_filter:
        console.print(f"[cyan]Episode filter: {sorted(episodes_filter)}[/cyan]")

    subtitle_indices = [int(i.strip()) for i in args.subtitle_indices.split(",")]
    console.print(f"[cyan]Subtitle stream indices: {subtitle_indices}[/cyan]")

    # Fetch AniList data
    from nadeshiko_dev_tools.common.anilist import CachedAnilist

    console.print(f"[cyan]Fetching AniList data for ID {args.anilist_id}...[/cyan]")
    anilist = CachedAnilist()
    anime_data = anilist.get_anime_with_id(args.anilist_id)
    console.print(f"[green]Found: {anime_data.title.romaji}[/green]")

    discord_audit.init(args.discord_audit, anime_data.title.romaji, args.anilist_id)

    # Discover MKV files (filter by --source-pattern if provided)
    input_folder = os.path.abspath(args.input)
    mkv_files = sorted(f for f in os.listdir(input_folder) if f.endswith(".mkv"))
    if args.source_pattern:
        mkv_files = [f for f in mkv_files if args.source_pattern in f]
        console.print(f"[cyan]Source pattern filter: '{args.source_pattern}'[/cyan]")

    if not mkv_files:
        console.print(f"[red]No .mkv files found in {input_folder}[/red]")
        return 1

    console.print(f"[green]Found {len(mkv_files)} MKV file(s)[/green]")

    # Map filenames to episode numbers
    from guessit import guessit

    seen_episodes = {}
    for filename in mkv_files:
        filepath = os.path.join(input_folder, filename)
        episode_info = guessit(filename)
        ep_num = episode_info.get("episode")
        if ep_num is None:
            logger.warning(f"Could not determine episode number for {filename}, skipping")
            continue
        if isinstance(ep_num, list):
            ep_num = ep_num[0]
        season = episode_info.get("season")
        if season == 0:
            ep_num = 0
        if episodes_filter and ep_num not in episodes_filter:
            continue
        if ep_num in seen_episodes:
            logger.info(
                f"Skipping duplicate MKV for episode {ep_num}: {filename} "
                f"(using {os.path.basename(seen_episodes[ep_num])})"
            )
            continue
        seen_episodes[ep_num] = filepath

    episode_files = sorted(seen_episodes.items())

    if not episode_files:
        console.print("[yellow]No episodes to process after filtering[/yellow]")
        return 0

    console.print(
        f"[cyan]Processing {len(episode_files)} episode(s): "
        f"{[ep for ep, _ in episode_files]}[/cyan]"
    )

    # Create output structure
    output_folder = os.path.abspath(args.output)
    anime_folder = os.path.join(output_folder, str(anime_data.id))
    os.makedirs(anime_folder, exist_ok=True)

    from nadeshiko_dev_tools.common.file_utils import save_info_json

    hash_salt = save_info_json(anime_folder, anime_data)
    console.print(f"[green]Saved _info.json (salt: {hash_salt[:8]}...)[/green]")

    import deepl as deepl_lib

    deepl_token = os.getenv("TOKEN")
    translator = deepl_lib.Translator(deepl_token) if deepl_token else None
    if not translator:
        logger.warning("No DeepL token — segments missing EN/ES will be skipped")

    discord_audit.post("Starting extraction", stage="started")

    # ── Extract episode 1 first (validate release/subs before committing to full run) ──
    first_ep = episode_files[0]

    episode_stats = extract_episodes(
        [first_ep],
        anime_data,
        anime_folder,
        subtitle_indices,
        args.audio_index,
        input_folder,
        hash_salt,
        translator,
        args.dry_run,
    )

    first_ep_count = episode_stats.get(first_ep[0])
    if not args.dry_run and (first_ep_count is None or first_ep_count == 0):
        discord_audit.post(
            f"E{first_ep[0]} failed validation — stopping",
            stage="qc_ep1_failed",
            color=discord_audit.COLOR_FAILURE,
        )
        console.print("[red bold]First episode failed — stopping.[/red bold]")
        return 1

    # ── Extract remaining episodes ──
    remaining = [ef for ef in episode_files if ef[0] != first_ep[0]]

    if remaining:
        remaining_stats = extract_episodes(
            remaining,
            anime_data,
            anime_folder,
            subtitle_indices,
            args.audio_index,
            input_folder,
            hash_salt,
            translator,
            args.dry_run,
        )
        episode_stats.update(remaining_stats)

    # Print extraction summary
    console.print(f"\n[green bold]{'=' * 60}[/green bold]")
    console.print("[green bold]Extraction Complete[/green bold]")
    console.print(f"[green bold]{'=' * 60}[/green bold]")
    for ep, count in sorted(episode_stats.items()):
        if count is not None:
            console.print(f"  Episode {ep:>2}: {count} segments")
        else:
            console.print(f"  Episode {ep:>2}: [red]FAILED[/red]")

    if args.dry_run:
        console.print("\n[yellow]Dry-run complete.[/yellow]")
        return 0

    # ── QC all segments ──
    console.print("\n[magenta bold]QC all segments[/magenta bold]")

    ep_filter = episodes_filter or {ep for ep, _ in episode_files}
    qc_report = run_qc(anime_folder, episodes=ep_filter, checks={"segments"})
    passed = qc_report.summary()

    if passed:
        discord_audit.post("Extraction QC passed", stage="done", color=discord_audit.COLOR_SUCCESS)
    else:
        discord_audit.post(
            f"QC FAILED: {'; '.join(qc_report.errors)}",
            stage="qc_failed",
            color=discord_audit.COLOR_FAILURE,
        )

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
