"""Segment extractor — extract segments from MKV files.

Extracts episode 1 first for validation, then remaining episodes.
Runs segment QC after extraction.

Usage:
    uv run process-media --anilist-id 21804 --input /mnt/storage/saiki-k \\
        --output /mnt/storage/saiki-k-output --subtitle-indices 2,4 --parallel
"""

import argparse
import logging
import os
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
    config,
    hash_salt: str,
    translator,
) -> dict[int, int | None]:
    """Extract segments from MKV files. Returns {episode_num: segment_count}."""
    import re
    import subprocess
    from multiprocessing.pool import ThreadPool as Pool

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

    pool = Pool(config.pool_size)
    episode_stats = {}
    input_folder = str(config.input_folder)

    for episode_number, filepath in episode_files:
        console.print(f"\n[cyan bold]{'='*60}[/cyan bold]")
        console.print(
            f"[cyan bold]Episode {episode_number}: {os.path.basename(filepath)}[/cyan bold]"
        )
        console.print(f"[cyan bold]{'='*60}[/cyan bold]")

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

                tag_language_normalizer = {"fre": "fra", "ger": "deu"}
                if tag_language_normalizer.get(tag_language):
                    tag_language = tag_language_normalizer.get(tag_language)

                subtitle_language = babelfish.Language(tag_language).alpha2
                logger.info(f"Extracting subtitle stream #{index}: {subtitle_language} ({codec})")

                if subtitle_language not in SUPPORTED_LANGUAGES:
                    logger.info(f"Language {subtitle_language} not supported, skipping")
                    continue

                format_map = {"subrip": "srt", "ass": "ass", "ssa": "ssa"}
                ffmpeg_format = format_map.get(codec, codec)
                output_ext = "srt" if codec == "subrip" else codec
                output_sub_path = os.path.join(tmp_folder, f"tmp_{index}.{output_ext}")
                subprocess.call(
                    [
                        "ffmpeg", "-y", "-i", filepath, "-map", f"0:{index}",
                        "-f", ffmpeg_format, output_sub_path,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
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
                f"{lang}: {len(sub.data)} lines"
                for lang, sub in sorted(matching_subtitles.items())
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
                    audio_streams = [
                        s for s in file_probe["streams"] if s["codec_type"] == "audio"
                    ]
                    if audio_streams:
                        ep_audio_index = audio_streams[0]["index"]

            console.print(f"[cyan]Audio stream: #{ep_audio_index}[/cyan]")

            episode_folder = os.path.join(anime_folder, str(episode_number))
            os.makedirs(episode_folder, exist_ok=True)

            segment_count = process_episode_segments(
                pool,
                filepath,
                anime_folder,
                episode_number,
                str(episode_number),
                matching_subtitles,
                translator,
                anime_data,
                config,
                hash_salt,
                sync_external_subs=True,
                audio_index=ep_audio_index,
            )

            episode_stats[episode_number] = segment_count
            console.print(
                f"[green bold]Episode {episode_number}: "
                f"{segment_count} segments generated[/green bold]"
            )
            discord_audit.post(
                f"E{episode_number}: {segment_count or '?'} segments extracted",
                stage="extracting",
            )

        except Exception:
            logger.error(
                f"[red]Failed to process episode {episode_number}[/red]", exc_info=True
            )
            episode_stats[episode_number] = None

    pool.close()
    pool.join()

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
    parser.add_argument("--dry-run", action="store_true", help="Parse only, no media generation")
    parser.add_argument("--parallel", action="store_true", help="Process episodes in parallel")
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

    # Discover MKV files
    input_folder = os.path.abspath(args.input)
    mkv_files = sorted(f for f in os.listdir(input_folder) if f.endswith(".mkv"))

    if not mkv_files:
        console.print(f"[red]No .mkv files found in {input_folder}[/red]")
        return 1

    console.print(f"[green]Found {len(mkv_files)} MKV file(s)[/green]")

    # Map filenames to episode numbers
    from guessit import guessit

    episode_files = []
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
        episode_files.append((ep_num, filepath))

    episode_files.sort(key=lambda x: x[0])

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

    info_json_path = os.path.join(anime_folder, "info.json")
    hash_salt = save_info_json(info_json_path, anime_data, str(anime_data.id))
    console.print(f"[green]Saved info.json (salt: {hash_salt[:8]}...)[/green]")

    import deepl as deepl_lib

    from nadeshiko_dev_tools.common.config import ProcessingConfig

    deepl_token = os.getenv("TOKEN")
    translator = deepl_lib.Translator(deepl_token) if deepl_token else None
    if not translator:
        logger.warning("No DeepL token — segments missing EN/ES will be skipped")

    config = ProcessingConfig(
        input_folder=input_folder,
        dryrun=args.dry_run,
        parallel=args.parallel,
    )

    discord_audit.post("Starting extraction", stage="started")

    # ── Extract episode 1 (validate release/subs) ──
    console.print("\n[magenta bold]Extract first episode (validation)[/magenta bold]")

    first_ep = episode_files[0]
    first_config = ProcessingConfig(
        input_folder=input_folder,
        dryrun=args.dry_run,
        parallel=False,
    )

    episode_stats = extract_episodes(
        [first_ep], anime_data, anime_folder, subtitle_indices,
        args.audio_index, first_config, hash_salt, translator,
    )

    # ── QC episode 1 ──
    if not args.dry_run:
        qc_report = run_qc(anime_folder, episodes={first_ep[0]}, checks={"segments"})
        if not qc_report.summary():
            discord_audit.post(
                f"QC FAILED on E{first_ep[0]}: {'; '.join(qc_report.errors)}",
                stage="qc_ep1_failed",
                color=discord_audit.COLOR_FAILURE,
            )
            console.print("[red bold]QC failed on first episode — stopping.[/red bold]")
            return 1
        discord_audit.post(f"E{first_ep[0]} QC passed", stage="qc_ep1_passed",
                           color=discord_audit.COLOR_SUCCESS)

    # ── Extract remaining episodes ──
    remaining = [ef for ef in episode_files if ef[0] != first_ep[0]]

    if remaining:
        console.print(
            f"\n[magenta bold]Extract remaining {len(remaining)} episodes[/magenta bold]"
        )
        remaining_stats = extract_episodes(
            remaining, anime_data, anime_folder, subtitle_indices,
            args.audio_index, config, hash_salt, translator,
        )
        episode_stats.update(remaining_stats)

    # Print extraction summary
    console.print(f"\n[green bold]{'='*60}[/green bold]")
    console.print("[green bold]Extraction Complete[/green bold]")
    console.print(f"[green bold]{'='*60}[/green bold]")
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
        discord_audit.post("Extraction QC passed", stage="done",
                           color=discord_audit.COLOR_SUCCESS)
    else:
        discord_audit.post(
            f"QC FAILED: {'; '.join(qc_report.errors)}",
            stage="qc_failed",
            color=discord_audit.COLOR_FAILURE,
        )

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
