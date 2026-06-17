"""process-youtube — turn downloaded YouTube subtitles into _data.json segments.
Usage:
    uv run process-youtube /mnt/storage/yt/UCxxxxxxxxxxxxxxx
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from nadeshiko_dev_tools.common import discord_audit
from nadeshiko_dev_tools.common.quality_check import run_qc
from nadeshiko_dev_tools.common.translator import get_translator
from nadeshiko_dev_tools.youtube_segment_extractor.extractor import (
    list_video_folders,
    load_channel_info,
    process_video_folder,
)

load_dotenv()

console = Console()
logger = logging.getLogger("process-youtube")
handler = RichHandler(console=console, show_time=True, show_path=False, markup=True)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def process_videos(
    video_dirs: list[tuple[str, str]],
    hash_salt: str,
    translator,
    group_sentences: bool,
    llm_grouper=None,
) -> dict[str, int | None]:
    """Build _data.json for each video. Returns {video_id: segment_count | None}."""
    stats: dict[str, int | None] = {}
    for video_id, video_folder in video_dirs:
        console.print(f"\n[cyan bold]{'=' * 60}[/cyan bold]")
        console.print(f"[cyan bold]Video {video_id}[/cyan bold]")
        console.print(f"[cyan bold]{'=' * 60}[/cyan bold]")
        try:
            count = process_video_folder(
                video_folder,
                hash_salt,
                translator,
                group_sentences=group_sentences,
                llm_grouper=llm_grouper,
            )
            stats[video_id] = count
            console.print(f"[green bold]{video_id}: {count} segments generated[/green bold]")
            discord_audit.post(f"{video_id}: {count} segments", stage="processing")
        except Exception:
            logger.error(f"[red]Failed to process {video_id}[/red]", exc_info=True)
            stats[video_id] = None
    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process YouTube subtitle folders into segment data"
    )
    parser.add_argument("channel_folder", help="Path to a channel folder produced by fetch-youtube")
    parser.add_argument(
        "--no-group-sentences",
        action="store_true",
        help="Keep one segment per subtitle cue instead of grouping cues into sentences",
    )
    parser.add_argument(
        "--no-llm-grouping",
        action="store_true",
        help="Group cues into sentences with the Japanese punctuation/sentence-end heuristic "
        "instead of the LLM (the default). Use to avoid LLM calls or when OPENAI_API_KEY is "
        "unavailable.",
    )
    parser.add_argument(
        "--discord-audit", action="store_true", help="Send progress to DISCORD_AUDIT_WEBHOOK_URL"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    translator = get_translator()
    if not translator:
        console.print(
            "[yellow]No translator configured — JA lines without an EN/ES cue "
            "will be ignored[/yellow]"
        )

    channel_folder = os.path.abspath(args.channel_folder)
    if not os.path.isdir(channel_folder):
        console.print(f"[red]Not a directory: {channel_folder}[/red]")
        return 1

    try:
        info = load_channel_info(channel_folder)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return 1
    hash_salt = info["hash_salt"]
    channel_name = info.get("name", info.get("channel_id", ""))

    discord_audit.init(args.discord_audit, channel_name, info.get("channel_id", ""))

    video_dirs = list_video_folders(channel_folder)
    if not video_dirs:
        console.print(f"[yellow]No video folders with _meta.json under {channel_folder}[/yellow]")
        return 0

    console.print(f"[green]Channel: {channel_name}[/green]")
    console.print(f"[cyan]Processing {len(video_dirs)} video(s)[/cyan]")
    discord_audit.post("Starting processing", stage="started")

    group_sentences = not args.no_group_sentences

    llm_grouper = None
    if group_sentences and not args.no_llm_grouping:
        from nadeshiko_dev_tools.common.llm_translator import get_openai_translator

        llm_grouper = get_openai_translator()
        if llm_grouper is None:
            console.print(
                "[yellow]No OpenAI translator available (set OPENAI_API_KEY) — "
                "using punctuation heuristic for sentence grouping[/yellow]"
            )
        else:
            console.print("[cyan]Sentence grouping: LLM boundary segmentation[/cyan]")

    # ── Process the first video, then validate before committing to the rest ──
    first = video_dirs[0]
    stats = process_videos([first], hash_salt, translator, group_sentences, llm_grouper)

    if stats.get(first[0]) in (None, 0):
        discord_audit.post(
            f"{first[0]} failed validation — stopping",
            stage="validation_failed",
            color=discord_audit.COLOR_FAILURE,
        )
        console.print("[red bold]First video failed — stopping.[/red bold]")
        return 1

    # ── Process remaining videos ──
    stats.update(
        process_videos(video_dirs[1:], hash_salt, translator, group_sentences, llm_grouper)
    )

    # ── Summary ──
    console.print(f"\n[green bold]{'=' * 60}[/green bold]")
    console.print("[green bold]Processing Complete[/green bold]")
    console.print(f"[green bold]{'=' * 60}[/green bold]")
    for video_id, count in stats.items():
        if count is not None:
            console.print(f"  {video_id}: {count} segments")
        else:
            console.print(f"  {video_id}: [red]FAILED[/red]")

    # ── QC all segments ──
    console.print("\n[magenta bold]QC all segments[/magenta bold]")
    passed = run_qc(channel_folder, checks={"segments"}).summary()

    if passed:
        discord_audit.post("Processing QC passed", stage="done", color=discord_audit.COLOR_SUCCESS)
    else:
        discord_audit.post(
            "Processing QC FAILED", stage="qc_failed", color=discord_audit.COLOR_FAILURE
        )

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
