"""process-youtube — turn downloaded YouTube subtitles into _data.json segments.

Reads each video folder under a channel folder produced by `fetch-youtube` and
writes a _data.json with one segment per JA subtitle line, EN/ES merged from
overlapping cues (DeepL fills any per-line gaps when TOKEN is set).

Usage:
    uv run process-youtube /mnt/storage/yt/UCxxxxxxxxxxxxxxx
"""

import argparse
import logging
import os
import sys

import deepl as deepl_lib
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from nadeshiko_dev_tools.youtube_extractor.process import process_channel

load_dotenv()

console = Console()
logger = logging.getLogger("process-youtube")
handler = RichHandler(console=console, show_time=True, show_path=False, markup=True)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process YouTube subtitle folders into segment data"
    )
    parser.add_argument(
        "channel_folder",
        help="Path to a channel folder produced by fetch-youtube",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    deepl_token = os.getenv("TOKEN")
    translator = deepl_lib.Translator(deepl_token) if deepl_token else None
    if not translator:
        console.print(
            "[yellow]No DeepL token — JA lines without an EN/ES cue will be ignored[/yellow]"
        )

    channel_folder = os.path.abspath(args.channel_folder)
    if not os.path.isdir(channel_folder):
        console.print(f"[red]Not a directory: {channel_folder}[/red]")
        return 1

    console.print(f"[cyan]Processing channel folder: {channel_folder}[/cyan]")
    processed, total_segments, failed = process_channel(channel_folder, translator)

    console.print(
        f"\n[green bold]Done: {processed} video(s) processed, "
        f"{total_segments} segments total, {failed} failed[/green bold]"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
