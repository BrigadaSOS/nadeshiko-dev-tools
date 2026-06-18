"""Batch tokenizer — run Sudachi + UniDic on segments and update _data.json.

Usage:
    uv run tokenize-media /mnt/storage/output/21804
    uv run tokenize-media /mnt/storage/output/21804 --episodes 1,2
"""

import argparse
import json
import logging
import os
import sys

from rich.console import Console
from rich.logging import RichHandler

from nadeshiko_dev_tools.common.file_utils import atomic_write_json

console = Console()
logger = logging.getLogger("tokenize-media")
handler = RichHandler(console=console, show_time=True, show_path=False, markup=True)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def batch_tokenizer(media_folder: str, episodes: list[int] | None = None):
    """Run Sudachi + UniDic tokenization on all segments and update _data.json files.

    Discovers any subfolder containing a _data.json — works for both anime
    (digit-named episode folders) and YouTube (video_id folders).
    """
    from nadeshiko_dev_tools.tokenizer.tokenizer import JapaneseTokenizer, UnidicTokenizer

    console.print("[cyan bold]Running batch tokenizer...[/cyan bold]")
    sudachi = JapaneseTokenizer()
    unidic = UnidicTokenizer()
    console.print("[green]Tokenizers loaded (Sudachi + UniDic)[/green]")

    target_dirs = sorted(
        (d, os.path.join(media_folder, d))
        for d in os.listdir(media_folder)
        if os.path.isdir(os.path.join(media_folder, d))
        and os.path.exists(os.path.join(media_folder, d, "_data.json"))
    )

    if episodes:
        target_dirs = [(n, p) for n, p in target_dirs if n.isdigit() and int(n) in episodes]

    for label, dir_path in target_dirs:
        data_path = os.path.join(dir_path, "_data.json")
        with open(data_path) as f:
            data = json.load(f)

        segments = data.get("segments", [])
        tokenized = 0
        for seg in segments:
            if seg.get("pos_analysis") is not None:
                continue
            ja = seg.get("content_ja", "")
            if ja:
                seg["pos_analysis"] = {
                    "sudachi": sudachi.tokenize(ja),
                    "unidic": unidic.tokenize(ja),
                }
                tokenized += 1

        if tokenized == 0:
            console.print(f"  {label}: Already tokenized ({len(segments)} segments)")
            continue

        atomic_write_json(data_path, data)

        console.print(f"  {label}: Tokenized {tokenized}/{len(segments)} segments")

    console.print("[green bold]Batch tokenizer complete[/green bold]")


def main():
    parser = argparse.ArgumentParser(description="Batch tokenizer for processed segments")
    parser.add_argument("media_folder", help="Path to media folder (e.g. output/21804)")
    parser.add_argument("--episodes", default=None, help="Comma-separated episode numbers")
    args = parser.parse_args()

    media_folder = os.path.abspath(args.media_folder)
    ep_filter = None
    if args.episodes:
        ep_filter = [int(e.strip()) for e in args.episodes.split(",")]

    batch_tokenizer(media_folder, ep_filter)

    # QC is anime-specific (assumes digit-named episode folders). Skip for
    # YouTube channels, where the _info.json marks media_source as "youtube".
    info_path = os.path.join(media_folder, "_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        if info.get("media_source") == "youtube":
            console.print("[cyan]YouTube source — skipping QC[/cyan]")
            return 0

    from nadeshiko_dev_tools.common.quality_check import run_qc

    ep_set = set(ep_filter) if ep_filter else None
    report = run_qc(media_folder, episodes=ep_set, checks={"tokenizer"})
    passed = report.summary()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
