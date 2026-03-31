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

console = Console()
logger = logging.getLogger("tokenize-media")
handler = RichHandler(console=console, show_time=True, show_path=False, markup=True)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def batch_tokenizer(media_folder: str, episodes: list[int] | None = None):
    """Run Sudachi + UniDic tokenization on all segments and update _data.json files."""
    from nadeshiko_dev_tools.tokenizer.tokenizer import JapaneseTokenizer, UnidicTokenizer

    console.print("[cyan bold]Running batch tokenizer...[/cyan bold]")
    sudachi = JapaneseTokenizer()
    unidic = UnidicTokenizer()
    console.print("[green]Tokenizers loaded (Sudachi + UniDic)[/green]")

    episode_dirs = sorted(
        (int(d), os.path.join(media_folder, d))
        for d in os.listdir(media_folder)
        if os.path.isdir(os.path.join(media_folder, d)) and d.isdigit()
    )

    if episodes:
        episode_dirs = [(n, p) for n, p in episode_dirs if n in episodes]

    for ep_num, ep_path in episode_dirs:
        data_path = os.path.join(ep_path, "_data.json")
        if not os.path.exists(data_path):
            logger.warning(f"E{ep_num}: No _data.json, skipping")
            continue

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
            console.print(f"  E{ep_num}: Already tokenized ({len(segments)} segments)")
            continue

        with open(data_path, "w") as f:
            json.dump(data, f, ensure_ascii=False)

        console.print(f"  E{ep_num}: Tokenized {tokenized}/{len(segments)} segments")

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

    # QC: validate tokenizer output
    from nadeshiko_dev_tools.common.quality_check import run_qc

    ep_set = set(ep_filter) if ep_filter else None
    report = run_qc(media_folder, episodes=ep_set, checks={"tokenizer"})
    passed = report.summary()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
