"""Batch NSFW tagger — classify screenshots and update _data.json.

Usage:
    uv run tag-media /mnt/storage/output/21804
    uv run tag-media /mnt/storage/output/21804 --episodes 1,2
"""

import argparse
import json
import logging
import os
import sys

from nadeshiko_dev_tools.common.cuda_setup import ensure_nvidia_cuda12_libs

ensure_nvidia_cuda12_libs()

from rich.console import Console  # noqa: E402
from rich.logging import RichHandler  # noqa: E402

console = Console()
logger = logging.getLogger("tag-media")
handler = RichHandler(console=console, show_time=True, show_path=False, markup=True)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def batch_tagger(media_folder: str, episodes: list[int] | None = None):
    """Run NSFW tagger on all screenshots and update _data.json files."""
    from nadeshiko_dev_tools.nsfw_tagger.classifier import WDTagger

    console.print("[cyan bold]Running batch NSFW tagger...[/cyan bold]")
    tagger = WDTagger()
    console.print("[green]Tagger loaded (GPU)[/green]")

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
        to_tag = []
        for i, seg in enumerate(segments):
            if seg.get("content_analysis") is not None:
                continue
            screenshot = seg.get("files", {}).get("screenshot")
            if screenshot:
                img_path = os.path.join(ep_path, screenshot)
                if os.path.exists(img_path):
                    to_tag.append((i, img_path))

        if not to_tag:
            console.print(f"  E{ep_num}: Already tagged ({len(segments)} segments)")
            continue

        console.print(f"  E{ep_num}: Tagging {len(to_tag)}/{len(segments)} screenshots...")

        batch_size = 32
        for chunk_start in range(0, len(to_tag), batch_size):
            chunk = to_tag[chunk_start : chunk_start + batch_size]
            image_paths = [path for _, path in chunk]
            results = tagger.classify_batch(image_paths)

            for (seg_idx, _), result in zip(chunk, results, strict=True):
                segments[seg_idx]["content_rating"] = result.content_rating
                segments[seg_idx]["content_analysis"] = {
                    "scores": result.rating_scores,
                    "tags": result.tags,
                }

        with open(data_path, "w") as f:
            json.dump(data, f, ensure_ascii=False)

        console.print(f"  E{ep_num}: Done — tagged {len(to_tag)} segments")

    console.print("[green bold]Batch tagger complete[/green bold]")


def main():
    parser = argparse.ArgumentParser(description="Batch NSFW tagger for processed segments")
    parser.add_argument("media_folder", help="Path to media folder (e.g. output/21804)")
    parser.add_argument("--episodes", default=None, help="Comma-separated episode numbers")
    args = parser.parse_args()

    media_folder = os.path.abspath(args.media_folder)
    ep_filter = None
    if args.episodes:
        ep_filter = [int(e.strip()) for e in args.episodes.split(",")]

    batch_tagger(media_folder, ep_filter)

    # QC: validate tagger output
    from nadeshiko_dev_tools.common.quality_check import run_qc

    ep_set = set(ep_filter) if ep_filter else None
    report = run_qc(media_folder, episodes=ep_set, checks={"tagger"})
    passed = report.summary()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
