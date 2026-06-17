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
from nadeshiko_dev_tools.common.file_utils import atomic_write_json, discover_data_dirs

ensure_nvidia_cuda12_libs()

from rich.console import Console  # noqa: E402
from rich.logging import RichHandler  # noqa: E402

console = Console()
logger = logging.getLogger("tag-media")
handler = RichHandler(console=console, show_time=True, show_path=False, markup=True)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _default_batch_size(provider: str) -> int:
    if provider == "CPUExecutionProvider":
        return int(os.getenv("NADESHIKO_NSFW_CPU_BATCH_SIZE", "4"))
    return int(os.getenv("NADESHIKO_NSFW_GPU_BATCH_SIZE", "32"))

def batch_tagger(
    media_folder: str,
    episodes: list[int] | None = None,
    batch_size: int | None = None,
    force: bool = False,
):
    """Run NSFW tagger on all screenshots and update _data.json files.

    _data.json is rewritten after every batch so an interrupted CPU run can
    resume (already-tagged segments are skipped on the next run).
    """
    from nadeshiko_dev_tools.nsfw_tagger.classifier import WDTagger

    console.print("[cyan bold]Running batch NSFW tagger...[/cyan bold]")
    tagger = WDTagger()
    batch_size = batch_size or _default_batch_size(tagger.active_provider)
    console.print(
        f"[green]Tagger loaded ({tagger.active_provider}); batch_size={batch_size}[/green]"
    )

    data_dirs = discover_data_dirs(media_folder, episodes)

    for label, dir_path in data_dirs:
        data_path = os.path.join(dir_path, "_data.json")

        with open(data_path) as f:
            data = json.load(f)

        segments = data.get("segments", [])
        to_tag = []
        for i, seg in enumerate(segments):
            if seg.get("content_analysis") is not None and not force:
                continue
            screenshot = seg.get("files", {}).get("screenshot")
            if screenshot:
                img_path = os.path.join(dir_path, screenshot)
                if os.path.exists(img_path):
                    to_tag.append((i, img_path))

        if not to_tag:
            console.print(f"  {label}: Already tagged ({len(segments)} segments)")
            continue

        console.print(f"  {label}: Tagging {len(to_tag)}/{len(segments)} screenshots...")

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

            # Persist after each batch so an interrupted run can resume.
            atomic_write_json(data_path, data)
            console.print(
                f"  {label}: saved {min(chunk_start + batch_size, len(to_tag))}/{len(to_tag)}"
            )

        console.print(f"  {label}: Done — tagged {len(to_tag)} segments")

    console.print("[green bold]Batch tagger complete[/green bold]")


def _run_modal(media_folder: str, episodes: list[int] | None, args) -> bool:
    """Tag via Modal Cloud GPU. Useful in cases where there is no local GPU available"""
    targets = discover_data_dirs(media_folder, episodes)
    if not targets:
        console.print(f"[yellow]No folders with _data.json under {media_folder}[/yellow]")
        return True

    def _local() -> None:
        console.print("[cyan]Falling back to local tagging...[/cyan]")
        batch_tagger(
            media_folder,
            episodes,
            batch_size=args.batch_size,
            force=args.force,
        )

    try:
        from nadeshiko_dev_tools.nsfw_tagger import modal_runner
    except ImportError as e:
        console.print(
            f"[red]Modal backend unavailable: {e}[/red]\n"
            "[yellow]Install it with: uv sync --extra modal[/yellow]"
        )
        if args.fallback_local:
            _local()
            return True
        return False

    try:
        modal_runner.run_modal(targets, batch_size=args.batch_size or 32, force=args.force)
        return True
    except Exception as e:
        console.print(f"[red]Modal run failed: {e}[/red]")
        if args.fallback_local:
            _local()
            return True
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch NSFW tagger for processed segments")
    parser.add_argument("media_folder", help="Path to media folder (e.g. output/21804)")
    parser.add_argument("--episodes", default=None, help="Comma-separated episode numbers")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Images per ONNX inference batch. Defaults to 4 on CPU, 32 on GPU.",
    )
    parser.add_argument(
        "--modal",
        action="store_true",
        help="Offload tagging to a Modal GPU (requires the 'modal' extra and a Modal token).",
    )
    parser.add_argument(
        "--fallback-local",
        action="store_true",
        help="Fall back to local tagging if the Modal run fails.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-tag segments that already have a content rating.",
    )
    args = parser.parse_args()

    media_folder = os.path.abspath(args.media_folder)
    ep_filter = None
    if args.episodes:
        ep_filter = [int(e.strip()) for e in args.episodes.split(",")]

    if args.modal:
        if not _run_modal(media_folder, ep_filter, args):
            return 1
    else:
        batch_tagger(
            media_folder,
            ep_filter,
            batch_size=args.batch_size,
            force=args.force,
        )

    # QC: validate tagger output
    from nadeshiko_dev_tools.common.quality_check import run_qc

    ep_set = set(ep_filter) if ep_filter else None
    report = run_qc(media_folder, episodes=ep_set, checks={"tagger"})
    passed = report.summary()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
