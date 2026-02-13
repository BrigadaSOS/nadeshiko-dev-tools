"""CLI entry point for F0 pitch contour extraction."""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="F0 pitch contour extraction for Nadeshiko audio segments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract pitch contours for a media folder
  %(prog)s extract /mnt/storage/nade-processed/132126

  # Extract without vocal separation (faster, less accurate)
  %(prog)s extract /mnt/storage/nade-processed/132126 --no-separation

  # Save isolated vocals for debugging
  %(prog)s extract /mnt/storage/nade-processed/132126 --save-vocals

  # Show extraction coverage statistics
  %(prog)s stats /mnt/storage/nade-processed/132126
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- extract subcommand --
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract F0 pitch contours from audio segments",
    )
    extract_parser.add_argument(
        "media_folder",
        type=Path,
        help="Path to media folder (must contain _info.json)",
    )
    extract_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess episodes that already have _pitch.json",
    )
    extract_parser.add_argument(
        "--no-separation",
        action="store_true",
        help="Skip Demucs vocal separation (faster, but BGM may affect results)",
    )
    extract_parser.add_argument(
        "--save-vocals",
        action="store_true",
        help="Save isolated vocals WAV files for debugging",
    )
    extract_parser.add_argument(
        "--sample-ms",
        type=int,
        default=10,
        help="Time step between F0 samples in milliseconds (default: 10)",
    )
    extract_parser.add_argument(
        "--pitch-floor",
        type=float,
        default=75.0,
        help="Minimum pitch frequency in Hz (default: 75)",
    )
    extract_parser.add_argument(
        "--pitch-ceiling",
        type=float,
        default=600.0,
        help="Maximum pitch frequency in Hz (default: 600)",
    )

    # -- stats subcommand --
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show extraction coverage and summary statistics",
    )
    stats_parser.add_argument(
        "media_folder",
        type=Path,
        help="Path to media folder (must contain _info.json)",
    )

    args = parser.parse_args()

    # Validate media folder
    media_folder = args.media_folder.resolve()
    if not (media_folder / "_info.json").exists():
        console.print(
            f"[red]Not a valid media folder (missing _info.json):"
            f" {media_folder}[/red]"
        )
        sys.exit(1)

    if args.command == "extract":
        from .batch import run_batch

        run_batch(
            media_folder=media_folder,
            resume=not args.no_resume,
            separation=not args.no_separation,
            save_vocals=args.save_vocals,
            sample_ms=args.sample_ms,
            pitch_floor=args.pitch_floor,
            pitch_ceiling=args.pitch_ceiling,
        )

    elif args.command == "stats":
        _stats(media_folder)


def _stats(media_folder: Path) -> None:
    """Show extraction coverage statistics."""
    from nadeshiko_dev_tools.common.archive import discover_episodes, discover_files

    from .batch import PITCH_FILE

    episodes = discover_episodes(media_folder)

    if not episodes:
        console.print("[yellow]No episode folders found.[/yellow]")
        return

    total_episodes = 0
    done_episodes = 0
    total_segments = 0
    done_segments = 0
    f0_value_counts: list[int] = []

    for ep_dir in episodes:
        segs = discover_files(ep_dir, "*.mp3")
        if not segs:
            continue

        total_episodes += 1
        total_segments += len(segs)

        pitch_file = ep_dir / PITCH_FILE
        if pitch_file.exists():
            done_episodes += 1
            with open(pitch_file) as f:
                data = json.load(f)
            seg_data = data.get("segments", {})
            done_segments += len(seg_data)
            for seg in seg_data.values():
                f0_value_counts.append(len(seg.get("f0", [])))

    console.print()
    console.print("[bold]Pitch Extraction Coverage[/bold]")
    ep_pct = (done_episodes / total_episodes * 100) if total_episodes else 0
    seg_pct = (done_segments / total_segments * 100) if total_segments else 0
    console.print(f"  Media: {media_folder.name}")
    console.print(
        f"  Episodes: {done_episodes}/{total_episodes} ({ep_pct:.1f}%)"
    )
    console.print(
        f"  Segments: {done_segments:,}/{total_segments:,} ({seg_pct:.1f}%)"
    )

    if f0_value_counts:
        avg_frames = sum(f0_value_counts) / len(f0_value_counts)
        console.print(f"  Avg F0 frames per segment: {avg_frames:.0f}")


if __name__ == "__main__":
    main()
