"""CLI entry point for mora-aligned pitch analysis."""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mora-aligned F0 pitch with accent predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Align mora for a single media
  %(prog)s align /mnt/storage/nade-processed/132126

  # Show alignment coverage statistics
  %(prog)s stats /mnt/storage/nade-processed/132126

  # Generate visualization for one episode
  %(prog)s visualize /mnt/storage/nade-processed/132126 --episode 12

  # Generate visualization for all episodes
  %(prog)s visualize /mnt/storage/nade-processed/132126
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- align subcommand --
    align_parser = subparsers.add_parser(
        "align",
        help="Compute mora alignment with accent and F0 data",
    )
    align_parser.add_argument(
        "media_folder",
        type=Path,
        help="Path to media folder (must contain _info.json)",
    )
    align_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess episodes that already have _mora_pitch.json",
    )

    # -- stats subcommand --
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show mora alignment coverage and statistics",
    )
    stats_parser.add_argument(
        "media_folder",
        type=Path,
        help="Path to media folder (must contain _info.json)",
    )

    # -- visualize subcommand --
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Generate mora_viz.html visualization",
    )
    viz_parser.add_argument(
        "media_folder",
        type=Path,
        help="Path to media folder (must contain _info.json)",
    )
    viz_parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Process only this episode number",
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

    if args.command == "align":
        from .batch import run_batch

        run_batch(
            media_folder=media_folder,
            resume=not args.no_resume,
        )

    elif args.command == "stats":
        _stats(media_folder)

    elif args.command == "visualize":
        from .visualizer import run_visualize

        run_visualize(
            media_folder=media_folder,
            episode_num=args.episode,
        )


def _stats(media_folder: Path) -> None:
    """Show mora alignment coverage statistics."""
    from nadeshiko_dev_tools.common.archive import discover_episodes

    from .batch import DATA_FILE, MORA_PITCH_FILE

    episodes = discover_episodes(media_folder)

    if not episodes:
        console.print("[yellow]No episode folders found.[/yellow]")
        return

    total_episodes = 0
    done_episodes = 0
    total_segments = 0
    aligned_segments = 0
    total_words = 0
    total_mora = 0
    accent_counts = {"H": 0, "L": 0}

    for ep_dir in episodes:
        data_file = ep_dir / DATA_FILE
        mora_file = ep_dir / MORA_PITCH_FILE

        if not data_file.exists():
            continue

        with open(data_file) as f:
            data = json.load(f)
        seg_count = len(data.get("segments", []))
        if not seg_count:
            continue

        total_episodes += 1
        total_segments += seg_count

        if mora_file.exists():
            done_episodes += 1
            with open(mora_file) as f:
                mora_data = json.load(f)
            mora_segs = mora_data.get("segments", {})
            aligned_segments += len(mora_segs)

            for seg in mora_segs.values():
                words = seg.get("words", [])
                total_words += len(words)
                for word in words:
                    mora_list = word.get("mora", [])
                    total_mora += len(mora_list)
                    for m in mora_list:
                        accent = m.get("accent", "")
                        if accent in accent_counts:
                            accent_counts[accent] += 1

    console.print()
    console.print("[bold]Mora Alignment Coverage[/bold]")
    ep_pct = (done_episodes / total_episodes * 100) if total_episodes else 0
    seg_pct = (aligned_segments / total_segments * 100) if total_segments else 0
    console.print(f"  Media: {media_folder.name}")
    console.print(
        f"  Episodes: {done_episodes}/{total_episodes} ({ep_pct:.1f}%)"
    )
    console.print(
        f"  Segments: {aligned_segments:,}/{total_segments:,} ({seg_pct:.1f}%)"
    )
    console.print(f"  Words: {total_words:,}")
    console.print(f"  Mora: {total_mora:,}")

    if total_mora:
        h_pct = accent_counts["H"] / total_mora * 100
        l_pct = accent_counts["L"] / total_mora * 100
        console.print(
            f"  Accent: H={accent_counts['H']:,} ({h_pct:.1f}%),"
            f" L={accent_counts['L']:,} ({l_pct:.1f}%)"
        )


if __name__ == "__main__":
    main()
