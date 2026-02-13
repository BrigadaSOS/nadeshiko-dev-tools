"""CLI entry point for content rating tagging."""

import argparse
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def _positive_int(value: str) -> int:
    """Argparse type for positive integers."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("Must be >= 1")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Content rating classification for Nadeshiko segment images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify all images for a media
  %(prog)s classify /mnt/storage/nade-processed/132126

  # Batch process all media folders
  %(prog)s batch /mnt/storage/nade-processed

  # Export SQL update file from results
  %(prog)s export /mnt/storage/nade-processed/132126

  # Review classification results
  %(prog)s review /mnt/storage/nade-processed/132126
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- classify subcommand --
    classify_parser = subparsers.add_parser(
        "classify",
        help="Batch classify images using WD Tagger v3",
    )
    classify_parser.add_argument(
        "media_folder",
        type=Path,
        help="Path to media folder (must contain _info.json)",
    )
    classify_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess even if results already exist",
    )
    classify_parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=None,
        help="Initial inference batch size (default: 16 or NSFW_TAGGER_BATCH_SIZE)",
    )

    # -- batch subcommand --
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch classify all media folders in a directory",
    )
    batch_parser.add_argument(
        "root_dir",
        type=Path,
        help="Path to directory containing media folders",
    )
    batch_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess even if results already exist",
    )
    batch_parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=None,
        help="Initial inference batch size (default: 16 or NSFW_TAGGER_BATCH_SIZE)",
    )

    # -- export subcommand --
    export_parser = subparsers.add_parser(
        "export",
        help="Export classification results as SQL update file",
    )
    export_parser.add_argument(
        "media_folder",
        type=Path,
        help="Path to media folder (must contain _info.json)",
    )

    # -- review subcommand --
    review_parser = subparsers.add_parser(
        "review",
        help="Review classification results and statistics",
    )
    review_parser.add_argument(
        "media_folder",
        type=Path,
        help="Path to media folder (must contain _info.json)",
    )

    args = parser.parse_args()

    # Handle batch command separately (different argument)
    if args.command == "batch":
        from .batch import run_batch_all

        run_batch_all(
            root_dir=args.root_dir.resolve(),
            resume=not args.no_resume,
            batch_size=args.batch_size,
        )
        return

    # For other commands, validate media folder
    media_folder = args.media_folder.resolve()
    if not (media_folder / "_info.json").exists():
        console.print(
            f"[red]Not a valid media folder (missing _info.json):"
            f" {media_folder}[/red]"
        )
        sys.exit(1)

    if args.command == "classify":
        from .batch import run_batch

        run_batch(
            media_folder=media_folder,
            resume=not args.no_resume,
            batch_size=args.batch_size,
        )

    elif args.command == "export":
        from .export import export_sql

        export_sql(media_folder=media_folder)

    elif args.command == "review":
        _review(media_folder)


def _review(media_folder: Path) -> None:
    """Review classification results."""
    import json

    from .batch import RESULTS_DIR_NAME, RESULTS_FILE

    results_dir = media_folder / RESULTS_DIR_NAME
    result_file = results_dir / RESULTS_FILE

    if not result_file.exists():
        console.print("[red]No results found. Run classify first.[/red]")
        return

    with open(result_file) as f:
        data = json.load(f)

    by_rating = {"SAFE": 0, "SUGGESTIVE": 0, "QUESTIONABLE": 0, "EXPLICIT": 0}
    flagged = []

    for ep_num, ep_data in sorted(data.items(), key=lambda x: int(x[0])):
        for hashed_id, result in ep_data.items():
            cr = result["content_rating"]
            by_rating[cr] = by_rating.get(cr, 0) + 1
            if cr != "SAFE":
                flagged.append((int(ep_num), hashed_id, result))

    total = sum(by_rating.values())
    console.print(f"[bold]Media {media_folder.name}[/bold]")
    console.print(f"  Total segments: {total}")
    console.print(f"  Content ratings: {by_rating}")
    console.print()

    if flagged:
        console.print("[bold]Flagged segments (non-safe):[/bold]")
        for ep, hash_id, result in sorted(flagged):
            tags = list(result.get("tags", {}).keys())[:5]
            tag_str = ", ".join(tags) if tags else ""
            console.print(
                f"  ep={ep} hash={hash_id} "
                f"rating={result['content_rating']}"
                f" tags=[{tag_str}]"
            )

    # Summary stats
    summary_file = results_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        if "elapsed_seconds" in summary:
            console.print(
                f"\n  [dim]Classified in {summary['elapsed_seconds']:.0f}s[/dim]"
            )


if __name__ == "__main__":
    main()
