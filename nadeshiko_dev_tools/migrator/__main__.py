"""CLI entry point for migrating old format (v5) to new format (v6)."""

import argparse
import sys

from rich.console import Console

from nadeshiko_dev_tools.migrator.migrator import migrate_anime

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate anime data from old format (v5) to new format (v6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate a single anime directory
  %(prog)s /path/to/old/bungou-stray-dogs /path/to/output

  # Migrate specific episodes only
  %(prog)s /path/to/old/bungou-stray-dogs /path/to/output --episodes 1,3,5

  # Dry run (no file processing, just show what would be done)
  %(prog)s /path/to/old/bungou-stray-dogs /path/to/output --dry-run

  # Skip video generation (faster, audio + screenshots only)
  %(prog)s /path/to/old/bungou-stray-dogs /path/to/output --skip-video
        """,
    )

    parser.add_argument(
        "input_dir",
        help="Path to old format anime directory (e.g., /path/to/bungou-stray-dogs)",
    )
    parser.add_argument(
        "output_dir",
        help="Path to output directory where v6 folders will be created",
    )
    parser.add_argument(
        "--episodes",
        metavar="1,3,5",
        help="Comma-separated list of episode numbers to migrate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without processing files",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Skip video generation (audio + screenshots only)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel workers per episode (default: 4)",
    )
    parser.add_argument(
        "--auto-id",
        action="store_true",
        help="Auto-accept default AniList IDs (sequel detection) without prompting",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt and start migration immediately",
    )

    args = parser.parse_args()

    episodes_filter = None
    if args.episodes:
        try:
            episodes_filter = {int(e.strip()) for e in args.episodes.split(",")}
        except ValueError:
            console.print(
                "[red]Invalid episodes format. Use comma-separated numbers like '1,3,5'[/red]"
            )
            sys.exit(1)

    config = {
        "episodes": episodes_filter,
        "dry_run": args.dry_run,
        "skip_video": args.skip_video,
        "workers": args.workers,
        "auto_id": args.auto_id,
        "yes": args.yes,
    }

    try:
        migrate_anime(args.input_dir, args.output_dir, config)
    except KeyboardInterrupt:
        console.print("\n[yellow]Migration cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
