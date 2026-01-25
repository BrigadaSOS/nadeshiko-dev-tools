"""CLI entry point for uploading segments to Nadeshiko."""

import sys

from rich.console import Console

from media_sub_splitter.utils.uploader import upload_all

console = Console()


def main() -> None:
    """Main CLI entry point for uploading."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload media segments to Nadeshiko API (and optionally R2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update backend only (default - no R2 upload)
  %(prog)s /path/to/output

  # Upload files to R2 as well
  %(prog)s /path/to/output --upload-r2

  # Upload to production (requires confirmation)
  %(prog)s /path/to/output --prod

  # Upload specific media by ID
  %(prog)s /path/to/output --media-id 12345

  # Upload specific episode
  %(prog)s /path/to/output --media-id 12345 --episode 1

  # Dry run (no actual uploads)
  %(prog)s /path/to/output --dry-run

  # Dry run against production
  %(prog)s /path/to/output --prod --dry-run
        """,
    )

    parser.add_argument(
        "output",
        help="Path to media-sub-splitter output directory",
    )
    parser.add_argument(
        "--media-id",
        metavar="ID",
        help="Only upload specific media by folder name (AniList ID)",
    )
    parser.add_argument(
        "--episode",
        metavar="N",
        type=int,
        help="Only upload specific episode number",
    )
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Upload to production (default is local)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    parser.add_argument(
        "--upload-r2",
        action="store_true",
        help="Upload files to R2 (default is to only update the backend)",
    )

    args = parser.parse_args()

    try:
        upload_all(
            output_path=args.output,
            media_id=args.media_id,
            episode=args.episode,
            env="prod" if args.prod else "local",
            dry_run=args.dry_run,
            upload_r2=args.upload_r2,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Upload cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
