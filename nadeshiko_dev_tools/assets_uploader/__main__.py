"""CLI entry point for uploading segments to Nadeshiko."""

import sys

from rich.console import Console

from nadeshiko_dev_tools.assets_uploader.uploader import upload_all

console = Console()


def main() -> None:
    """Main CLI entry point for uploading."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload media segments to Nadeshiko API (and optionally R2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run against dev (default behavior)
  %(prog)s /path/to/output/12345 --dev

  # Actually upload to dev
  %(prog)s /path/to/output/12345 --dev --apply

  # Dry run against production
  %(prog)s /path/to/output/12345 --prod

  # Upload to production with R2 files
  %(prog)s /path/to/output/12345 --prod --apply --upload-r2

  # Upload specific episode
  %(prog)s /path/to/output/12345 --dev --episode 1 --apply
        """,
    )

    parser.add_argument(
        "media_folder",
        help="Path to media ID folder (e.g., /path/to/output/12345)",
    )
    parser.add_argument(
        "--episode",
        metavar="N",
        type=int,
        help="Only upload specific episode number",
    )

    env_group = parser.add_mutually_exclusive_group(required=True)
    env_group.add_argument(
        "--dev",
        action="store_true",
        help="Target local/dev environment",
    )
    env_group.add_argument(
        "--prod",
        action="store_true",
        help="Target production environment",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform the upload (default is dry-run)",
    )
    parser.add_argument(
        "--upload-r2",
        action="store_true",
        help="Upload files to R2 (default is to only update the backend)",
    )
    parser.add_argument(
        "--update-info",
        action="store_true",
        help="Only update media/character/list info (skip episodes and segments)",
    )

    args = parser.parse_args()

    try:
        upload_all(
            output_path=args.media_folder,
            media_id=None,
            episode=args.episode,
            env="prod" if args.prod else "local",
            dry_run=not args.apply,
            upload_r2=args.upload_r2,
            update_info_only=args.update_info,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Upload cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
