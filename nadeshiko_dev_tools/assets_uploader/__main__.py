"""CLI entry point for uploading segments to Nadeshiko."""

import sys

from rich.console import Console

from nadeshiko_dev_tools.assets_uploader.uploader import upload_all

console = Console()


def main() -> None:
    """Main CLI entry point for uploading."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload media segments to Nadeshiko API with configurable storage target",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to local API with local storage
  %(prog)s /path/to/output/12345 --target local --storage local

  # Dry run to dev API with R2 storage references (no upload)
  %(prog)s /path/to/output/12345 --target dev --storage r2

  # Upload to R2 and apply changes on prod API
  %(prog)s /path/to/output/12345 --target prod --storage r2 --apply --upload-r2

  # Upload specific episode
  %(prog)s /path/to/output/12345 --target dev --storage local --episode 1 --apply
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

    parser.add_argument(
        "--target",
        choices=["local", "dev", "prod"],
        default="local",
        help="API target environment",
    )
    parser.add_argument(
        "--storage",
        choices=["local", "r2"],
        default="local",
        help="Storage target for media file URLs",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform the upload (default is dry-run)",
    )
    parser.add_argument(
        "--upload-r2",
        "--upload-to-r2",
        dest="upload_r2",
        action="store_true",
        help="Actually upload files to R2 (requires --storage r2)",
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
            env=args.target,
            storage_target=args.storage,
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
