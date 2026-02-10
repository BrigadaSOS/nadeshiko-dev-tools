"""Move a fully processed anime folder to the processed archive."""

import argparse
import shutil
import sys
from pathlib import Path

from rich.console import Console

console = Console()

DEFAULT_DST = Path("/mnt/storage/nade-processed")


def move_processed(folder: Path, dst: Path, dry_run: bool = True) -> None:
    """Move a single anime folder to the processed archive."""
    if not folder.exists():
        console.print(f"[red]Folder does not exist: {folder}[/red]")
        sys.exit(1)

    if not (folder / "_info.json").exists():
        console.print(f"[red]Not a valid media folder (missing _info.json): {folder}[/red]")
        sys.exit(1)

    dst.mkdir(parents=True, exist_ok=True)

    target = dst / folder.name
    if target.exists():
        console.print(f"[yellow]Already exists in destination: {target}[/yellow]")
        sys.exit(1)

    if dry_run:
        console.print(f"[cyan][DRY RUN] Would move: {folder} -> {target}[/cyan]")
    else:
        shutil.move(str(folder), str(target))
        console.print(f"[green]Moved: {folder} -> {target}[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Move a processed anime folder to archive")
    parser.add_argument(
        "media_folder",
        type=Path,
        help="Path to the media folder to move (e.g., /mnt/storage/nade-toprocess/12345)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_DST,
        help=f"Destination directory (default: {DEFAULT_DST})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move the folder (default is dry-run)",
    )

    args = parser.parse_args()
    move_processed(args.media_folder, args.dst, dry_run=not args.apply)


if __name__ == "__main__":
    main()
