#!/usr/bin/env python3
"""Generate SQL to backfill pos_analysis into the Segment table.

Reads _data.json files from disk and emits UPDATE statements
that set pos_analysis for each segment matched by hashed_id + media_id + episode.

Usage:
    uv run python scripts/backfill_pos_analysis.py [--storage-path PATH] [--batch-size N] [--dry-run]
"""

import argparse
import json
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

console = Console()

DEFAULT_STORAGE_PATH = "/mnt/storage/nade-processed"
DEFAULT_OUTPUT_PATH = "scripts/output/backfill_pos_analysis.sql"
DEFAULT_BATCH_SIZE = 1000


def escape_sql_string(s: str) -> str:
    """Escape a string for use in a SQL literal (single-quote doubling)."""
    return s.replace("'", "''")


def generate_sql(storage_path: str, output_path: str, batch_size: int, dry_run: bool) -> None:
    storage = Path(storage_path)
    if not storage.exists():
        console.print(f"[red]Error: storage path does not exist: {storage}[/red]")
        sys.exit(1)

    # Discover all media directories (contain episode subdirectories)
    media_dirs = sorted(
        [d for d in storage.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    if not media_dirs:
        console.print("[yellow]No media directories found[/yellow]")
        sys.exit(1)

    console.print(f"[bold]Storage path:[/bold] {storage}")
    console.print(f"[bold]Media directories:[/bold] {len(media_dirs)}")
    if dry_run:
        console.print("[yellow]DRY RUN - will count statements but not write file[/yellow]")

    total_updates = 0
    statements: list[str] = []

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing media dirs...", total=len(media_dirs))

        for media_dir in media_dirs:
            # Find episode subdirectories (numeric names)
            episode_dirs = sorted(
                [d for d in media_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                key=lambda p: int(p.name),
            )

            for episode_dir in episode_dirs:
                data_json_path = episode_dir / "_data.json"
                if not data_json_path.exists():
                    continue

                try:
                    with open(data_json_path, encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    console.print(
                        f"[yellow]Warning: Could not read {data_json_path}: {e}[/yellow]"
                    )
                    continue

                media_id = data.get("media", {}).get("anilist_id")
                episode_number = data.get("metadata", {}).get("number")

                if media_id is None or episode_number is None:
                    continue

                for segment in data.get("segments", []):
                    pos_analysis = segment.get("pos_analysis")
                    hashed_id = segment.get("segment_hash")

                    if not pos_analysis or not hashed_id:
                        continue

                    json_str = escape_sql_string(
                        json.dumps(pos_analysis, ensure_ascii=False)
                    )

                    stmt = (
                        f"UPDATE \"Segment\" SET pos_analysis = '{json_str}'::jsonb "
                        f"WHERE hashed_id = '{escape_sql_string(hashed_id)}' "
                        f"AND media_id = {media_id} AND episode = {episode_number};"
                    )
                    statements.append(stmt)
                    total_updates += 1

            progress.update(task, advance=1)

    console.print(f"\n[bold]Total UPDATE statements:[/bold] {total_updates}")

    if dry_run:
        console.print("[yellow]Dry run complete. No file written.[/yellow]")
        return

    if total_updates == 0:
        console.print("[yellow]No segments with pos_analysis found. Nothing to write.[/yellow]")
        return

    # Write SQL file with batched transactions
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        f.write("-- Backfill pos_analysis for existing segments\n")
        f.write(f"-- Generated {total_updates} UPDATE statements\n\n")

        for i in range(0, len(statements), batch_size):
            batch = statements[i : i + batch_size]
            f.write(f"-- Batch {i // batch_size + 1}\n")
            f.write("BEGIN;\n")
            for stmt in batch:
                f.write(stmt + "\n")
            f.write("COMMIT;\n\n")

    console.print(f"[bold green]SQL written to {output}[/bold green]")
    console.print(
        f"[bold]Batches:[/bold] {(total_updates + batch_size - 1) // batch_size} "
        f"(size {batch_size})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate SQL to backfill pos_analysis into the Segment table"
    )
    parser.add_argument(
        "--storage-path",
        default=DEFAULT_STORAGE_PATH,
        help=f"Path to nade-processed storage (default: {DEFAULT_STORAGE_PATH})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output SQL file path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Statements per transaction batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count statements without writing the SQL file",
    )

    args = parser.parse_args()
    generate_sql(args.storage_path, args.output, args.batch_size, args.dry_run)


if __name__ == "__main__":
    main()
