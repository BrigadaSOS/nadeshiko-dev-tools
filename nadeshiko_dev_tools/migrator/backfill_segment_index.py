#!/usr/bin/env python3
"""Backfill segment_index with original v5 IDs for already-migrated data.

This script updates _data.json files that were migrated using sequential
segment_index (1, 2, 3...) to use the original v5 IDs instead.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from nadeshiko_dev_tools.common.timestamps import parse_timestamp_to_ms

console = Console()


def parse_data_tsv(path: str) -> list[dict]:
    """Parse a v5 data.tsv file into a list of segment dicts.

    Returns list of dicts with keys: id, start_ms, end_ms, content_ja
    """
    if not os.path.exists(path):
        return []

    segments = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            segments.append({
                "id": int(row["ID"]) if row.get("ID", "").strip() else 0,
                "start_ms": parse_timestamp_to_ms(row["START_TIME"]),
                "end_ms": parse_timestamp_to_ms(row["END_TIME"]),
                "content_ja": row.get("CONTENT", "").strip(),
            })
    return segments


def find_matching_v5_id(
    v5_segments: list[dict], start_ms: int, end_ms: int, content_ja: str
) -> int | None:
    """Find the matching v5 segment ID by timing and content.

    First tries to match by exact timing (start_ms and end_ms).
    If not found, tries to match by content only.
    """
    # First try: exact timing match
    for seg in v5_segments:
        if seg["start_ms"] == start_ms and seg["end_ms"] == end_ms:
            return seg["id"]

    # Second try: content match
    for seg in v5_segments:
        if seg["content_ja"] == content_ja:
            console.print(
                "[yellow]Warning: Matched by content only (timing differs)[/yellow]"
            )
            return seg["id"]

    return None


def backfill_episode_data_json(
    v5_data_tsv: str, v6_data_json: str, dry_run: bool = False
) -> bool:
    """Backfill segment_index in a single episode's _data.json.

    Returns True if changes were made, False otherwise.
    """
    # Parse v5 data.tsv
    v5_segments = parse_data_tsv(v5_data_tsv)
    if not v5_segments:
        console.print(f"[yellow]No v5 segments found in {v5_data_tsv}[/yellow]")
        return False

    # Read v6 _data.json
    if not os.path.exists(v6_data_json):
        console.print(f"[yellow]No v6 _data.json found at {v6_data_json}[/yellow]")
        return False

    with open(v6_data_json, encoding="utf-8") as f:
        v6_data = json.load(f)

    segments = v6_data.get("segments", [])
    ignored_segments = v6_data.get("ignored_segments", [])

    if not segments and not ignored_segments:
        console.print(f"[yellow]No segments found in {v6_data_json}[/yellow]")
        return False

    changes_made = False

    # Process regular segments
    for segment in segments:
        v5_id = find_matching_v5_id(
            v5_segments,
            segment["start_ms"],
            segment["end_ms"],
            segment["content_ja"],
        )
        if v5_id is not None and segment["segment_index"] != v5_id:
            old_index = segment["segment_index"]
            segment["segment_index"] = v5_id
            changes_made = True
            console.print(
                f"  [cyan]Segment[/cyan]: {old_index} -> {v5_id} "
                f"({segment['content_ja'][:30]}...)"
            )

    # Process ignored segments
    for segment in ignored_segments:
        v5_id = find_matching_v5_id(
            v5_segments,
            segment["start_ms"],
            segment["end_ms"],
            segment["content_ja"],
        )
        if v5_id is not None and segment["segment_index"] != v5_id:
            old_index = segment["segment_index"]
            segment["segment_index"] = v5_id
            changes_made = True
            console.print(
                f"  [cyan]Ignored segment[/cyan]: {old_index} -> {v5_id} "
                f"({segment.get('content_ja', '')[:30]}...)"
            )

    # Write back to file if changes were made
    if changes_made and not dry_run:
        with open(v6_data_json, "w", encoding="utf-8") as f:
            json.dump(v6_data, f, ensure_ascii=False, indent=2)
        console.print(f"[green]Updated {v6_data_json}[/green]")
    elif changes_made and dry_run:
        console.print(f"[yellow]Would update {v6_data_json} (dry run)[/yellow]")

    return changes_made


def backfill_season(
    v5_season_path: str, v6_season_path: str, dry_run: bool = False
) -> int:
    """Backfill all episodes in a season.

    Returns the number of episodes updated.
    """
    updated_count = 0

    # Find all episode directories in v6
    v6_episodes = [
        d for d in os.listdir(v6_season_path)
        if os.path.isdir(os.path.join(v6_season_path, d)) and d.isdigit()
    ]

    for episode_num in sorted(v6_episodes, key=int):
        v5_episode_path = os.path.join(v5_season_path, episode_num)
        v6_episode_path = os.path.join(v6_season_path, episode_num)

        v5_data_tsv = os.path.join(v5_episode_path, "data.tsv")
        v6_data_json = os.path.join(v6_episode_path, "_data.json")

        if os.path.exists(v5_data_tsv) and os.path.exists(v6_data_json):
            console.print(f"\n[bold]Episode {episode_num}[/bold]")
            if backfill_episode_data_json(v5_data_tsv, v6_data_json, dry_run):
                updated_count += 1
        else:
            if not os.path.exists(v5_data_tsv):
                console.print(
                    f"[yellow]Episode {episode_num}: No v5 data.tsv found[/yellow]"
                )
            if not os.path.exists(v6_data_json):
                console.print(
                    f"[yellow]Episode {episode_num}: No v6 _data.json found[/yellow]"
                )

    return updated_count


def main():
    parser = argparse.ArgumentParser(
        description="Backfill segment_index with original v5 IDs"
    )
    parser.add_argument(
        "v5_path",
        help="Path to v5 anime directory (containing season folders with episode/data.tsv)",
    )
    parser.add_argument(
        "v6_path",
        help="Path to v6 output directory (containing season folders with episode/_data.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without actually modifying files",
    )

    args = parser.parse_args()

    v5_path = Path(args.v5_path)
    v6_path = Path(args.v6_path)

    if not v5_path.exists():
        console.print(f"[red]Error: v5 path does not exist: {v5_path}[/red]")
        sys.exit(1)

    if not v6_path.exists():
        console.print(f"[red]Error: v6 path does not exist: {v6_path}[/red]")
        sys.exit(1)

    console.print(f"[bold]V5 path:[/bold] {v5_path}")
    console.print(f"[bold]V6 path:[/bold] {v6_path}")
    if args.dry_run:
        console.print("[yellow]DRY RUN MODE - No files will be modified[/yellow]")

    # Find all seasons
    v5_seasons = [
        d
        for d in os.listdir(v5_path)
        if os.path.isdir(os.path.join(v5_path, d)) and d.isdigit()
    ]

    if not v5_seasons:
        console.print("[yellow]No season directories found in v5 path[/yellow]")
        sys.exit(1)

    total_updated = 0

    with Progress() as progress:
        task = progress.add_task("[cyan]Backfilling seasons...", total=len(v5_seasons))

        for season_num in sorted(v5_seasons, key=int):
            v5_season_path = os.path.join(v5_path, season_num)
            v6_season_path = os.path.join(v6_path, season_num)

            if not os.path.exists(v6_season_path):
                console.print(
                    f"\n[yellow]Season {season_num}: No corresponding v6 directory found[/yellow]"
                )
                progress.update(task, advance=1)
                continue

            console.print(f"\n[bold]Season {season_num}[/bold]")
            updated = backfill_season(v5_season_path, v6_season_path, args.dry_run)
            total_updated += updated
            progress.update(task, advance=1)

    console.print(f"\n[bold green]Total episodes updated: {total_updated}[/bold green]")
    if args.dry_run:
        console.print("[yellow]Run without --dry-run to apply changes[/yellow]")


if __name__ == "__main__":
    main()
