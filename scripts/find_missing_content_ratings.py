#!/usr/bin/env python3
"""Find media folders with segments missing content_rating.

Checks both _nsfw_results/results.json and inline _data.json content_rating
to identify segments that would fail during upload.
"""

import json
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def check_media(media_dir: Path) -> dict:
    """Check a single media folder for missing content ratings.

    Returns a dict with episode -> list of missing segment hashes,
    or empty dict if everything is fine.
    """
    nsfw_results_path = media_dir / "_nsfw_results" / "results.json"

    # Build flat NSFW lookup
    nsfw_lookup: dict[str, dict] = {}
    if nsfw_results_path.exists():
        with open(nsfw_results_path) as f:
            data = json.load(f)
        for ep_data in data.values():
            for hashed_id, result in ep_data.items():
                nsfw_lookup[hashed_id] = result

    missing: dict[str, list[str]] = {}

    for ep_dir in sorted(media_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        data_path = ep_dir / "_data.json"
        if not data_path.exists():
            continue

        with open(data_path) as f:
            ep_data = json.load(f)

        for seg in ep_data.get("segments", []):
            seg_hash = seg.get("segment_hash", "")
            has_nsfw = seg_hash in nsfw_lookup and nsfw_lookup[seg_hash].get("content_rating")
            has_inline = bool(seg.get("content_rating"))

            if not has_nsfw and not has_inline:
                missing.setdefault(ep_dir.name, []).append(seg_hash)

    return missing


def main():
    if len(sys.argv) < 2:
        console.print("Usage: find_missing_content_ratings.py <root_dir>")
        console.print("  e.g. find_missing_content_ratings.py /mnt/storage/nade-processed")
        sys.exit(1)

    root_dir = Path(sys.argv[1]).resolve()
    if not root_dir.is_dir():
        console.print(f"[red]Not a directory: {root_dir}[/red]")
        sys.exit(1)

    media_dirs = sorted(
        d for d in root_dir.iterdir()
        if d.is_dir() and (d / "_info.json").exists()
    )

    console.print(f"Scanning {len(media_dirs)} media folders in {root_dir}\n")

    affected = []

    for media_dir in media_dirs:
        missing = check_media(media_dir)
        if missing:
            total_missing = sum(len(v) for v in missing.values())
            affected.append((media_dir.name, missing, total_missing))

    if not affected:
        console.print("[green]All media folders have complete content ratings.[/green]")
        return

    console.print(f"[bold red]Found {len(affected)} media folder(s) with missing content ratings:[/bold red]\n")

    for media_id, missing, total_missing in affected:
        ep_count = len(missing)
        console.print(f"[bold]{media_id}[/bold]: {total_missing} segments missing across {ep_count} episode(s)")
        for ep, hashes in sorted(missing.items(), key=lambda x: int(x[0])):
            console.print(f"  ep {ep}: {len(hashes)} missing")

    console.print(f"\n[bold]To fix, re-run the NSFW tagger on affected media:[/bold]")
    for media_id, _, _ in affected:
        console.print(f"  nsfw-tagger classify {root_dir / media_id} --no-resume")


if __name__ == "__main__":
    main()
