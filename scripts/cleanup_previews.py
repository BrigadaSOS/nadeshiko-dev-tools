#!/usr/bin/env python3
"""Remove full-size screenshots and rename preview (*p.webp) files to {hash}.webp.

Usage:
    python scripts/cleanup_previews.py /path/to/media_folder [--dry-run]

Expects the media folder structure:
    media_folder/
        episode_1/
            abc1234567.webp    <- full-size (delete)
            abc1234567p.webp   <- preview   (rename to abc1234567.webp)
            ...
        episode_2/
            ...
"""

import argparse
import os
import sys


def process_episode_dir(episode_dir: str, dry_run: bool) -> tuple[int, int]:
    deleted = 0
    renamed = 0

    preview_files = [f for f in os.listdir(episode_dir) if f.endswith("p.webp")]

    for preview_name in sorted(preview_files):
        hash_id = preview_name[:-len("p.webp")]
        full_size_name = f"{hash_id}.webp"

        full_size_path = os.path.join(episode_dir, full_size_name)
        preview_path = os.path.join(episode_dir, preview_name)

        if os.path.exists(full_size_path):
            if dry_run:
                print(f"  [dry-run] delete {full_size_name}")
            else:
                os.remove(full_size_path)
            deleted += 1

        if dry_run:
            print(f"  [dry-run] rename {preview_name} -> {full_size_name}")
        else:
            os.rename(preview_path, full_size_path)
        renamed += 1

    return deleted, renamed


def main():
    parser = argparse.ArgumentParser(
        description="Remove full-size screenshots and rename *p.webp previews to {hash}.webp"
    )
    parser.add_argument("media_folder", help="Path to the media folder containing episode dirs")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    args = parser.parse_args()

    media_folder = args.media_folder
    if not os.path.isdir(media_folder):
        print(f"Error: {media_folder} is not a directory", file=sys.stderr)
        sys.exit(1)

    total_deleted = 0
    total_renamed = 0

    for entry in sorted(os.listdir(media_folder)):
        episode_dir = os.path.join(media_folder, entry)
        if not os.path.isdir(episode_dir):
            continue

        deleted, renamed = process_episode_dir(episode_dir, args.dry_run)
        if deleted or renamed:
            print(f"{entry}: deleted {deleted}, renamed {renamed}")
            total_deleted += deleted
            total_renamed += renamed

    print(f"\nTotal: deleted {total_deleted} full-size, renamed {total_renamed} previews")


if __name__ == "__main__":
    main()
