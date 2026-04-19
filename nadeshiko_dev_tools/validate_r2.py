#!/usr/bin/env python3
"""Validate that all segments in the Nadeshiko API have matching R2 assets.

For each media, compares the segment count per episode (from API) against
R2 objects. Reports:
  - Episodes where R2 has fewer segments than the API expects
  - Segments with incomplete file sets (missing mp3, mp4, or webp)
  - Orphaned R2 files not matching any expected episode

Usage:
    uv run validate-r2 --target prod
    uv run validate-r2 --target prod --media 21804
    uv run validate-r2 --target prod --dry-run
"""

import argparse
import os
import sys
from collections import defaultdict

import boto3
from botocore.config import Config as BotoConfig
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

from nadeshiko_internal import Nadeshiko, NadeshikoError  # noqa: E402

EXPECTED_EXTENSIONS = {"mp3", "mp4", "webp"}

console = Console()


def get_api_client(target: str) -> Nadeshiko:
    env_config = {
        "local": {
            "api_key": os.getenv("NADESHIKO_LOCAL_API_KEY") or os.getenv("NADESHIKO_API_KEY", ""),
            "base_url": os.getenv("NADESHIKO_LOCAL_BASE_URL")
            or os.getenv("NADESHIKO_BASE_URL", "http://localhost:5000"),
        },
        "dev": {
            "api_key": os.getenv("NADESHIKO_DEV_API_KEY") or os.getenv("NADESHIKO_API_KEY", ""),
            "base_url": os.getenv("NADESHIKO_DEV_BASE_URL", "https://api-dev.nadeshiko.co"),
        },
        "prod": {
            "api_key": os.getenv("NADESHIKO_PROD_API_KEY", ""),
            "base_url": os.getenv("NADESHIKO_PROD_BASE_URL", "https://api.nadeshiko.co"),
        },
    }
    cfg = env_config.get(target)
    if not cfg or not cfg["api_key"]:
        console.print(f"[red]Error: No API key for target '{target}'[/red]")
        sys.exit(1)
    return Nadeshiko(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        headers={"User-Agent": "NadeshikoDevTools/1.0"},
    )


def get_r2_client():
    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
    secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")

    missing = [
        v
        for v in ["R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"]
        if not os.environ.get(v)
    ]
    if missing:
        console.print(f"[red]Error: Missing R2 credentials: {', '.join(missing)}[/red]")
        sys.exit(1)

    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        region_name="auto",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=BotoConfig(max_pool_connections=50),
    )


def list_r2_objects(s3, bucket: str, prefix: str) -> list[dict]:
    paginator = s3.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects.extend(page.get("Contents", []))
    return objects


def parse_r2_objects(objects: list[dict], base_prefix: str) -> dict[int, dict[str, set[str]]]:
    """Parse R2 objects into {episode: {hash: {extensions}}}."""
    episodes: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for obj in objects:
        key = obj["Key"]
        rel = key[len(base_prefix) :]
        parts = rel.strip("/").split("/")
        if len(parts) != 2:
            continue
        try:
            ep_num = int(parts[0])
        except ValueError:
            continue
        filename = parts[1]
        if "." not in filename:
            continue
        hashed_id, ext = filename.rsplit(".", 1)
        episodes[ep_num][hashed_id].add(ext)
    return dict(episodes)


def get_api_episodes(client: Nadeshiko, media_public_id: str) -> dict[int, int]:
    """Get {episode_number: segment_count} from the API."""
    episodes = {}
    try:
        for ep in client.iter_list_episodes(media_public_id=media_public_id):
            ep_num = getattr(ep, "episode_number", None)
            seg_count = getattr(ep, "segment_count", 0) or 0
            if ep_num is not None:
                episodes[ep_num] = seg_count
    except NadeshikoError:
        pass
    return episodes


def find_media_info(client: Nadeshiko, anilist_id: str | None) -> list[dict]:
    """Get all media or a specific one by AniList ID."""
    from nadeshiko_internal.types import UNSET

    found = []
    try:
        for media in client.iter_list_media():
            ext_ids = getattr(media, "external_ids", UNSET)
            al_id = None
            if ext_ids is not UNSET and ext_ids is not None:
                al_id = str(getattr(ext_ids, "anilist", ""))

            if anilist_id and al_id != str(anilist_id):
                continue

            found.append(
                {
                    "public_id": getattr(media, "public_id", None),
                    "title": getattr(media, "title", "?"),
                    "storage_base_path": getattr(media, "storage_base_path", None),
                    "anilist_id": al_id,
                }
            )
    except NadeshikoError as e:
        console.print(f"[red]Error listing media: {e.detail}[/red]")

    return found


def validate_media(client: Nadeshiko, s3, bucket: str, media: dict) -> dict:
    """Validate a single media. Returns summary dict."""
    storage_path = media["storage_base_path"] or f"media/{media['anilist_id']}"
    prefix = f"{storage_path}/"

    # Get R2 objects
    r2_objects = list_r2_objects(s3, bucket, prefix)
    r2_episodes = parse_r2_objects(r2_objects, prefix)

    # Get API episode counts
    api_episodes = get_api_episodes(client, media["public_id"])

    issues = []
    total_missing_files = 0
    total_incomplete_segments = 0
    total_orphan_r2 = 0

    all_episodes = sorted(set(list(api_episodes.keys()) + list(r2_episodes.keys())))

    for ep_num in all_episodes:
        api_count = api_episodes.get(ep_num, 0)
        r2_hashes = r2_episodes.get(ep_num, {})
        r2_count = len(r2_hashes)

        # Check for incomplete segments (missing mp3/mp4/webp)
        incomplete = []
        for hashed_id, exts in r2_hashes.items():
            missing_exts = EXPECTED_EXTENSIONS - exts
            if missing_exts:
                incomplete.append((hashed_id, missing_exts))
                total_incomplete_segments += 1
                total_missing_files += len(missing_exts)

        # Count mismatch
        if api_count != r2_count:
            diff = api_count - r2_count
            if diff > 0:
                issues.append(
                    f"  Episode {ep_num}: API has {api_count} segments, R2 has {r2_count} "
                    f"([red]{diff} segments missing from R2[/red])"
                )
            else:
                issues.append(
                    f"  Episode {ep_num}: API has {api_count} segments, R2 has {r2_count} "
                    f"([yellow]{-diff} orphaned in R2[/yellow])"
                )
                total_orphan_r2 += -diff

        if incomplete:
            for hashed_id, missing_exts in incomplete:
                issues.append(
                    f"  Episode {ep_num}: segment {hashed_id} missing "
                    f"[red]{', '.join(sorted(missing_exts))}[/red]"
                )

        # Episode exists in API but has no R2 data at all
        if ep_num in api_episodes and ep_num not in r2_episodes and api_count > 0:
            issues.append(
                f"  Episode {ep_num}: [red]{api_count} segments in API but NO R2 data[/red]"
            )

    return {
        "media": media,
        "api_episodes": api_episodes,
        "r2_episodes": r2_episodes,
        "issues": issues,
        "total_missing_files": total_missing_files,
        "total_incomplete_segments": total_incomplete_segments,
        "total_orphan_r2": total_orphan_r2,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate R2 assets against Nadeshiko API")
    parser.add_argument("--target", required=True, choices=["local", "dev", "prod"])
    parser.add_argument("--media", default=None, help="Specific AniList ID to validate")
    parser.add_argument("--bucket", default="nadeshiko-production", help="R2 bucket name")
    args = parser.parse_args()

    client = get_api_client(args.target)
    s3 = get_r2_client()

    console.print(f"\n[bold]Validating R2 assets against {args.target}...[/bold]\n")

    media_list = find_media_info(client, args.media)
    if not media_list:
        console.print("[yellow]No media found[/yellow]")
        return 0

    console.print(f"Found {len(media_list)} media to validate\n")

    all_ok = True
    summary_rows = []

    for media in media_list:
        title = media["title"]
        anilist_id = media["anilist_id"]
        console.print(f"[bold]Checking:[/bold] {title} (AniList: {anilist_id})...")

        result = validate_media(client, s3, args.bucket, media)

        api_seg_total = sum(result["api_episodes"].values())
        r2_seg_total = sum(len(h) for h in result["r2_episodes"].values())

        if result["issues"]:
            all_ok = False
            console.print("  [red]ISSUES FOUND[/red]")
            for issue in result["issues"]:
                console.print(issue)
        else:
            console.print(f"  [green]OK[/green] ({api_seg_total} segments, all files present)")

        summary_rows.append(
            {
                "title": title,
                "anilist_id": anilist_id,
                "api_segments": api_seg_total,
                "r2_segments": r2_seg_total,
                "missing_files": result["total_missing_files"],
                "incomplete": result["total_incomplete_segments"],
                "orphan_r2": result["total_orphan_r2"],
                "status": "OK" if not result["issues"] else "ISSUES",
            }
        )
        console.print()

    # Summary table
    console.print("\n[bold]Summary[/bold]")
    table = Table()
    table.add_column("Media", style="bold")
    table.add_column("AniList", justify="right")
    table.add_column("API Segs", justify="right")
    table.add_column("R2 Segs", justify="right")
    table.add_column("Missing Files", justify="right")
    table.add_column("Incomplete", justify="right")
    table.add_column("R2 Orphans", justify="right")
    table.add_column("Status")

    for row in summary_rows:
        status_style = "green" if row["status"] == "OK" else "red"
        table.add_row(
            row["title"],
            str(row["anilist_id"]),
            str(row["api_segments"]),
            str(row["r2_segments"]),
            str(row["missing_files"]),
            str(row["incomplete"]),
            str(row["orphan_r2"]),
            f"[{status_style}]{row['status']}[/{status_style}]",
        )

    console.print(table)

    if all_ok:
        console.print("\n[green]All media validated successfully![/green]")
    else:
        console.print("\n[red]Some media have issues — see details above.[/red]")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
