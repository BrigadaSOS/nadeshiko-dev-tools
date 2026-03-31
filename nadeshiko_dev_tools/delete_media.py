#!/usr/bin/env python3
"""Remove a media and all its assets from Nadeshiko API and R2 storage.

Usage:
    uv run delete-media 21804 --target dev
    uv run delete-media 21804 --target dev --dry-run
    uv run delete-media 21804 --target prod -y
    uv run delete-media 21804 --target dev --r2-only
"""

import argparse
import os
import sys

import boto3
from botocore.config import Config as BotoConfig
from dotenv import load_dotenv

load_dotenv()

from nadeshiko_internal import Nadeshiko  # noqa: E402
from nadeshiko_internal.api.media import (  # noqa: E402
    delete_episode,
    delete_media,
    get_episode,
    list_media,
)
from nadeshiko_internal.models import (  # noqa: E402
    Error400,
    Error401,
    Error403,
    Error404,
    Error409,
    Error429,
    Error500,
)

API_ERROR_TYPES = (Error400, Error401, Error403, Error404, Error409, Error429, Error500)


# --- API ---


def _first_set(*env_vars, default=""):
    for var in env_vars:
        val = os.getenv(var)
        if val:
            return val
    return default


def get_api_client(target: str) -> Nadeshiko:
    """Create Nadeshiko API client for the given target environment."""
    env_config = {
        "local": {
            "api_key": _first_set("NADESHIKO_LOCAL_API_KEY", "NADESHIKO_API_KEY"),
            "base_url": _first_set(
                "NADESHIKO_LOCAL_BASE_URL", "NADESHIKO_BASE_URL", default="http://localhost:5000"
            ),
        },
        "dev": {
            "api_key": _first_set("NADESHIKO_DEV_API_KEY", "NADESHIKO_API_KEY"),
            "base_url": _first_set("NADESHIKO_DEV_BASE_URL", default="https://api-dev.nadeshiko.co"),
        },
        "prod": {
            "api_key": os.getenv("NADESHIKO_PROD_API_KEY", ""),
            "base_url": os.getenv("NADESHIKO_PROD_BASE_URL", "https://api.nadeshiko.co"),
        },
    }

    if target not in env_config:
        print(f"Error: Unknown target '{target}'. Use: local, dev, prod", file=sys.stderr)
        sys.exit(1)

    cfg = env_config[target]
    if not cfg["api_key"]:
        print(f"Error: No API key configured for target '{target}'", file=sys.stderr)
        sys.exit(1)

    return Nadeshiko(token=cfg["api_key"], base_url=cfg["base_url"])


def find_media(client: Nadeshiko, media_id: str):
    """Find media by external ID (AniList ID) in the API."""
    from nadeshiko_internal.types import UNSET

    result = list_media.sync(client=client)
    if isinstance(result, API_ERROR_TYPES):
        print(f"  Error listing media: {result}", file=sys.stderr)
        return None

    if not result:
        return None

    media_items = getattr(result, "media", []) or []
    for media in media_items:
        ext_ids = getattr(media, "external_ids", UNSET)
        if ext_ids is UNSET or ext_ids is None:
            continue
        anilist_val = getattr(ext_ids, "anilist", None)
        if str(anilist_val) == str(media_id):
            return media

    return None


def remove_from_api(client: Nadeshiko, media_id: str, dry_run: bool) -> bool:
    """Remove media and all episodes from the Nadeshiko API."""
    print("\n--- Nadeshiko API cleanup ---")

    media = find_media(client, media_id)
    if not media:
        print(f"  Media {media_id} not found in API (may already be deleted)")
        return True

    internal_id = getattr(media, "public_id", None) or getattr(media, "id", None)
    title = getattr(media, "title", media_id)
    print(f"  Found: {title} (ID: {internal_id})")

    episodes_to_delete = []
    for ep_num in range(1, 100):
        result = get_episode.sync(media_id=internal_id, episode_number=ep_num, client=client)
        if isinstance(result, Error404):
            break
        if isinstance(result, API_ERROR_TYPES):
            continue
        if result:
            seg_count = getattr(result, "segment_count", "?")
            episodes_to_delete.append(ep_num)
            print(f"  Episode {ep_num}: {seg_count} segments")

    if not episodes_to_delete:
        print("  No episodes found")

    if dry_run:
        print(f"\n  [dry-run] Would delete {len(episodes_to_delete)} episodes + media record")
        return True

    for ep_num in episodes_to_delete:
        result = delete_episode.sync(media_id=internal_id, episode_number=ep_num, client=client)
        if isinstance(result, API_ERROR_TYPES):
            print(f"  Error deleting episode {ep_num}: {result}")
        else:
            print(f"  Deleted episode {ep_num}")

    result = delete_media.sync(id=internal_id, client=client)
    if isinstance(result, API_ERROR_TYPES):
        print(f"  Error deleting media: {result}")
        return False

    print(f"  Deleted media record {internal_id}")
    return True


# --- R2 ---


def remove_from_r2(storage_path: str, dry_run: bool, skip_confirm: bool = False) -> int:
    """Remove all files under a storage path from R2."""
    print("\n--- R2 storage cleanup ---")

    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
    secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket = os.environ.get("R2_BUCKET", "nadeshiko-production")

    missing = [
        v for v in ["R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"]
        if not os.environ.get(v)
    ]
    if missing:
        print(f"Error: Missing R2 credentials: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="auto",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=BotoConfig(max_pool_connections=50),
    )

    if not storage_path.startswith("media/"):
        storage_path = f"media/{storage_path}"
    prefix = f"{storage_path}/"

    print(f"Listing objects in {bucket}/{prefix}...")
    paginator = s3.get_paginator("list_objects_v2")

    all_objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        all_objects.extend(page.get("Contents", []))

    if not all_objects:
        print(f"No objects found with prefix '{prefix}'")
        return 0

    print(f"Found {len(all_objects)} objects")
    for obj in sorted(all_objects, key=lambda o: o["Key"]):
        print(f"  - {obj['Key']} ({obj.get('Size', 0)} bytes)")

    if dry_run:
        print(f"\n[dry-run] Would delete {len(all_objects)} objects")
        return len(all_objects)

    if not skip_confirm:
        response = input(f"\nDelete {len(all_objects)} objects? [y/N] ")
        if response.lower() != "y":
            print("Aborted")
            return 0

    total_deleted = 0
    batch_size = 1000
    for i in range(0, len(all_objects), batch_size):
        batch = all_objects[i : i + batch_size]
        delete_keys = [{"Key": obj["Key"]} for obj in batch]
        s3.delete_objects(Bucket=bucket, Delete={"Objects": delete_keys})
        total_deleted += len(delete_keys)
        print(f"Deleted {total_deleted}/{len(all_objects)} objects...")

    print(f"\nDeleted {total_deleted} objects from {bucket}/{prefix}")
    return total_deleted


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(
        description="Remove media from Nadeshiko API and R2 storage"
    )
    parser.add_argument("media_id", help="Media/AniList ID to remove (e.g., '20812')")
    parser.add_argument(
        "--target", required=True, choices=["local", "dev", "prod"], help="Target environment"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--api-only", action="store_true", help="Only delete from API, skip R2")
    parser.add_argument("--r2-only", action="store_true", help="Only delete from R2, skip API")
    args = parser.parse_args()

    if args.target == "prod" and not args.dry_run and not args.yes:
        response = input("You are about to delete from PRODUCTION. Are you sure? [y/N] ")
        if response.lower() != "y":
            print("Aborted")
            return 1

    print(f"Removing media {args.media_id} from {args.target}")
    if args.dry_run:
        print("[DRY RUN MODE]")

    if not args.r2_only:
        client = get_api_client(args.target)
        api_ok = remove_from_api(client, args.media_id, args.dry_run)
        if not api_ok:
            print("\nAPI cleanup failed. R2 cleanup skipped.")
            return 1

    if not args.api_only:
        remove_from_r2(args.media_id, args.dry_run, skip_confirm=args.yes or args.dry_run)

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
