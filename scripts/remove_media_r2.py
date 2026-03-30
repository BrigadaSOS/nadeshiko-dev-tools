#!/usr/bin/env python3
"""Remove a media folder from R2 storage.

Usage:
    python scripts/remove_media_r2.py MEDIA_ID [--dry-run]

Removes all files under media/{MEDIA_ID}/ from R2.
"""

import argparse
import os
import sys

import boto3
from botocore.config import Config as BotoConfig
from dotenv import load_dotenv

load_dotenv()


def get_r2_config() -> tuple[str, str, str, str]:
    """Get R2 credentials from environment variables.

    Returns:
        tuple of (endpoint, access_key_id, secret_access_key, bucket)
    """
    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
    secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket = os.environ.get("R2_BUCKET", "nadeshiko-production")

    missing = []
    if not account_id:
        missing.append("R2_ACCOUNT_ID")
    if not access_key_id:
        missing.append("R2_ACCESS_KEY_ID")
    if not secret_access_key:
        missing.append("R2_SECRET_ACCESS_KEY")

    if missing:
        print(f"Error: Missing R2 credentials: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    return endpoint, access_key_id, secret_access_key, bucket


def remove_media_folder(storage_path: str, dry_run: bool, skip_confirm: bool = False) -> int:
    """Remove all files under a storage path from R2.

    Args:
        storage_path: The storage path or media ID to remove.
            Accepts either a full path like "media/21459" or just an ID like "21459".
        dry_run: If True, print actions without executing

    Returns:
        Number of objects deleted (or that would be deleted)
    """
    endpoint, access_key_id, secret_access_key, bucket = get_r2_config()

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=BotoConfig(max_pool_connections=50),
    )

    if not storage_path.startswith("media/"):
        storage_path = f"media/{storage_path}"
    prefix = f"{storage_path}/"

    # List and count objects first
    print(f"Listing objects in {bucket}/{prefix}...")
    paginator = s3.get_paginator("list_objects_v2")

    all_objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = page.get("Contents", [])
        all_objects.extend(objects)

    if not all_objects:
        print(f"No objects found with prefix '{prefix}'")
        return 0

    print(f"Found {len(all_objects)} objects")
    for obj in sorted(all_objects, key=lambda o: o["Key"]):
        print(f"  - {obj['Key']} ({obj.get('Size', 0)} bytes)")

    if dry_run:
        print(f"\n[dry-run] Would delete {len(all_objects)} objects")
        return len(all_objects)

    # Confirm deletion
    if not skip_confirm:
        response = input(f"\nDelete {len(all_objects)} objects? [y/N] ")
        if response.lower() != "y":
            print("Aborted")
            return 0

    # Delete in batches (R2/S3 supports up to 1000 keys per delete_objects call)
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


def main():
    parser = argparse.ArgumentParser(description="Remove a media folder from R2 storage")
    parser.add_argument(
        "media_id",
        help="Media ID or storage path to remove (e.g., '7674' or 'media/7674')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List objects without deleting",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    args = parser.parse_args()

    remove_media_folder(args.media_id, args.dry_run, skip_confirm=args.yes)


if __name__ == "__main__":
    main()
