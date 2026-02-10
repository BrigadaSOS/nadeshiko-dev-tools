#!/usr/bin/env python3
"""Remove a media folder from R2 storage.

Usage:
    python scripts/remove_media_r2.py MEDIA_ID [--dry-run] [--env ENV]

Removes all files under media/{MEDIA_ID}/ from R2.
"""

import argparse
import os
import sys

import boto3
from botocore.config import Config as BotoConfig
from dotenv import load_dotenv

load_dotenv()

R2_ENDPOINT_TEMPLATE = "https://{account_id}.r2.cloudflarestorage.com"


def get_r2_config(env: str) -> tuple[str, str, str, str]:
    """Get R2 credentials for the specified environment.

    Returns:
        tuple of (endpoint, access_key_id, secret_access_key, bucket)
    """
    env = env.lower()

    if env == "prod":
        account_id = os.environ.get("R2_PROD_ACCOUNT_ID", os.environ.get("R2_ACCOUNT_ID"))
        access_key_id = os.environ.get("R2_PROD_ACCESS_KEY_ID", os.environ.get("R2_ACCESS_KEY_ID"))
        secret_access_key = os.environ.get("R2_PROD_SECRET_ACCESS_KEY", os.environ.get("R2_SECRET_ACCESS_KEY"))
        bucket = os.environ.get("R2_PROD_BUCKET", os.environ.get("R2_BUCKET", "nadeshiko-production"))
    elif env == "dev":
        account_id = os.environ.get("R2_DEV_ACCOUNT_ID", os.environ.get("R2_ACCOUNT_ID"))
        access_key_id = os.environ.get("R2_DEV_ACCESS_KEY_ID", os.environ.get("R2_ACCESS_KEY_ID"))
        secret_access_key = os.environ.get("R2_DEV_SECRET_ACCESS_KEY", os.environ.get("R2_SECRET_ACCESS_KEY"))
        bucket = os.environ.get("R2_DEV_BUCKET", os.environ.get("R2_BUCKET", "nadeshiko-dev"))
    else:  # local
        account_id = os.environ.get("R2_ACCOUNT_ID")
        access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
        secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")
        bucket = os.environ.get("R2_BUCKET", "nadeshiko-dev")

    missing = []
    if not account_id:
        missing.append("R2_ACCOUNT_ID (or R2_{ENV}_ACCOUNT_ID)")
    if not access_key_id:
        missing.append("R2_ACCESS_KEY_ID (or R2_{ENV}_ACCESS_KEY_ID)")
    if not secret_access_key:
        missing.append("R2_SECRET_ACCESS_KEY (or R2_{ENV}_SECRET_ACCESS_KEY)")

    if missing:
        print(f"Error: Missing R2 credentials: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    endpoint = R2_ENDPOINT_TEMPLATE.format(account_id=account_id)
    return endpoint, access_key_id, secret_access_key, bucket


def remove_media_folder(media_id: str, env: str, dry_run: bool) -> int:
    """Remove all files under media/{media_id}/ from R2.

    Args:
        media_id: The media ID to remove
        env: Environment to use (local, dev, prod)
        dry_run: If True, print actions without executing

    Returns:
        Number of objects deleted (or that would be deleted)
    """
    endpoint, access_key_id, secret_access_key, bucket = get_r2_config(env)

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=BotoConfig(max_pool_connections=50),
    )

    prefix = f"media/{media_id}/"

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
    parser = argparse.ArgumentParser(
        description="Remove a media folder from R2 storage"
    )
    parser.add_argument("media_id", help="Media ID to remove (e.g., '7674')")
    parser.add_argument(
        "--env",
        choices=["local", "dev", "prod"],
        default="prod",
        help="R2 environment to use (default: prod)",
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

    # Monkey-patch to skip confirmation if -y is set
    if args.yes:
        input = lambda x: "y"  # noqa: E731 (type: ignore)

    remove_media_folder(args.media_id, args.env, args.dry_run)


if __name__ == "__main__":
    main()
