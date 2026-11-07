#!/usr/bin/env python3
"""Remove a media and all its assets from Nadeshiko API and R2 storage.

Staging (`dev`) and production share one R2 bucket and the same files: the dev upload puts them
there and the prod upload reuses them. So R2 is only ever cleaned as part of a **prod** delete;
`--target dev`/`local` remove the API record and leave the files alone. The API doesn't expose a
media's storage path, so the R2 prefix is derived from the external id that actually matched a
media in the API, with the uploader's rules (anilist -> media/<id>, tmdb -> media/mv/<id>,
youtube -> media/yt/<id>, asin -> media/ab/<id>). Nothing is deleted from R2 when no media matched.

Usage:
    uv run delete-media 21804 --target dev              # API only (files are shared with prod)
    uv run delete-media 21804 --target prod --dry-run   # preview API + R2
    uv run delete-media 21804 --target prod -y
    uv run delete-media --target prod --r2-only --r2-path media/21804   # orphaned files only
"""

import argparse
import os
import re
import sys

import boto3
from botocore.config import Config as BotoConfig
from dotenv import load_dotenv

load_dotenv()

from nadeshiko_internal import Nadeshiko, NadeshikoError  # noqa: E402

# --- API ---


class MediaLookupError(RuntimeError):
    """The API could not be asked; "not found" can't be concluded from it."""


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
            # "dev" is staging; api-dev.nadeshiko.co no longer exists.
            "base_url": _first_set(
                "NADESHIKO_DEV_BASE_URL", default="https://api-stg.nadeshiko.co"
            ),
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

    return Nadeshiko(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        headers={"User-Agent": "NadeshikoDevTools/1.0"},
    )


_EXTERNAL_ID_SOURCES = ("anilist", "tmdb", "tvdb", "imdb", "youtube", "asin")

# R2 folder per external-id source, mirroring the uploader's MediaSource.storage_base_path.
# tvdb/imdb have no upload source, so their folder is unknown and R2 is left alone.
_STORAGE_PREFIX_BY_SOURCE = {
    "anilist": "media/{id}",
    "tmdb": "media/mv/{id}",
    "youtube": "media/yt/{id}",
    "asin": "media/ab/{id}",
}


def storage_path_for(source: str, external_id: str) -> str | None:
    """R2 folder the uploader used for a media matched by this source, or None if unknown."""
    template = _STORAGE_PREFIX_BY_SOURCE.get(source)
    return template.format(id=external_id) if template else None


def find_media(client: Nadeshiko, media_id: str):
    """Find media by external ID (AniList/TMDB/YouTube channel/ASIN/etc.).

    Returns (media, source) for the first match, or (None, None) only when the full listing
    succeeded and nothing matched; an API error raises MediaLookupError instead, so a failed
    lookup is never mistaken for "already deleted".
    """
    from nadeshiko_internal.types import UNSET

    try:
        for media in client.iter_list_media():
            ext_ids = getattr(media, "external_ids", UNSET)
            if ext_ids is UNSET or ext_ids is None:
                continue
            for source in _EXTERNAL_ID_SOURCES:
                if str(getattr(ext_ids, source, None)) == str(media_id):
                    return media, source
    except NadeshikoError as e:
        raise MediaLookupError(f"Error listing media: {e.detail}") from e
    return None, None


def remove_from_api(client: Nadeshiko, media_id: str, dry_run: bool) -> tuple[bool, str | None]:
    """Remove media and all episodes from the Nadeshiko API.

    Returns (ok, storage_path); the path is None when the media isn't in the API or its R2
    folder can't be derived from the matched source.
    """
    print("\n--- Nadeshiko API cleanup ---")

    try:
        media, source = find_media(client, media_id)
    except MediaLookupError as e:
        print(f"  {e}", file=sys.stderr)
        return False, None
    if not media:
        print(f"  Media {media_id} not found in API (may already be deleted)")
        return True, None

    public_id = getattr(media, "public_id", media_id)
    title = getattr(media, "name_romaji", None) or getattr(media, "name_en", None) or media_id
    storage_base_path = storage_path_for(source, str(media_id))
    print(f"  Found: {title} (ID: {public_id}, matched {source}, storage: {storage_base_path})")

    episodes_to_delete = []
    try:
        for ep in client.iter_list_episodes(media_public_id=public_id):
            ep_num = getattr(ep, "episode_number", None)
            if ep_num is None:
                continue
            seg_count = getattr(ep, "segment_count", "?")
            episodes_to_delete.append(ep_num)
            print(f"  Episode {ep_num}: {seg_count} segments")
    except NadeshikoError as e:
        print(f"  Error listing episodes: {e.detail}", file=sys.stderr)
        return False, storage_base_path

    if not episodes_to_delete:
        print("  No episodes found")

    if dry_run:
        print(f"\n  [dry-run] Would delete {len(episodes_to_delete)} episodes + media record")
        return True, storage_base_path

    for ep_num in episodes_to_delete:
        try:
            client.delete_episode(media_public_id=public_id, episode_number=ep_num)
            print(f"  Deleted episode {ep_num}")
        except NadeshikoError as e:
            print(f"  Error deleting episode {ep_num}: {e.detail}")

    try:
        client.delete_media(media_public_id=public_id)
        print(f"  Deleted media record {public_id}")
        return True, storage_base_path
    except NadeshikoError as e:
        print(f"  Error deleting media: {e.detail}")
        return False, storage_base_path


# --- R2 ---

# One media's own folder, nothing broader: media/<anilist numeric id> or media/<mv|yt|ab>/<id>.
# The AniList form is digits only, so a source folder itself ("media/mv") never passes as an id.
_MEDIA_PREFIX_RE = re.compile(r"^media/(?:[0-9]+|(?:mv|yt|ab)/[A-Za-z0-9_.-]+)$")


def validate_storage_path(storage_path: str | None) -> str:
    """Return the R2 prefix of exactly one media folder, or raise ValueError.

    Guards against deleting more than one media: an empty path, `media/`, a path with `..`,
    or anything that isn't a single media folder is refused.
    """
    path = (storage_path or "").strip().strip("/")
    if not _MEDIA_PREFIX_RE.match(path) or ".." in path:
        raise ValueError(
            f"Refusing R2 path {storage_path!r}: expected one media folder such as "
            "media/21804, media/mv/46198 or media/ab/B07PPMBZHX"
        )
    return path


def r2_cleanup_refusal(target: str) -> str | None:
    """Why R2 must not be touched for this target, or None when it may."""
    if target != "prod":
        return (
            f"R2 is not touched for --target {target}: staging and production share the same "
            "files. Remove them with --target prod, which deletes the prod media first."
        )
    return None


def remove_from_r2(storage_path: str, dry_run: bool, skip_confirm: bool = False) -> int:
    """Remove all files under one media's storage path from R2."""
    print("\n--- R2 storage cleanup ---")

    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
    secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket = os.environ.get("R2_BUCKET", "nadeshiko-production")

    missing = [
        v
        for v in ["R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"]
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

    prefix = f"{validate_storage_path(storage_path)}/"

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
        response = input(f"\nDelete {len(all_objects)} objects from {bucket}/{prefix}? [y/N] ")
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Remove media from Nadeshiko API and R2 storage")
    parser.add_argument(
        "media_id",
        nargs="?",
        help="Media ID to remove: AniList/TMDB id (e.g. '20812'), YouTube channel id or ASIN",
    )
    parser.add_argument(
        "--target", required=True, choices=["local", "dev", "prod"], help="Target environment"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--api-only", action="store_true", help="Only delete from API, skip R2")
    parser.add_argument(
        "--r2-only",
        action="store_true",
        help="Only delete orphaned files from R2 (prod only; needs --r2-path)",
    )
    parser.add_argument(
        "--r2-path",
        help="R2 folder for --r2-only, e.g. media/21804 or media/mv/46198 (never guessed)",
    )
    args = parser.parse_args(argv)
    if args.r2_only and args.api_only:
        parser.error("--r2-only and --api-only exclude each other")
    if args.r2_only and not args.r2_path:
        parser.error("--r2-only needs --r2-path: the R2 folder is never guessed from an id")
    if not args.r2_only and not args.media_id:
        parser.error("media_id is required unless --r2-only is given")
    return args


def main(argv=None):
    args = parse_args(argv)

    if args.target == "prod" and not args.dry_run and not args.yes:
        response = input("You are about to delete from PRODUCTION. Are you sure? [y/N] ")
        if response.lower() != "y":
            print("Aborted")
            return 1

    print(f"Removing media {args.media_id or args.r2_path} from {args.target}")
    if args.dry_run:
        print("[DRY RUN MODE]")

    storage_path = args.r2_path
    if not args.r2_only:
        client = get_api_client(args.target)
        api_ok, base_path = remove_from_api(client, args.media_id, args.dry_run)
        if not api_ok:
            print("\nAPI cleanup failed. R2 cleanup skipped.")
            return 1
        storage_path = base_path

    if not args.api_only:
        refusal = r2_cleanup_refusal(args.target)
        if refusal:
            print(f"\n{refusal}")
        elif not storage_path:
            print(
                "\nNo R2 folder (media not found, or matched by tvdb/imdb): R2 left untouched. "
                "Use --r2-only --r2-path for orphaned files."
            )
        else:
            try:
                validate_storage_path(storage_path)
            except ValueError as e:
                print(f"\n{e}", file=sys.stderr)
                return 1
            remove_from_r2(storage_path, args.dry_run, skip_confirm=args.yes or args.dry_run)

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
