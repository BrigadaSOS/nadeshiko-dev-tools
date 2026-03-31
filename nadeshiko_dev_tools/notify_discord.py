#!/usr/bin/env python3
"""Send a Discord webhook notification for a newly published anime on Nadeshiko.

Usage:
    uv run python scripts/notify_discord.py <ANILIST_ID>
    uv run python scripts/notify_discord.py <ANILIST_ID> <OUTPUT_FOLDER>
    uv run python scripts/notify_discord.py 128547 --dry-run

Stats are fetched from the Nadeshiko API by default. If an output folder is
provided, stats are computed from local _data.json files instead.
"""

import argparse
import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()


def get_anilist_info(anilist_id: int) -> dict:
    """Fetch anime details from AniList GraphQL API."""
    resp = requests.post(
        "https://graphql.anilist.co",
        json={
            "query": "{ Media(id: %d) { title { romaji english native } episodes coverImage { large } } }"
            % anilist_id
        },
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()["data"]["Media"]


def get_nadeshiko_public_id(anilist_id: int, target: str) -> str | None:
    """Find the Nadeshiko publicId for a given AniList ID."""
    from nadeshiko_internal import Nadeshiko
    from nadeshiko_internal.api.media import list_media
    from nadeshiko_internal.types import UNSET

    env_keys = {
        "dev": ("NADESHIKO_DEV_API_KEY", "https://api-dev.nadeshiko.co"),
        "prod": ("NADESHIKO_PROD_API_KEY", "https://api.nadeshiko.co"),
    }
    key_var, base_url = env_keys.get(target, env_keys["prod"])
    api_key = os.getenv(key_var)
    if not api_key:
        print(f"Warning: {key_var} not set", file=sys.stderr)
        return None

    client = Nadeshiko(token=api_key, base_url=base_url)
    result = list_media.sync(client=client)
    if not result or not hasattr(result, "media"):
        return None

    for media in result.media:
        ext_ids = getattr(media, "external_ids", UNSET)
        if ext_ids is UNSET or ext_ids is None:
            continue
        if str(getattr(ext_ids, "anilist", "")) == str(anilist_id):
            return media.public_id
    return None


def compute_stats_local(output_folder: str) -> tuple[int, int, float]:
    """Compute total episodes, segments, and duration hours from local _data.json files."""
    total_episodes = 0
    total_segments = 0
    total_duration_ms = 0

    for entry in sorted(os.listdir(output_folder)):
        data_path = os.path.join(output_folder, entry, "_data.json")
        if not os.path.isfile(data_path):
            continue
        with open(data_path) as f:
            data = json.load(f)
        segments = data.get("segments", [])
        total_episodes += 1
        total_segments += len(segments)
        total_duration_ms += sum(s.get("duration_ms", 0) for s in segments)

    return total_episodes, total_segments, total_duration_ms / 3_600_000


def compute_stats_api(public_id: str, target: str) -> tuple[int, int, float]:
    """Fetch episode/segment stats from the Nadeshiko API."""
    from nadeshiko_internal import Nadeshiko
    from nadeshiko_internal.api.media import list_media

    env_keys = {
        "dev": ("NADESHIKO_DEV_API_KEY", "https://api-dev.nadeshiko.co"),
        "prod": ("NADESHIKO_PROD_API_KEY", "https://api.nadeshiko.co"),
    }
    key_var, base_url = env_keys.get(target, env_keys["prod"])
    client = Nadeshiko(token=os.getenv(key_var), base_url=base_url)

    # Get total segment count and episode count from media
    result = list_media.sync(client=client)
    media = None
    for m in (result.media if result and hasattr(result, "media") else []):
        if m.public_id == public_id:
            media = m
            break

    if not media:
        return 0, 0, 0.0

    episode_count = getattr(media, "episode_count", 0) or 0
    segment_count = getattr(media, "segment_count", 0) or 0

    # Estimate duration: ~3s per segment is typical
    estimated_hours = (segment_count * 3) / 3600

    return episode_count, segment_count, estimated_hours


def send_webhook(
    webhook_url: str,
    public_id: str,
    english_title: str,
    native_title: str,
    romaji_title: str,
    episodes: int,
    segments: int,
    hours: float,
    cover_url: str,
) -> bool:
    """Send Discord embed webhook. Returns True on success."""
    nadeshiko_url = f"https://nadeshiko.co/search?media={public_id}"

    alt_names = [n for n in [native_title, romaji_title] if n and n != english_title]
    alt_line = f"**Alternative names:** {', '.join(alt_names)}\n" if alt_names else ""

    embed = {
        "title": "New anime content on Nadeshiko!",
        "url": nadeshiko_url,
        "description": (
            f"**Name:** {english_title or romaji_title}\n"
            f"{alt_line}"
            f"**Episodes:** {episodes}\n"
            f"**Sentences:** {segments:,}\n"
            f"**Duration:** {hours:.1f} hours\n\n"
            f"[**View on Nadeshiko \u2192**]({nadeshiko_url})"
        ),
        "color": 16739688,
        "image": {"url": cover_url},
    }

    resp = requests.post(webhook_url, json={"embeds": [embed]})
    if resp.status_code in (200, 204):
        return True
    print(f"Webhook failed: {resp.status_code} {resp.text}", file=sys.stderr)
    return False


def main():
    parser = argparse.ArgumentParser(description="Send Discord notification for new Nadeshiko anime")
    parser.add_argument("anilist_id", type=int, help="AniList media ID")
    parser.add_argument("output_folder", nargs="?", default=None, help="Processed output folder (optional — stats fetched from API if omitted)")
    parser.add_argument("--target", default="prod", choices=["dev", "prod"], help="API target (default: prod)")
    parser.add_argument("--dry-run", action="store_true", help="Print embed without sending")
    args = parser.parse_args()

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url and not args.dry_run:
        print("Error: DISCORD_WEBHOOK_URL not set in .env", file=sys.stderr)
        return 1

    print(f"Fetching AniList info for {args.anilist_id}...")
    anilist = get_anilist_info(args.anilist_id)
    title = anilist["title"]

    print(f"Looking up Nadeshiko publicId on {args.target}...")
    public_id = get_nadeshiko_public_id(args.anilist_id, args.target)
    if not public_id:
        print("Error: Could not find media on Nadeshiko API", file=sys.stderr)
        return 1

    if args.output_folder:
        print(f"Computing stats from {args.output_folder}...")
        episodes, segments, hours = compute_stats_local(args.output_folder)
    else:
        print(f"Fetching stats from Nadeshiko API...")
        episodes, segments, hours = compute_stats_api(public_id, args.target)

    print(f"\n  Title: {title.get('english') or title['romaji']}")
    print(f"  PublicId: {public_id}")
    print(f"  Episodes: {episodes}, Segments: {segments:,}, Duration: {hours:.1f}h")
    print(f"  Cover: {anilist['coverImage']['large']}")

    if args.dry_run:
        print("\n[DRY RUN] Would send webhook")
        return 0

    print("\nSending Discord webhook...")
    ok = send_webhook(
        webhook_url=webhook_url,
        public_id=public_id,
        english_title=title.get("english"),
        native_title=title.get("native"),
        romaji_title=title.get("romaji"),
        episodes=episodes,
        segments=segments,
        hours=hours,
        cover_url=anilist["coverImage"]["large"],
    )
    print("Sent!" if ok else "Failed!")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
