#!/usr/bin/env python3
"""Send a Discord webhook notification for newly published Nadeshiko media.

Examples:
    notify-discord 128547
    notify-discord --source anilist 128547 --target prod
    notify-discord --source youtube UCxvDCtgrqL2r-GrhYj-mQfQ --target prod \
        --output-folder /mnt/storage/youtube/UC...
    notify-discord --public-id 7yUBi43pGq00 --source youtube --dry-run

Stats are fetched from the Nadeshiko API by default. If an output folder is
provided, stats are computed from local _data.json files instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class MediaStats:
    episodes: int
    segments: int
    hours: float
    generated_langs: list[str] = field(default_factory=list)


@dataclass
class NotificationInfo:
    public_id: str
    english_title: str | None
    native_title: str | None
    romaji_title: str | None
    cover_url: str | None


def _api_env(target: str) -> tuple[str, str]:
    if target == "dev":
        return (
            "NADESHIKO_DEV_API_KEY",
            os.getenv("NADESHIKO_DEV_BASE_URL") or "https://api-dev.nadeshiko.co",
        )
    return (
        "NADESHIKO_PROD_API_KEY",
        os.getenv("NADESHIKO_PROD_BASE_URL") or "https://api.nadeshiko.co",
    )


def build_api_client(target: str):
    """Build a Nadeshiko SDK client for the selected target."""
    from nadeshiko_internal import Nadeshiko

    key_var, base_url = _api_env(target)
    api_key = os.getenv(key_var)
    if not api_key:
        raise RuntimeError(f"{key_var} not set")
    return Nadeshiko(
        api_key=api_key,
        base_url=base_url,
        headers={"User-Agent": "NadeshikoDevTools/1.0"},
    )


def get_anilist_info(anilist_id: int) -> dict:
    """Fetch anime details from AniList GraphQL API."""
    query = (
        f"{{ Media(id: {anilist_id}) {{ title {{ romaji english native }}"
        f" episodes coverImage {{ large }} }} }}"
    )
    resp = requests.post(
        "https://graphql.anilist.co",
        json={"query": query},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"]["Media"]


def _external_id(media: Any, source: str) -> str | None:
    from nadeshiko_internal.types import UNSET

    ext_ids = getattr(media, "external_ids", UNSET)
    if ext_ids is UNSET or ext_ids is None:
        return None
    value = getattr(ext_ids, source, None)
    return None if value is None else str(value)


def iter_matching_media(media_id: str, target: str, source: str):
    """Yield API media whose external ID for source matches media_id."""
    from nadeshiko_internal import NadeshikoError

    client = build_api_client(target)
    try:
        for media in client.iter_list_media():
            if _external_id(media, source) == str(media_id):
                yield media
    except NadeshikoError as e:
        print(f"Error listing media: {e.detail}", file=sys.stderr)


def get_nadeshiko_public_id(
    media_id: str | int, target: str, source: str = "anilist"
) -> str | None:
    """Find a Nadeshiko publicId by external AniList or YouTube ID."""
    for media in iter_matching_media(str(media_id), target, source):
        return media.public_id
    return None


def _title_part(title_obj: Any, key: str) -> str | None:
    if title_obj is None:
        return None
    if isinstance(title_obj, dict):
        return title_obj.get(key)
    return getattr(title_obj, key, None)


def _media_cover_url(media: Any) -> str | None:
    for attr in ("cover_image_url", "cover_url", "cover", "banner_image_url"):
        value = getattr(media, attr, None)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            nested = value.get("large") or value.get("url")
            if nested:
                return nested
    return None


def get_local_notification_info(output_folder: str, public_id: str) -> NotificationInfo | None:
    """Read title/cover hints from a local channel/media _info.json when available."""
    info_path = os.path.join(output_folder, "_info.json")
    if not os.path.isfile(info_path):
        return None
    with open(info_path) as f:
        data = json.load(f)

    name = (
        data.get("name")
        or data.get("english_name")
        or data.get("romaji_name")
        or data.get("japanese_name")
        or data.get("channel_id")
    )
    channel_id = data.get("channel_id")
    cover_url = data.get("cover_url") or data.get("cover")
    if cover_url and not str(cover_url).startswith(("http://", "https://")):
        if channel_id:
            cover_url = f"https://cdn.nadeshiko.co/media/yt/{channel_id}/{cover_url}"
        else:
            cover_path = os.path.join(output_folder, str(cover_url))
            cover_url = cover_path if os.path.exists(cover_path) else None
    return NotificationInfo(
        public_id=public_id,
        english_title=name,
        native_title=data.get("japanese_name"),
        romaji_title=data.get("romaji_name") or name,
        cover_url=cover_url,
    )


def get_api_notification_info(
    media_id: str, target: str, source: str, public_id: str | None = None
) -> NotificationInfo | None:
    """Fetch notification title/cover data from Nadeshiko API for non-AniList sources."""
    from nadeshiko_internal import NadeshikoError

    try:
        client = build_api_client(target)
        media_iter = client.iter_list_media()
        for media in media_iter:
            if public_id and getattr(media, "public_id", None) != public_id:
                continue
            if not public_id and _external_id(media, source) != str(media_id):
                continue
            title = getattr(media, "title", None)
            name = (
                _title_part(title, "english")
                or _title_part(title, "romaji")
                or _title_part(title, "native")
                or getattr(media, "english_name", None)
                or getattr(media, "romaji_name", None)
                or getattr(media, "japanese_name", None)
                or str(media_id)
            )
            return NotificationInfo(
                public_id=getattr(media, "public_id", public_id),
                english_title=_title_part(title, "english") or name,
                native_title=_title_part(title, "native"),
                romaji_title=_title_part(title, "romaji") or name,
                cover_url=_media_cover_url(media),
            )
    except NadeshikoError as e:
        print(f"Error listing media: {e.detail}", file=sys.stderr)
    return None


def compute_stats_local(output_folder: str) -> MediaStats:
    """Compute total videos/episodes, segments, duration hours, and generated langs."""
    total_episodes = 0
    total_segments = 0
    total_duration_ms = 0
    generated_langs: set[str] = set()

    for entry in sorted(os.listdir(output_folder)):
        data_path = os.path.join(output_folder, entry, "_data.json")
        if not os.path.isfile(data_path):
            continue
        with open(data_path) as f:
            data = json.load(f)
        segments = data.get("segments", [])
        total_episodes += 1
        total_segments += len(segments)
        for segment in segments:
            total_duration_ms += segment.get("duration_ms", 0)
            if segment.get("is_mt_es"):
                generated_langs.add("es")
            if segment.get("is_mt_en"):
                generated_langs.add("en")

    return MediaStats(
        episodes=total_episodes,
        segments=total_segments,
        hours=total_duration_ms / 3_600_000,
        generated_langs=sorted(generated_langs),
    )


def compute_stats_api(public_id: str, target: str) -> MediaStats:
    """Fetch episode/video and segment stats from the Nadeshiko API."""
    from nadeshiko_internal import NadeshikoError

    client = build_api_client(target)

    found_media = None
    try:
        for media in client.iter_list_media():
            if media.public_id == public_id:
                found_media = media
                break
    except NadeshikoError:
        pass

    if not found_media:
        return MediaStats(episodes=0, segments=0, hours=0.0)

    episode_count = getattr(found_media, "episode_count", 0) or 0
    segment_count = getattr(found_media, "segment_count", 0) or 0
    estimated_hours = (segment_count * 3) / 3600

    return MediaStats(episodes=episode_count, segments=segment_count, hours=estimated_hours)


def nadeshiko_url(public_id: str, target: str) -> str:
    base = "https://dev.nadeshiko.co" if target == "dev" else "https://nadeshiko.co"
    return f"{base}/search?media={public_id}"


def build_webhook_payload(
    public_id: str,
    source: str,
    english_title: str | None,
    native_title: str | None,
    romaji_title: str | None,
    episodes: int,
    segments: int,
    hours: float,
    cover_url: str | None,
    target: str = "prod",
) -> dict:
    """Build Discord webhook payload for the selected source."""
    url = nadeshiko_url(public_id, target)
    display_title = english_title or romaji_title or native_title or public_id
    alt_names = [n for n in [native_title, romaji_title] if n and n != display_title]
    alt_line = f"**Alternative names:** {', '.join(alt_names)}\n" if alt_names else ""
    is_youtube = source == "youtube"
    count_label = "Videos" if is_youtube else "Episodes"

    embed = {
        "title": "New YouTube content on Nadeshiko!"
        if is_youtube
        else "New anime content on Nadeshiko!",
        "url": url,
        "description": (
            f"**Name:** {display_title}\n"
            f"{alt_line}"
            f"**{count_label}:** {episodes}\n"
            f"**Sentences:** {segments:,}\n"
            f"**Duration:** {hours:.1f} hours\n"
            "\n"
            f"[**View on Nadeshiko →**]({url})"
        ),
        "color": 16739688,
    }
    if cover_url:
        embed["image"] = {"url": cover_url}
    return {"embeds": [embed]}


def send_webhook_payload(webhook_url: str, payload: dict) -> bool:
    """Send Discord embed webhook. Returns True on success."""
    resp = requests.post(webhook_url, json=payload, timeout=30)
    if resp.status_code in (200, 204):
        return True
    print(f"Webhook failed: {resp.status_code} {resp.text}", file=sys.stderr)
    return False


def send_webhook(
    webhook_url: str,
    public_id: str,
    english_title: str | None,
    native_title: str | None,
    romaji_title: str | None,
    episodes: int,
    segments: int,
    hours: float,
    cover_url: str | None,
    source: str = "anilist",
    target: str = "prod",
) -> bool:
    """Backward-compatible wrapper around payload creation + POST."""
    return send_webhook_payload(
        webhook_url,
        build_webhook_payload(
            public_id=public_id,
            source=source,
            english_title=english_title,
            native_title=native_title,
            romaji_title=romaji_title,
            episodes=episodes,
            segments=segments,
            hours=hours,
            cover_url=cover_url,
            target=target,
        ),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send Discord notification for new Nadeshiko media"
    )
    parser.add_argument(
        "media_id",
        nargs="?",
        help="External media ID (AniList ID by default, YouTube channel ID with --source youtube)",
    )
    parser.add_argument(
        "legacy_output_folder",
        nargs="?",
        default=None,
        help="Backward-compatible positional output folder",
    )
    parser.add_argument(
        "--source",
        default="anilist",
        choices=["anilist", "youtube"],
        help="External source used to look up Nadeshiko media (default: anilist)",
    )
    parser.add_argument(
        "--public-id",
        default=None,
        help="Nadeshiko publicId; skips external-ID lookup when supplied",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Processed output/channel folder for local stats and generated language flags",
    )
    parser.add_argument(
        "--target",
        default="prod",
        choices=["dev", "prod"],
        help="API/frontend target (default: prod)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print embed without sending")
    args = parser.parse_args(argv)
    args.output_folder = args.output_folder or args.legacy_output_folder
    if not args.media_id and not args.public_id:
        parser.error("media_id or --public-id is required")
    return args


def _notification_info(args: argparse.Namespace, public_id: str) -> NotificationInfo | None:
    media_id = args.media_id or ""
    if args.source == "anilist" and args.media_id:
        print(f"Fetching AniList info for {args.media_id}...")
        anilist = get_anilist_info(int(args.media_id))
        title = anilist["title"]
        return NotificationInfo(
            public_id=public_id,
            english_title=title.get("english"),
            native_title=title.get("native"),
            romaji_title=title.get("romaji"),
            cover_url=anilist["coverImage"]["large"],
        )
    local_info = (
        get_local_notification_info(args.output_folder, public_id)
        if args.output_folder
        else None
    )
    if local_info and args.source == "youtube":
        return local_info
    return get_api_notification_info(media_id, args.target, args.source, public_id=public_id)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url and not args.dry_run:
        print("Error: DISCORD_WEBHOOK_URL not set in .env", file=sys.stderr)
        return 1

    public_id = args.public_id
    if not public_id:
        print(f"Looking up Nadeshiko publicId on {args.target} by {args.source}...")
        public_id = get_nadeshiko_public_id(args.media_id, args.target, source=args.source)
        if not public_id:
            print("Error: Could not find media on Nadeshiko API", file=sys.stderr)
            return 1

    info = _notification_info(args, public_id)
    if not info:
        print("Error: Could not load media notification info", file=sys.stderr)
        return 1

    if args.output_folder:
        print(f"Computing stats from {args.output_folder}...")
        stats = compute_stats_local(args.output_folder)
    else:
        print("Fetching stats from Nadeshiko API...")
        stats = compute_stats_api(public_id, args.target)

    print(f"\n  Source: {args.source}")
    print(f"  Title: {info.english_title or info.romaji_title or info.native_title}")
    print(f"  PublicId: {public_id}")
    print(
        f"  {'Videos' if args.source == 'youtube' else 'Episodes'}: {stats.episodes}, "
        f"Segments: {stats.segments:,}, Duration: {stats.hours:.1f}h"
    )
    if info.cover_url:
        print(f"  Cover: {info.cover_url}")

    payload = build_webhook_payload(
        public_id=public_id,
        source=args.source,
        english_title=info.english_title,
        native_title=info.native_title,
        romaji_title=info.romaji_title,
        episodes=stats.episodes,
        segments=stats.segments,
        hours=stats.hours,
        cover_url=info.cover_url,
        target=args.target,
    )

    if args.dry_run:
        print("\n[DRY RUN] Would send webhook")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("\nSending Discord webhook...")
    ok = send_webhook_payload(webhook_url, payload)
    print("Sent!" if ok else "Failed!")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
