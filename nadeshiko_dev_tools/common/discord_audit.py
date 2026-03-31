"""Discord audit webhook for processing progress updates.

Posts to DISCORD_AUDIT_WEBHOOK_URL (internal audit channel, not user-facing).
"""

import json
import logging
import os
import urllib.request
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

COLOR_PROGRESS = 0x5865F2  # blurple
COLOR_SUCCESS = 0x57F287  # green
COLOR_FAILURE = 0xED4245  # red

# Module-level state — call init() before post()
_enabled = False
_title = ""
_anilist_id = 0


def init(enabled: bool, title: str, anilist_id: int):
    """Initialize audit webhook state."""
    global _enabled, _title, _anilist_id
    _enabled = enabled
    _title = title
    _anilist_id = anilist_id


def post(message: str, stage: str, color: int = COLOR_PROGRESS, fields: list | None = None):
    """Post to the internal audit Discord webhook. No-op if not enabled."""
    if not _enabled:
        return

    webhook_url = os.environ.get("DISCORD_AUDIT_WEBHOOK_URL")
    if not webhook_url:
        return

    embed = {
        "title": _title,
        "description": message,
        "color": color,
        "fields": [
            {"name": "Stage", "value": stage, "inline": True},
            {"name": "AniList ID", "value": str(_anilist_id), "inline": True},
        ],
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if fields:
        embed["fields"].extend(fields)

    payload = json.dumps({"embeds": [embed]}).encode("utf-8")
    try:
        req = urllib.request.Request(
            webhook_url, data=payload, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        logger.warning(f"Audit webhook failed: {e}")
