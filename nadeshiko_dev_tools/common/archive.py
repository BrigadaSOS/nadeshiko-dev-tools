"""Shared archive discovery utilities.

The processed archive has the structure:
  {archive_dir}/{anilist_id}/{episode_number}/
    *.mp3          — audio segments
    *.webp         — preview screenshots
    *.mp4          — video segments
    _data.json     — segment metadata
    _info.json     — media metadata (at media level)
"""

from pathlib import Path


def discover_media(archive_dir: Path) -> list[Path]:
    """Find all media folders (numeric AniList IDs) in the archive."""
    return sorted(
        [d for d in archive_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )


def discover_episodes(media_dir: Path) -> list[Path]:
    """Find all episode folders within a media directory."""
    return sorted(
        [d for d in media_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )


def discover_files(episode_dir: Path, pattern: str) -> list[Path]:
    """Find files matching a glob pattern in an episode directory."""
    return sorted(episode_dir.glob(pattern))


def filter_media_by_ids(
    media_dirs: list[Path], media_ids: list[int] | None
) -> list[Path]:
    """Filter media directories to only those matching the given IDs."""
    if not media_ids:
        return media_dirs
    media_set = {str(m) for m in media_ids}
    return [d for d in media_dirs if d.name in media_set]
