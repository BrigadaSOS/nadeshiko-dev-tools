import json
import logging
import os
import subprocess
from collections import namedtuple
from pathlib import Path

import ffmpeg
import requests

logger = logging.getLogger(__name__)

MatchingMkvSource = namedtuple(
    "MatchingMkvSource", ["filepath", "episode", "audio_streams", "subtitle_streams"]
)


def download_and_save_image(url: str, output_dir: str, prefix: str) -> str:
    image_data = requests.get(url).content
    temp_filename = f"{prefix}_temp{os.path.splitext(url)[1]}"
    temp_filepath = os.path.join(output_dir, temp_filename)

    with open(temp_filepath, "wb") as handler:
        handler.write(image_data)

    if prefix == "cover":
        scale_filter = "scale='min(460,iw)':'min(690,ih)'"
    elif prefix == "banner":
        scale_filter = "scale='min(1200,iw)':'min(400,ih)'"
    else:
        scale_filter = None

    filename = f"{prefix}.webp"
    filepath = os.path.join(output_dir, filename)

    if scale_filter:
        subprocess.call(
            [
                "ffmpeg",
                "-y",
                "-i",
                temp_filepath,
                "-vf",
                scale_filter,
                "-c:v",
                "libwebp",
                "-quality",
                "85",
                "-method",
                "6",
                filepath,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        os.remove(temp_filepath)
    else:
        os.rename(temp_filepath, filepath)

    return filename


def discover_input_folders(input_folder: Path) -> list:
    media_folders = []

    try:
        for folder_name in sorted(os.listdir(input_folder)):
            folder_path = os.path.join(input_folder, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # Count .mkv files in this folder
            mkv_files = [f for f in os.listdir(folder_path) if f.endswith(".mkv")]

            if mkv_files:
                media_folders.append(
                    {
                        "name": folder_name,
                        "path": folder_path,
                        "file_count": len(mkv_files),
                        "files": sorted(mkv_files),
                    }
                )
    except PermissionError as e:
        logger.error(f"Permission error accessing {input_folder}: {e}")

    return media_folders


def save_info_json(info_json_path: str, anime_data, anime_folder_name: str) -> str:
    """Save _info.json for an anime folder.

    Returns:
        str: The hash salt for this media.
    """
    info_json_path = os.path.join(os.path.dirname(info_json_path), "_info.json")

    if os.path.exists(info_json_path):
        logger.info(f"_info.json already exists at {info_json_path}, skipping creation")
        # Load existing salt
        with open(info_json_path) as f:
            existing_data = json.load(f)
            return existing_data.get("hash_salt", "")

    logger.info(f"Creating _info.json at {info_json_path}")

    anime_folder_fullpath = os.path.dirname(info_json_path)

    # Generate a unique salt for this media
    import secrets

    hash_salt = secrets.token_hex(16)

    info_json = {
        "id": anime_data.id,
        "anilist_id": anime_data.id,
        "version": "6",
        "japanese_name": getattr(anime_data.title, "native", None),
        "english_name": getattr(anime_data.title, "english", None),
        "romaji_name": anime_data.title.romaji,
        "airing_format": anime_data.format,
        "airing_status": anime_data.status,
        "genres": anime_data.genres or [],
        "episodes": anime_data.episodes,
        "hash_salt": hash_salt,
    }

    if hasattr(anime_data, "cover") and hasattr(anime_data.cover, "extra_large"):
        cover_filename = download_and_save_image(
            anime_data.cover.extra_large, anime_folder_fullpath, "cover"
        )
        info_json["cover"] = cover_filename

    if hasattr(anime_data, "banner") and anime_data.banner:
        banner_filename = download_and_save_image(
            anime_data.banner, anime_folder_fullpath, "banner"
        )
        info_json["banner"] = banner_filename

    logger.info(f"Json Data: {info_json}")

    with open(info_json_path, "wb") as f:
        json_data = json.dumps(info_json, indent=2, ensure_ascii=False).encode("utf8")
        f.write(json_data)

    return hash_salt


def write_data_json(
    output_path: str,
    segments: list,
    episode_number: int,
    duration_ms: int,
    anime_data,
    ignored_segments: list,
):
    """Write _data.json file with segment information."""
    data_json = {
        "metadata": {
            "version": "6",
            "number": episode_number,
            "duration_ms": duration_ms,
            "total_segments": len(segments),
        },
        "media": {
            "anilist_id": anime_data.id,
        },
        "segments": segments,
        "ignored_segments": ignored_segments or [],
    }

    data_json_path = os.path.join(output_path, "_data.json")
    with open(data_json_path, "w", encoding="utf-8") as f:
        json.dump(data_json, f, ensure_ascii=False, indent=2)

    logger.debug(f"[E{episode_number}] Created _data.json at {data_json_path}")


def discover_matching_mkv_files(episode_filepath: str, episode_number: int) -> list:
    """Find other .mkv files in the same directory that match the episode number.

    Args:
        episode_filepath: Path to the current episode being processed
        episode_number: The episode number to match

    Returns:
        List of MatchingMkvSource objects including the current file
    """
    from guessit import guessit

    from media_sub_splitter.utils.text_utils import extract_anime_title_for_guessit

    parent_dir = Path(episode_filepath).parent
    matching_files = []

    try:
        for filename in os.listdir(parent_dir):
            if not filename.endswith(".mkv"):
                continue

            filepath = os.path.join(parent_dir, filename)

            # Get episode info using guessit
            cleaned_name = extract_anime_title_for_guessit(filepath)
            guessed_info = guessit(cleaned_name)
            file_episode = guessed_info.get("episode")

            if file_episode == episode_number:
                try:
                    # Probe the file for stream info
                    probe = ffmpeg.probe(filepath)
                    audio_streams = [s for s in probe["streams"] if s["codec_type"] == "audio"]
                    subtitle_streams = [
                        s for s in probe["streams"] if s["codec_type"] == "subtitle"
                    ]

                    matching_files.append(
                        MatchingMkvSource(
                            filepath=filepath,
                            episode=episode_number,
                            audio_streams=audio_streams,
                            subtitle_streams=subtitle_streams,
                        )
                    )
                    logger.debug(f"Found matching MKV: {filename} (episode {file_episode})")
                except ffmpeg.Error as e:
                    logger.warning(f"Could not probe {filepath}: {e}")
    except PermissionError as e:
        logger.error(f"Permission error accessing {parent_dir}: {e}")

    # Sort by filename for consistent ordering
    matching_files.sort(key=lambda x: x.filepath)

    return matching_files
