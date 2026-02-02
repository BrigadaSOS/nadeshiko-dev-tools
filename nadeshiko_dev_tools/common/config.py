import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_FILE = ".media-sub-splitter.json"

# Config keys
KEY_AUDIO = "audio"
KEY_SUBTITLES = "subtitles"
KEY_MULTI_MKV = "multi_mkv"
KEY_UPLOADER = "uploader"


@dataclass
class ProcessingConfig:
    input_folder: Path
    output_folder: Path
    deepl_token: str | None = None
    verbose: bool = False
    dryrun: bool = False
    extra_punctuation: bool = False
    parallel: bool = False
    pool_size: int = 6
    episodes: set[int] | None = None  # None means all episodes
    sync_external_subs: bool | None = None  # None means prompt user


def load_subtitle_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
    return {}


def save_subtitle_config(config: dict) -> None:
    """Save subtitle configuration to config file."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save config: {e}")


def load_multi_mkv_config() -> dict:
    """Load multi-MKV configuration from config file.

    Returns:
        Dict mapping folder names to their multi-MKV selections.
        Example:
        {
            "Season 01": {
                "main_mkv_index": 0,
                "audio_index": 1,
                "subtitle_file_indices": [0, 1]
            }
        }
    """
    config = load_subtitle_config()
    return config.get(KEY_MULTI_MKV, {})


def save_multi_mkv_config(multi_mkv_config: dict) -> None:
    """Save multi-MKV configuration to config file.

    Args:
        multi_mkv_config: Dict mapping folder names to their selections.
    """
    config = load_subtitle_config()
    config[KEY_MULTI_MKV] = multi_mkv_config
    save_subtitle_config(config)


def get_multi_mkv_selection(folder_name: str) -> dict | None:
    """Get multi-MKV selection for a specific folder.

    Args:
        folder_name: The folder name

    Returns:
        Dict with keys: main_mkv_index, audio_index, subtitle_file_indices
        or None if not found.
    """
    multi_mkv_config = load_multi_mkv_config()
    return multi_mkv_config.get(folder_name)


def save_multi_mkv_selection(
    folder_name: str,
    main_mkv_index: int,
    audio_index: int | None,
    subtitle_file_indices: list[int],
) -> None:
    """Save a multi-MKV selection for a folder.

    Args:
        folder_name: The folder name
        main_mkv_index: Index of the main MKV file in the matching list
        audio_index: Index of the audio track to use
        subtitle_file_indices: List of file indices to extract subtitles from
    """
    multi_mkv_config = load_multi_mkv_config()
    multi_mkv_config[folder_name] = {
        "main_mkv_index": main_mkv_index,
        "audio_index": audio_index,
        "subtitle_file_indices": subtitle_file_indices,
    }
    save_multi_mkv_config(multi_mkv_config)


def clear_multi_mkv_selection(folder_name: str) -> None:
    """Remove a multi-MKV selection for a folder.

    Args:
        folder_name: The folder name to remove
    """
    multi_mkv_config = load_multi_mkv_config()
    if folder_name in multi_mkv_config:
        del multi_mkv_config[folder_name]
        save_multi_mkv_config(multi_mkv_config)
