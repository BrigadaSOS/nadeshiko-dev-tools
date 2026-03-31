import json
import logging
import os
import subprocess

import requests

logger = logging.getLogger(__name__)


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



def _format_fuzzy_date(date_obj) -> str | None:
    """Format AniList fuzzy date to ISO format.

    Args:
        date_obj: AniList date object with year, month, day attributes

    Returns:
        ISO format date string (YYYY-MM-DD) or None if date is incomplete
    """
    if not date_obj:
        return None
    year = getattr(date_obj, "year", None)
    month = getattr(date_obj, "month", None)
    day = getattr(date_obj, "day", None)
    if year and month and day:
        return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def save_info_json(
    info_json_path: str,
    anime_data,
    anime_folder_name: str,
) -> str:
    """Save _info.json for an anime folder.

    Returns:
        str: The hash salt for this media.
    """
    info_json_path = os.path.join(os.path.dirname(info_json_path), "_info.json")

    if os.path.exists(info_json_path):
        logger.info(f"_info.json already exists at {info_json_path}, skipping creation")
        # Load existing salt
        try:
            with open(info_json_path) as f:
                existing_data = json.load(f)
                return existing_data.get("hash_salt", "")
        except json.JSONDecodeError:
            logger.warning(f"_info.json at {info_json_path} is corrupted or empty, recreating it")
            # Fall through to recreate the file

    logger.info(f"Creating _info.json at {info_json_path}")

    anime_folder_fullpath = os.path.dirname(info_json_path)

    # Generate a unique salt for this media
    import secrets

    hash_salt = secrets.token_hex(16)

    info_json = {
        "id": anime_data.id,
        "anilist_id": anime_data.id,
        "media_source": "anilist",
        "category": "ANIME",
        "version": "6",
        "japanese_name": getattr(anime_data.title, "native", None),
        "english_name": getattr(anime_data.title, "english", None),
        "romaji_name": anime_data.title.romaji,
        "airing_format": str(anime_data.format) if anime_data.format else None,
        "airing_status": str(anime_data.status) if anime_data.status else None,
        "source": str(anime_data.source)
        if hasattr(anime_data, "source") and anime_data.source
        else None,
        "genres": anime_data.genres or [],
        "episodes": anime_data.episodes,
        "hash_salt": hash_salt,
    }

    # Add start_date and end_date from AniList
    if hasattr(anime_data, "start_date") and anime_data.start_date:
        start_date = _format_fuzzy_date(anime_data.start_date)
        if start_date:
            info_json["start_date"] = start_date

    if hasattr(anime_data, "end_date") and anime_data.end_date:
        end_date = _format_fuzzy_date(anime_data.end_date)
        if end_date:
            info_json["end_date"] = end_date

    # Add studio (first one only)
    if hasattr(anime_data, "studios") and anime_data.studios:
        if hasattr(anime_data.studios, "nodes") and anime_data.studios.nodes:
            info_json["studio"] = anime_data.studios.nodes[0].name
        else:
            info_json["studio"] = None

    # Add season as a proper object
    if hasattr(anime_data, "season") and anime_data.season:
        season_obj = {
            "name": str(anime_data.season),
        }
        if hasattr(anime_data, "season_year") and anime_data.season_year:
            season_obj["year"] = anime_data.season_year
        info_json["season"] = season_obj
    elif hasattr(anime_data, "season_year") and anime_data.season_year:
        # If only season_year exists without season
        info_json["season"] = {"year": anime_data.season_year}

    # Add synonyms
    if hasattr(anime_data, "synonyms") and anime_data.synonyms:
        info_json["synonyms"] = anime_data.synonyms

    # Add relations (related anime)
    if hasattr(anime_data, "relations") and anime_data.relations:
        info_json["relations"] = (
            [
                {
                    "relationType": str(rel.relationType) if hasattr(rel, "relationType") else None,
                    "relationTitleId": rel.node.id,
                    "relationTitleEnglish": getattr(rel.node.title, "english", None),
                    "relationTitleNative": getattr(rel.node.title, "native", None),
                    "relationTitleRomaji": getattr(rel.node.title, "romaji", None),
                    "relationTitleType": str(rel.node.type) if hasattr(rel.node, "type") else None,
                }
                for rel in anime_data.relations.edges
            ]
            if hasattr(anime_data.relations, "edges")
            else []
        )

    # Add streaming episodes (legal streaming links)
    if hasattr(anime_data, "streaming_episodes") and anime_data.streaming_episodes:
        info_json["streaming_episodes"] = (
            [
                {"title": ep.title, "url": ep.url, "site": ep.site}
                for ep in anime_data.streaming_episodes
            ]
            if hasattr(anime_data.streaming_episodes, "__iter__")
            else []
        )

    # Add characters (main characters and voice actors) - flattened structure
    if hasattr(anime_data, "characters") and anime_data.characters:
        characters_list = []
        if hasattr(anime_data.characters, "edges"):
            for char in anime_data.characters.edges:
                char_data = {
                    "characterId": char.node.id if hasattr(char, "node") else None,
                    "characterNameEnglish": getattr(char.node.name, "full", None)
                    if hasattr(char, "node") and hasattr(char.node, "name")
                    else None,
                    "characterNameJapanese": getattr(char.node.name, "native", None)
                    if hasattr(char, "node") and hasattr(char.node, "name")
                    else None,
                    "characterImageUrl": getattr(char.node.image, "medium", None)
                    if hasattr(char, "node") and hasattr(char.node, "image")
                    else None,
                    "characterRole": str(char.role) if hasattr(char, "role") else None,
                }

                # Add first voice actor if available
                if hasattr(char, "voiceActors") and char.voiceActors:
                    va = char.voiceActors[0]
                    char_data["seiyuuId"] = va.id if hasattr(va, "id") else None
                    char_data["seiyuuNameEnglish"] = (
                        getattr(va.name, "full", None) if hasattr(va, "name") else None
                    )
                    char_data["seiyuuNameJapanese"] = (
                        getattr(va.name, "native", None) if hasattr(va, "name") else None
                    )
                    char_data["seiyuuImageUrl"] = (
                        getattr(va.image, "medium", None) if hasattr(va, "image") else None
                    )
                else:
                    char_data["seiyuuId"] = None
                    char_data["seiyuuNameEnglish"] = None
                    char_data["seiyuuNameJapanese"] = None
                    char_data["seiyuuImageUrl"] = None

                characters_list.append(char_data)

        info_json["characters"] = characters_list

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


