"""Core migration logic for converting v5 format to v6 format."""

import csv
import json
import logging
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import questionary
from rich.console import Console

from nadeshiko_dev_tools.common.anilist import CachedAnilist
from nadeshiko_dev_tools.common.file_utils import save_info_json, write_data_json
from nadeshiko_dev_tools.media_sub_splitter.main import generate_segment_hash

console = Console()
logger = logging.getLogger(__name__)

MAX_SEGMENT_CONTENT_LENGTH = 500


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------


def parse_timestamp_to_ms(timestamp: str) -> int:
    """Convert 'H:MM:SS.ffffff' or 'H:MM:SS' timestamp to integer milliseconds."""
    match = re.match(r"(\d+):(\d+):(\d+)(?:\.(\d+))?", timestamp.strip())
    if not match:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    hours, minutes, seconds, frac = match.groups()
    frac = frac or "0"
    # Normalize fractional part to microseconds (6 digits)
    frac = frac.ljust(6, "0")[:6]
    total_ms = (
        int(hours) * 3600000
        + int(minutes) * 60000
        + int(seconds) * 1000
        + int(frac) // 1000
    )
    return total_ms


# ---------------------------------------------------------------------------
# TSV parsing
# ---------------------------------------------------------------------------


def parse_data_tsv(path: str) -> list[dict]:
    """Parse a v5 data.tsv file into a list of segment dicts.

    Returns list of dicts with keys:
        id, subs_jp_ids, subs_es_ids, subs_en_ids,
        start_ms, end_ms,
        name_audio, name_screenshot,
        content_ja, content_es, content_en,
        is_mt_es, is_mt_en,
        actor_ja, actor_es, actor_en
    """
    segments = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            def parse_ids(raw: str) -> list[int]:
                if not raw or not raw.strip():
                    return []
                return [int(x) for x in raw.split(",") if x.strip()]

            def parse_bool(raw: str) -> bool:
                if not raw or not raw.strip():
                    return False
                return raw.strip().lower() == "true"

            segment = {
                "id": int(row["ID"]) if row.get("ID", "").strip() else 0,
                "subs_jp_ids": parse_ids(row.get("SUBS_JP_IDS", "")),
                "subs_es_ids": parse_ids(row.get("SUBS_ES_IDS", "")),
                "subs_en_ids": parse_ids(row.get("SUBS_EN_IDS", "")),
                "start_ms": parse_timestamp_to_ms(row["START_TIME"]),
                "end_ms": parse_timestamp_to_ms(row["END_TIME"]),
                "name_audio": row.get("NAME_AUDIO", "").strip(),
                "name_screenshot": row.get("NAME_SCREENSHOT", "").strip(),
                "content_ja": row.get("CONTENT", "").strip(),
                "content_es": row.get("CONTENT_TRANSLATION_SPANISH", "").strip(),
                "content_en": row.get("CONTENT_TRANSLATION_ENGLISH", "").strip(),
                "is_mt_es": parse_bool(row.get("CONTENT_SPANISH_MT", "")),
                "is_mt_en": parse_bool(row.get("CONTENT_ENGLISH_MT", "")),
                "actor_ja": row.get("ACTOR_JA", "").strip() or None,
                "actor_es": row.get("ACTOR_ES", "").strip() or None,
                "actor_en": row.get("ACTOR_EN", "").strip() or None,
            }
            segments.append(segment)
    return segments


# ---------------------------------------------------------------------------
# Subtitle structure builder
# ---------------------------------------------------------------------------


def build_subtitles_dict(old_segment: dict) -> dict:
    """Build v6 subtitles structure from old segment data.

    Since the old format has joined content strings and comma-separated IDs,
    we create a single subtitle entry per language using the first subtitle ID,
    the full content, and segment-level timestamps.
    """
    subs = {}
    for lang, ids_key, content_key, actor_key in [
        ("ja", "subs_jp_ids", "content_ja", "actor_ja"),
        ("es", "subs_es_ids", "content_es", "actor_es"),
        ("en", "subs_en_ids", "content_en", "actor_en"),
    ]:
        ids = old_segment[ids_key]
        content = old_segment[content_key]
        actor = old_segment[actor_key]

        if content and ids:
            subs[lang] = [
                {
                    "id": ids[0],
                    "text": content,
                    "start_ms": old_segment["start_ms"],
                    "end_ms": old_segment["end_ms"],
                    "actor": actor,
                }
            ]
        else:
            subs[lang] = []
    return subs


# ---------------------------------------------------------------------------
# AniList relation helpers
# ---------------------------------------------------------------------------


def find_sequel_anilist_id(anime_data) -> int | None:
    """Scan AniList relations for a SEQUEL and return its ID."""
    if not hasattr(anime_data, "relations") or not anime_data.relations:
        return None
    if not hasattr(anime_data.relations, "edges"):
        return None
    for edge in anime_data.relations.edges:
        rel_type = str(getattr(edge, "relationType", "")) if hasattr(edge, "relationType") else ""
        if rel_type == "SEQUEL":
            node_type = str(getattr(edge.node, "type", "")) if hasattr(edge.node, "type") else ""
            if node_type == "ANIME":
                return edge.node.id
    return None


# ---------------------------------------------------------------------------
# Screenshot generation (960x540 web-optimized)
# ---------------------------------------------------------------------------


def generate_screenshot(input_path: str, output_webp: str) -> bool:
    """Generate a 960x540 web-optimized screenshot from the original."""
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-vf",
                "scale=960:540",
                "-c:v",
                "libwebp",
                "-quality",
                "85",
                "-method",
                "6",
                output_webp,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return result.returncode == 0
    except Exception as err:
        logger.error(f"Error generating screenshot: {err}")
        return False


# ---------------------------------------------------------------------------
# Video generation (screenshot + audio → mp4)
# ---------------------------------------------------------------------------


def generate_video_from_screenshot_and_audio(
    screenshot_path: str, audio_path: str, video_path: str
) -> bool:
    """Generate web-optimized video from a screenshot and audio file."""
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loop",
                "1",
                "-framerate",
                "24",
                "-i",
                screenshot_path,
                "-i",
                audio_path,
                "-vf",
                "scale=1280:720,setsar=1",
                "-c:v",
                "libx264",
                "-profile:v",
                "baseline",
                "-level",
                "3.0",
                "-preset",
                "faster",
                "-tune",
                "fastdecode",
                "-crf",
                "35",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-aac_coder",
                "twoloop",
                "-b:a",
                "96k",
                "-movflags",
                "+faststart",
                "-shortest",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return result.returncode == 0
    except Exception as err:
        logger.error(f"Error generating video: {err}")
        return False


# ---------------------------------------------------------------------------
# Per-episode migration
# ---------------------------------------------------------------------------


def migrate_episode(
    episode_number: int,
    old_episode_path: str,
    output_dir: str,
    anime_data,
    hash_salt: str,
    config: dict,
) -> dict:
    """Migrate a single episode from v5 to v6.

    Returns dict with keys: segments_count, ignored_count, failed_count
    """
    anilist_id = anime_data.id
    tsv_path = os.path.join(old_episode_path, "data.tsv")

    if not os.path.exists(tsv_path):
        logger.warning(f"[E{episode_number}] No data.tsv found at {tsv_path}, skipping")
        return {"segments_count": 0, "ignored_count": 0, "failed_count": 0}

    old_segments = parse_data_tsv(tsv_path)
    console.print(
        f"  [cyan][E{episode_number}] Parsed {len(old_segments)} segments from data.tsv[/cyan]"
    )

    # Create episode output directory
    episode_output_dir = os.path.join(output_dir, str(episode_number))
    os.makedirs(episode_output_dir, exist_ok=True)

    workers = config.get("workers", 4)

    # Process segments in parallel
    results = [None] * len(old_segments)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for idx, old_seg in enumerate(old_segments):
            future = pool.submit(
                _process_single_segment,
                old_seg,
                idx + 1,
                anilist_id,
                episode_number,
                old_episode_path,
                episode_output_dir,
                hash_salt,
                config,
            )
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    # Collect results in order
    segments = []
    ignored_segments = []
    failed_count = 0
    for status, data in results:
        if status == "ok":
            segments.append(data)
        elif status == "ignored":
            ignored_segments.append(data)
        elif status == "failed":
            ignored_segments.append(data)
            failed_count += 1

    # --- Compute duration_ms and write _data.json ---
    if segments or ignored_segments:
        all_ends = [s["end_ms"] for s in segments] + [s["end_ms"] for s in ignored_segments]
        duration_ms = max(all_ends) if all_ends else 0

        write_data_json(
            episode_output_dir,
            segments,
            episode_number,
            duration_ms,
            anime_data,
            ignored_segments,
        )

        ignored_msg = f", {len(ignored_segments)} ignored" if ignored_segments else ""
        failed_msg = f", {failed_count} failed" if failed_count else ""
        console.print(
            f"  [green][E{episode_number}] Created _data.json with "
            f"{len(segments)} segments{ignored_msg}{failed_msg}[/green]"
        )

    return {
        "segments_count": len(segments),
        "ignored_count": len(ignored_segments),
        "failed_count": failed_count,
    }


def _process_single_segment(
    old_seg: dict,
    segment_index: int,
    anilist_id: int,
    episode_number: int,
    old_episode_path: str,
    episode_output_dir: str,
    hash_salt: str,
    config: dict,
) -> tuple[str, dict]:
    """Process a single segment.

    Returns (status, data) where status is 'ok', 'ignored', or 'failed'.
    """
    try:
        subtitles = build_subtitles_dict(old_seg)

        # --- Validation ---
        missing_languages = []
        if not old_seg["content_ja"]:
            missing_languages.append("ja")
        if not old_seg["content_es"]:
            missing_languages.append("es")
        if not old_seg["content_en"]:
            missing_languages.append("en")

        if missing_languages:
            reason = f"missing required languages: {','.join(missing_languages)}"
            return ("ignored", _build_ignored_segment(old_seg, segment_index, reason, subtitles))

        if len(old_seg["content_ja"]) > MAX_SEGMENT_CONTENT_LENGTH:
            reason = (
                f"content too long ({len(old_seg['content_ja'])} > {MAX_SEGMENT_CONTENT_LENGTH})"
            )
            return ("ignored", _build_ignored_segment(old_seg, segment_index, reason, subtitles))

        # --- Hash generation ---
        subs_jp_ids = old_seg["subs_jp_ids"]
        if not subs_jp_ids:
            return ("ignored", _build_ignored_segment(
                old_seg, segment_index, "no japanese subtitle IDs", subtitles
            ))

        segment_hash = generate_segment_hash(
            anilist_id, episode_number, subs_jp_ids[0], subs_jp_ids, hash_salt
        )

        audio_filename = f"{segment_hash}.mp3"
        screenshot_filename = f"{segment_hash}.webp"
        video_filename = f"{segment_hash}.mp4"

        audio_out = os.path.join(episode_output_dir, audio_filename)
        screenshot_out = os.path.join(episode_output_dir, screenshot_filename)
        video_out = os.path.join(episode_output_dir, video_filename)

        seg_dict = _build_segment_dict(
            old_seg, segment_index, segment_hash,
            audio_filename, screenshot_filename, video_filename,
            subtitles,
        )

        # --- Resumability / dry-run ---
        if os.path.exists(audio_out) or config.get("dry_run"):
            return ("ok", seg_dict)

        # --- Check source files ---
        old_audio_path = os.path.join(old_episode_path, old_seg["name_audio"])
        old_screenshot_path = os.path.join(old_episode_path, old_seg["name_screenshot"])

        missing_files = []
        if not os.path.exists(old_audio_path):
            missing_files.append(old_seg["name_audio"])
        if not os.path.exists(old_screenshot_path):
            missing_files.append(old_seg["name_screenshot"])
        if missing_files:
            reason = f"missing files: {', '.join(missing_files)}"
            return ("ignored", _build_ignored_segment(old_seg, segment_index, reason, subtitles))

        # --- Audio: copy as-is (skip re-encoding to avoid lossy-to-lossy quality loss) ---
        shutil.copy2(old_audio_path, audio_out)

        # --- Screenshot: generate 960x540 webp ---
        if not generate_screenshot(old_screenshot_path, screenshot_out):
            return ("failed", _build_ignored_segment(
                old_seg, segment_index, "screenshot generation failed", subtitles
            ))

        # --- Video ---
        if not config.get("skip_video") and not generate_video_from_screenshot_and_audio(
            screenshot_out, audio_out, video_out
        ):
            return ("failed", _build_ignored_segment(
                old_seg, segment_index, "video generation failed", subtitles
            ))

        return ("ok", seg_dict)

    except Exception as err:
        logger.error(f"[E{episode_number}] Segment {segment_index} failed: {err}")
        subtitles = build_subtitles_dict(old_seg)
        return ("failed", _build_ignored_segment(
            old_seg, segment_index, f"exception: {err}", subtitles
        ))


def _build_segment_dict(
    old_seg: dict,
    segment_index: int,
    segment_hash: str,
    audio_filename: str,
    screenshot_filename: str,
    video_filename: str,
    subtitles: dict,
) -> dict:
    return {
        "segment_hash": segment_hash,
        "segment_index": segment_index,
        "start_ms": old_seg["start_ms"],
        "end_ms": old_seg["end_ms"],
        "duration_ms": old_seg["end_ms"] - old_seg["start_ms"],
        "content_ja": old_seg["content_ja"],
        "content_es": old_seg["content_es"],
        "content_en": old_seg["content_en"],
        "is_mt_es": old_seg["is_mt_es"],
        "is_mt_en": old_seg["is_mt_en"],
        "actor_ja": old_seg["actor_ja"],
        "actor_es": old_seg["actor_es"],
        "actor_en": old_seg["actor_en"],
        "files": {
            "audio": audio_filename,
            "screenshot": screenshot_filename,
            "video": video_filename,
        },
        "subtitles": subtitles,
    }


def _build_ignored_segment(
    old_seg: dict, segment_index: int, reason: str, subtitles: dict
) -> dict:
    return {
        "segment_index": segment_index,
        "start_ms": old_seg["start_ms"],
        "end_ms": old_seg["end_ms"],
        "duration_ms": old_seg["end_ms"] - old_seg["start_ms"],
        "content_ja": old_seg["content_ja"],
        "content_es": old_seg["content_es"] or None,
        "content_en": old_seg["content_en"] or None,
        "actor_ja": old_seg["actor_ja"],
        "actor_es": old_seg["actor_es"],
        "actor_en": old_seg["actor_en"],
        "reason": reason,
        "files": None,
        "subtitles": subtitles,
    }


# ---------------------------------------------------------------------------
# Per-season migration
# ---------------------------------------------------------------------------


def migrate_season(
    season_dir: str,
    anilist_id: int,
    output_dir: str,
    config: dict,
) -> dict:
    """Migrate a single season, fetching fresh AniList data.

    Returns dict with keys: anilist_id, episodes, total_segments, total_ignored, total_failed
    """
    anilist = CachedAnilist()

    try:
        anime_data = anilist.get_anime_with_id(anilist_id)
    except Exception as e:
        console.print(f"  [red]Failed to fetch AniList data for ID {anilist_id}: {e}[/red]")
        return {
            "anilist_id": anilist_id,
            "episodes": 0,
            "total_segments": 0,
            "total_ignored": 0,
            "total_failed": 0,
        }

    romaji = getattr(anime_data.title, "romaji", "Unknown")
    console.print(f"  [green]Fetched AniList data: {romaji} (ID: {anilist_id})[/green]")

    # Create output directory: {output_dir}/{anilist_id}/
    season_output_dir = os.path.join(output_dir, str(anilist_id))
    os.makedirs(season_output_dir, exist_ok=True)

    # Save _info.json (also downloads cover/banner and generates hash_salt)
    info_json_path = os.path.join(season_output_dir, "_info.json")
    hash_salt = save_info_json(info_json_path, anime_data, str(anilist_id))

    # Discover episode directories (E01, E02, ...)
    episode_dirs = sorted(
        [d for d in os.listdir(season_dir) if d.startswith("E") and d[1:].isdigit()],
        key=lambda d: int(d[1:]),
    )

    episodes_filter = config.get("episodes")
    total_segments = 0
    total_ignored = 0
    total_failed = 0
    episodes_processed = 0

    for episode_dir_name in episode_dirs:
        episode_number = int(episode_dir_name[1:])

        if episodes_filter and episode_number not in episodes_filter:
            continue

        old_episode_path = os.path.join(season_dir, episode_dir_name)
        stats = migrate_episode(
            episode_number,
            old_episode_path,
            season_output_dir,
            anime_data,
            hash_salt,
            config,
        )
        total_segments += stats["segments_count"]
        total_ignored += stats["ignored_count"]
        total_failed += stats["failed_count"]
        episodes_processed += 1

    return {
        "anilist_id": anilist_id,
        "anime_data": anime_data,
        "episodes": episodes_processed,
        "total_segments": total_segments,
        "total_ignored": total_ignored,
        "total_failed": total_failed,
    }


# ---------------------------------------------------------------------------
# Top-level migration
# ---------------------------------------------------------------------------


def migrate_anime(input_dir: str, output_dir: str, config: dict) -> None:
    """Migrate an entire anime directory from v5 to v6.

    Discovers seasons, interactively prompts for AniList IDs,
    and migrates each season as a separate v6 media entry.
    """
    # Load old info.json
    info_json_path = os.path.join(input_dir, "info.json")
    if not os.path.exists(info_json_path):
        console.print(f"[red]No info.json found at {info_json_path}[/red]")
        return

    with open(info_json_path, encoding="utf-8") as f:
        old_info = json.load(f)

    base_anilist_id = old_info.get("id")
    anime_name = old_info.get("romaji_name") or old_info.get("english_name") or "Unknown"

    console.print(f"\n[bold cyan]Migrating: {anime_name}[/bold cyan]")
    console.print(f"  Base AniList ID: {base_anilist_id}")

    # Discover seasons
    season_dirs = sorted(
        [d for d in os.listdir(input_dir) if d.startswith("S") and d[1:].isdigit()],
        key=lambda d: int(d[1:]),
    )

    if not season_dirs:
        console.print("[red]No season directories found (expected S01, S02, ...)[/red]")
        return

    console.print(f"  Found {len(season_dirs)} season(s): {', '.join(season_dirs)}")

    # Interactively map each season to an AniList ID
    season_id_map: dict[str, int] = {}
    anilist = CachedAnilist()
    prev_anime_data = None

    for season_dir_name in season_dirs:
        season_number = int(season_dir_name[1:])

        if season_number == 1:
            default_id = base_anilist_id
        else:
            # Try to find sequel from previous season's AniList data
            sequel_id = find_sequel_anilist_id(prev_anime_data) if prev_anime_data else None
            default_id = sequel_id or base_anilist_id

        # Prompt user for the AniList ID (or auto-accept default)
        console.print(f"\n  [bold]{season_dir_name}[/bold]:")
        if config.get("auto_id") and default_id:
            chosen_id = default_id
            console.print(f"  Auto-selected AniList ID: {chosen_id}")
        else:
            answer = questionary.text(
                f"  AniList ID for {season_dir_name}?",
                default=str(default_id) if default_id else "",
            ).ask()

            if answer is None:
                console.print("[yellow]Cancelled by user[/yellow]")
                return

            try:
                chosen_id = int(answer.strip())
            except ValueError:
                console.print(
                    f"[red]Invalid ID: {answer}. Skipping {season_dir_name}.[/red]"
                )
                continue

        season_id_map[season_dir_name] = chosen_id

        # Fetch AniList data to use for sequel lookup in next iteration
        try:
            prev_anime_data = anilist.get_anime_with_id(chosen_id)
            romaji = getattr(prev_anime_data.title, "romaji", "Unknown")
            console.print(f"  → {romaji} (ID: {chosen_id})")
        except Exception as e:
            console.print(f"  [yellow]Could not fetch AniList data for {chosen_id}: {e}[/yellow]")
            prev_anime_data = None

    # Confirm before proceeding
    console.print("\n[bold]Migration plan:[/bold]")
    for season_name, aid in season_id_map.items():
        console.print(f"  {season_name} → AniList ID {aid}")

    if config.get("dry_run"):
        console.print("\n[yellow]Dry run mode — showing what would be done[/yellow]")

    if config.get("yes"):
        console.print("\n[green]Auto-confirmed (--yes)[/green]")
    else:
        proceed = questionary.confirm("Proceed with migration?", default=True).ask()
        if not proceed:
            console.print("[yellow]Migration cancelled[/yellow]")
            return

    os.makedirs(output_dir, exist_ok=True)

    # Migrate each season
    all_stats = []
    for season_dir_name, anilist_id in season_id_map.items():
        season_path = os.path.join(input_dir, season_dir_name)
        console.print(
            f"\n[bold cyan]--- {season_dir_name} (AniList ID: {anilist_id}) ---[/bold cyan]"
        )

        stats = migrate_season(season_path, anilist_id, output_dir, config)
        all_stats.append(stats)

    # Print summary
    console.print("\n[bold green]=== Migration Summary ===[/bold green]")
    total_seg = 0
    total_ign = 0
    total_fail = 0
    for stats in all_stats:
        aid = stats["anilist_id"]
        romaji = getattr(stats.get("anime_data", None), "title", None)
        romaji = getattr(romaji, "romaji", "N/A") if romaji else "N/A"
        console.print(
            f"  {romaji} (ID: {aid}): "
            f"{stats['episodes']} episodes, "
            f"{stats['total_segments']} segments, "
            f"{stats['total_ignored']} ignored, "
            f"{stats['total_failed']} failed"
        )
        total_seg += stats["total_segments"]
        total_ign += stats["total_ignored"]
        total_fail += stats["total_failed"]

    console.print(
        f"\n  [bold]Total: {total_seg} segments, "
        f"{total_ign} ignored, {total_fail} failed[/bold]"
    )

    # --- Verification: compare origin segment counts vs migration results ---
    console.print("\n[bold cyan]=== Verification ===[/bold cyan]")
    all_ok = True
    for season_dir_name, anilist_id in season_id_map.items():
        season_path = os.path.join(input_dir, season_dir_name)
        episode_dirs = sorted(
            [d for d in os.listdir(season_path) if d.startswith("E") and d[1:].isdigit()],
            key=lambda d: int(d[1:]),
        )
        episodes_filter = config.get("episodes")

        for episode_dir_name in episode_dirs:
            episode_number = int(episode_dir_name[1:])
            if episodes_filter and episode_number not in episodes_filter:
                continue

            tsv_path = os.path.join(season_path, episode_dir_name, "data.tsv")
            if not os.path.exists(tsv_path):
                continue

            origin_count = len(parse_data_tsv(tsv_path))

            # Read _data.json from output
            data_json_path = os.path.join(
                output_dir, str(anilist_id), str(episode_number), "_data.json"
            )
            if not os.path.exists(data_json_path):
                console.print(
                    f"  [red]MISSING: {anilist_id}/E{episode_number} — "
                    f"no _data.json (origin had {origin_count} segments)[/red]"
                )
                all_ok = False
                continue

            with open(data_json_path, encoding="utf-8") as f:
                data = json.load(f)

            migrated = len(data.get("segments", []))
            ignored = len(data.get("ignored_segments", []))
            accounted = migrated + ignored

            if accounted != origin_count:
                console.print(
                    f"  [red]MISMATCH: {anilist_id}/E{episode_number} — "
                    f"origin={origin_count}, migrated={migrated}, "
                    f"ignored={ignored}, accounted={accounted}[/red]"
                )
                all_ok = False
            else:
                console.print(
                    f"  [green]OK: {anilist_id}/E{episode_number} — "
                    f"{origin_count} origin → {migrated} migrated + {ignored} ignored[/green]"
                )

    if all_ok:
        console.print("[bold green]All episodes verified successfully![/bold green]")
    else:
        console.print("[bold red]Some episodes have mismatches — review above.[/bold red]")

    console.print("[bold green]=== Migration Complete ===[/bold green]\n")
