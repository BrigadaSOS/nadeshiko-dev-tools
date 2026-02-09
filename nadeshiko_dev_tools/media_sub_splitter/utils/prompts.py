import contextlib
import logging
import os
import signal
import subprocess
import sys

import questionary
from rich.console import Console

from nadeshiko_dev_tools.media_sub_splitter.utils.text_utils import extract_anime_title_for_guessit

console = Console()
logger = logging.getLogger(__name__)


def _get_config_key(folder_name: str, folder_path: str) -> str:
    """Create a unique config key from folder name and parent folder.

    Args:
        folder_name: The folder name (e.g., "Season 01")
        folder_path: The full path to the folder

    Returns:
        A unique key like "Bakuman/Season 01"
    """
    parent_name = os.path.basename(os.path.dirname(folder_path))
    return f"{parent_name}/{folder_name}"


def restore_terminal(signum=None, frame=None):
    """Restore terminal to sane state after interruption."""
    try:
        import questionary

        questionary.fixups.finalizer.fix()
    except Exception:
        pass
    with contextlib.suppress(FileNotFoundError):
        subprocess.run(["stty", "sane"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def setup_signal_handlers():
    """For graceful terminal restoration."""
    signal.signal(signal.SIGINT, lambda s, f: (restore_terminal(s, f), sys.exit(130)))
    with contextlib.suppress(Exception):
        signal.signal(signal.SIGTSTP, restore_terminal)  # Ctrl+Z


def map_folder_to_anilist(folder_info: dict, anilist_client) -> dict:
    folder_name = folder_info["name"]
    folder_path = folder_info["path"]
    console.print(f"\n[cyan]Processing folder: {folder_name}[/cyan]")
    console.print(f"  Files: {folder_info['file_count']}")

    choice = questionary.select(
        "How do you want to find the Anilist media?",
        choices=[
            questionary.Choice("Search by folder name", value="search"),
            questionary.Choice("Manual text input", value="manual"),
            questionary.Choice("Enter Anilist ID directly", value="id"),
            questionary.Choice("Skip this folder", value="skip"),
        ],
    ).ask()

    if choice == "skip":
        console.print(f"[dim]Skipping {folder_name}[/dim]")
        return None

    if choice == "id":
        anilist_id = questionary.text("Enter Anilist ID:").ask()
        if anilist_id and anilist_id.isdigit():
            try:
                anime_data = anilist_client.get_anime_with_id(int(anilist_id))
                console.print(f"[green]Found: {anime_data.title.romaji}[/green]")
                return anime_data
            except Exception as e:
                console.print(f"[red]Error fetching Anilist ID {anilist_id}: {e}[/red]")
                return None
        return None

    if choice == "manual":
        manual_query = questionary.text("Enter search term:").ask()
        if not manual_query:
            console.print("[dim]Skipping folder.[/dim]")
            return None
        return _search_and_select(anilist_client, manual_query)

    if choice == "search":
        # Use parent folder name for search
        parent_folder = os.path.basename(os.path.dirname(folder_path))
        search_query = extract_anime_title_for_guessit(parent_folder)
        console.print(f"[dim]Searching for: {search_query}[/dim]")

        search_results = None
        try:
            search_results = anilist_client.search(search_query)
        except Exception as e:
            console.print(f"[red]Search failed: {e}[/red]")

        if not search_results:
            console.print("[yellow]No results found.[/yellow]")
            return _retry_search(anilist_client)

        return _select_from_results(anilist_client, search_results)

    return None


def _search_and_select(anilist_client, search_query: str):
    """Search Anilist and let user select from results."""
    console.print(f"[dim]Searching for: {search_query}[/dim]")

    search_results = None
    try:
        search_results = anilist_client.search(search_query)
    except Exception as e:
        console.print(f"[red]Search failed: {e}[/red]")

    if not search_results:
        console.print("[yellow]No results found.[/yellow]")
        return _retry_search(anilist_client)

    return _select_from_results(anilist_client, search_results)


def _retry_search(anilist_client):
    """Prompt user to try another search term."""
    alt_query = questionary.text("Try another search term (or press Enter to skip):").ask()
    if alt_query:
        return _search_and_select(anilist_client, alt_query)
    console.print("[dim]Skipping folder.[/dim]")
    return None


def _select_from_results(anilist_client, search_results):
    """Let user select from Anilist search results."""
    choices = [
        questionary.Choice(
            f"{r.title.romaji} - {getattr(r.title, 'english', '') or ''} (ID: {r.id})",
            value=r.id,
        )
        for r in search_results[:10]  # Limit to 10 results
    ] + [questionary.Choice("Skip", value=None)]

    selected_id = questionary.select("Select matching media:", choices=choices).ask()
    if selected_id is None:
        return None

    # Fetch full anime data
    try:
        anime_data = anilist_client.get_anime_with_id(selected_id)
        console.print(f"[green]Found: {anime_data.title.romaji}[/green]")
        return anime_data
    except Exception as e:
        console.print(f"[red]Error fetching anime data: {e}[/red]")
        return None


def select_audio_tracks(folder_mappings: dict, all_file_details: list, audio_config: dict) -> dict:
    """Prompt user to select audio tracks for folders that need configuration."""
    needs_audio_config = False
    for folder_name, mapping in folder_mappings.items():
        config_key = _get_config_key(folder_name, mapping["path"])
        if config_key not in audio_config:
            first_file_details = next(
                (d for d in all_file_details if d["folder_name"] == folder_name), None
            )
            if first_file_details and first_file_details["audio_count"] > 1:
                needs_audio_config = True
                break

    if not needs_audio_config:
        return audio_config

    console.print("\n[cyan]Audio Track Configuration[/cyan]")
    for folder_name, mapping in folder_mappings.items():
        config_key = _get_config_key(folder_name, mapping["path"])
        if config_key in audio_config:
            continue

        folder_files = [d for d in all_file_details if d["folder_name"] == folder_name]
        if not folder_files or folder_files[0]["audio_count"] <= 1:
            continue

        console.print(f"\n[yellow]{folder_name}[/yellow]")
        audio_streams = folder_files[0]["audio_streams"]

        # Try to auto-select Japanese audio
        japanese_stream = None
        for stream in audio_streams:
            lang = stream.get("tags", {}).get("language", "").lower()
            if lang in ("jpn", "ja", "japanese"):
                japanese_stream = stream
                break

        if japanese_stream:
            lang = japanese_stream.get("tags", {}).get("language", "und")
            audio_config[config_key] = {
                "index": japanese_stream["index"],
                "language": lang,
            }
            console.print(
                f"  [green]✓[/green] Auto-selected Japanese audio: {lang} "
                f"(index {japanese_stream['index']})"
            )
        else:
            choices = []
            for s in audio_streams:
                lang = s.get("tags", {}).get("language", "und")
                title = s.get("tags", {}).get("title", "")
                title_suffix = f" - {title}" if title else ""
                label = f"{lang}{title_suffix} [dim](index: {s['index']})[/dim]"
                choices.append(questionary.Choice(label, value=s["index"]))

            selected_index = questionary.select(
                "Select audio track:",
                choices=choices,
            ).ask()

            if selected_index is not None:
                selected_stream = next(
                    (s for s in audio_streams if s["index"] == selected_index),
                    None,
                )
                if selected_stream:
                    lang = selected_stream.get("tags", {}).get("language", "und")
                    audio_config[config_key] = {
                        "index": selected_index,
                        "language": lang,
                    }
                    console.print(f"  [green]✓[/green] Saved: {lang} (index {selected_index})")

    return audio_config


def select_subtitle_tracks(
    folder_mappings: dict,
    all_file_details: list,
    subtitle_config: dict,
) -> dict:
    """Prompt user to select subtitle tracks for folders that need configuration."""
    import ffmpeg

    needs_sub_config = False
    for folder_name, mapping in folder_mappings.items():
        config_key = _get_config_key(folder_name, mapping["path"])
        if config_key not in subtitle_config:
            folder_files = [d for d in all_file_details if d["folder_name"] == folder_name]
            if folder_files and folder_files[0]["subtitle_count"] > 0:
                needs_sub_config = True
                break

    if not needs_sub_config:
        return subtitle_config

    console.print("\n[cyan]Subtitle Track Configuration[/cyan]")
    for folder_name, mapping in folder_mappings.items():
        config_key = _get_config_key(folder_name, mapping["path"])
        if config_key in subtitle_config:
            continue

        folder_files = [d for d in all_file_details if d["folder_name"] == folder_name]
        if not folder_files or folder_files[0]["subtitle_count"] == 0:
            continue

        console.print(f"\n[yellow]{folder_name}[/yellow]")
        sample_filepath = folder_files[0]["filepath"]
        try:
            probe = ffmpeg.probe(sample_filepath)
            subtitle_streams = [s for s in probe["streams"] if s["codec_type"] == "subtitle"]

            if subtitle_streams:
                choices = []
                for s in subtitle_streams:
                    lang = s.get("tags", {}).get("language", "und")
                    title = s.get("tags", {}).get("title", "")
                    title_suffix = f" - {title}" if title else ""
                    label = f"{lang}{title_suffix} [dim](index: {s['index']})[/dim]"
                    choices.append(questionary.Choice(label, value=s["index"]))

                selected_index = questionary.select(
                    "Select reference subtitle track:",
                    choices=choices,
                ).ask()

                if selected_index is not None:
                    selected_stream = next(
                        (s for s in subtitle_streams if s["index"] == selected_index),
                        None,
                    )
                    if selected_stream:
                        lang = selected_stream.get("tags", {}).get("language", "und")
                        subtitle_config[config_key] = {
                            "index": selected_index,
                            "language": lang,
                            "title": selected_stream.get("tags", {}).get("title", ""),
                        }
                        console.print(f"  [green]✓[/green] Saved: {lang} (index {selected_index})")
        except Exception as e:
            logger.warning(f"Could not probe {sample_filepath}: {e}")

    return subtitle_config


def select_subtitle_streams(
    subtitles_dict: dict,
    folder_name: str,
    subtitles_dict_remembered: dict,
) -> tuple:
    """Prompt user to select which subtitle streams to use for an episode.

    Returns:
        tuple: (selected_indices, updated subtitles_dict_remembered, sync_external_subs)
    """
    # Generate subtitle signature for comparison (list of title+language, sorted)
    current_signature = sorted([f"{d['title']}|{d['language']}" for d in subtitles_dict.values()])

    # Check if we have a remembered selection for this folder
    folder_remembered = subtitles_dict_remembered.get(folder_name, {})
    remembered_signature = folder_remembered.get("signature", [])
    remembered_titles = folder_remembered.get("selected_titles", [])
    remembered_sync = folder_remembered.get("sync_external_subs", True)

    # Determine if signatures are similar enough to skip asking
    signatures_match = current_signature == remembered_signature

    if signatures_match and remembered_titles:
        # Find indices matching the remembered titles
        selected_indices = [
            index
            for index, details in subtitles_dict.items()
            if f"{details['title']}|{details['language']}" in remembered_titles
        ]
        logger.info(f"Using remembered subtitle selection for folder '{folder_name}'")
        return selected_indices, subtitles_dict_remembered, remembered_sync

    if folder_remembered and not signatures_match:
        logger.info("Subtitle options differ from previous episode. Asking again...")

    # Build choices with all selected by default
    subtitle_choices = [
        questionary.Choice(
            f"{details['title']} ({details['language']})",
            value=index,
            checked=True,
        )
        for index, details in subtitles_dict.items()
    ]

    selected_indices = questionary.checkbox(
        "What subtitles do you want to use?",
        choices=subtitle_choices,
        instruction="(↑↓ to move, Space to select)",
    ).ask()
    if selected_indices is None:
        selected_indices = []

    # Prompt for sync preference
    sync_external_subs = questionary.confirm(
        "Sync external subtitles with internal track?",
        default=True,
    ).ask()

    # Prompt to remember selection for this folder
    remember = questionary.confirm(
        "Save selection for remaining files in this folder?",
        default=True,
    ).ask()
    if remember:
        selected_titles = [
            f"{subtitles_dict[idx]['title']}|{subtitles_dict[idx]['language']}"
            for idx in selected_indices
        ]
        subtitles_dict_remembered[folder_name] = {
            "signature": current_signature,
            "selected_titles": selected_titles,
            "sync_external_subs": sync_external_subs,
        }

    return selected_indices, subtitles_dict_remembered, sync_external_subs


def confirm_processing(total_episodes: int, folder_count: int) -> bool:
    """Prompt user to confirm processing."""
    console.print(
        f"[cyan]Ready to process {total_episodes} episode(s) across {folder_count} folder(s)[/cyan]"
    )
    confirmed = questionary.confirm(
        "Do you want to continue?",
        default=True,
    ).ask()

    if not confirmed:
        console.print("[yellow]Cancelled by user.[/yellow]")

    return confirmed


def _generate_mkv_signature(matching_mkv_sources: list) -> str:
    """Generate a signature for multi-MKV selection.

    The signature is based on the filenames and their audio/subtitle stream counts.
    This allows detecting when the available files have changed.

    Args:
        matching_mkv_sources: List of MatchingMkvSource objects

    Returns:
        A string signature
    """
    import hashlib

    parts = []
    for source in sorted(matching_mkv_sources, key=lambda x: x.filepath):
        filename = os.path.basename(source.filepath)
        audio_count = len(source.audio_streams)
        sub_count = len(source.subtitle_streams)
        parts.append(f"{filename}:{audio_count}:{sub_count}")

    return hashlib.md5("|".join(parts).encode()).hexdigest()


def _validate_remembered_selection(remembered: dict, matching_mkv_sources: list) -> bool:
    """Validate that a remembered selection still applies.

    Simplified: assumes all files in a folder have the same structure,
    so we only check that the main_mkv_index is within bounds.
    """
    # Check that the main_mkv index is valid
    main_mkv_index = remembered.get("main_mkv_index")
    return main_mkv_index is not None and main_mkv_index < len(matching_mkv_sources)


def select_mkv_sources_and_tracks(
    matching_mkv_sources: list,
    folder_name: str,
    remembered_selections: dict,
    folder_audio_config: dict,
    folder_path: str,
) -> tuple:
    """Select main .mkv, audio track, and subtitle sources from multiple .mkv files.

    Args:
        matching_mkv_sources: List of MatchingMkvSource objects
        folder_name: Current folder name for remembering selections
        remembered_selections: Previously saved selections
        folder_audio_config: Audio track config from folder level (may not apply to selected MKV)
        folder_path: Path to the folder (for config key generation)

    Returns:
        tuple: (main_mkv_filepath, audio_idx, subtitle_selections, remembered_selections)

    where subtitle_selections is:
    {
        "/path/to/file1.mkv": [stream_index_1, stream_index_2, ...],
        "/path/to/file2.mkv": [stream_index_3, ...],
    }
    """
    config_key = _get_config_key(folder_name, folder_path)

    # If only one .mkv file, use it with existing behavior
    if len(matching_mkv_sources) == 1:
        source = matching_mkv_sources[0]
        # Try to use folder audio_config if available
        audio_index = None
        if folder_audio_config and config_key in folder_audio_config:
            audio_index = folder_audio_config[config_key].get("index")
            # Verify the audio track exists in this file
            if audio_index is not None:
                audio_stream = next(
                    (s for s in source.audio_streams if s["index"] == audio_index), None
                )
                if audio_stream is None:
                    audio_index = None

        # Auto-select audio if not configured
        if audio_index is None:
            japanese_stream = None
            for stream in source.audio_streams:
                lang = stream.get("tags", {}).get("language", "").lower()
                if lang in ("jpn", "ja", "japanese"):
                    japanese_stream = stream
                    break

            if japanese_stream:
                audio_index = japanese_stream["index"]
            elif source.audio_streams:
                audio_index = source.audio_streams[0]["index"]
            else:
                audio_index = None

        # Use existing select_subtitle_streams for single file
        # Build subtitles_dict for the single file
        subtitles_dict = {}
        for stream in source.subtitle_streams:
            index = stream["index"]
            title = stream.get("tags", {}).get("title")
            language = stream.get("tags", {}).get("language")
            title = title if title else language
            if title and language:
                subtitles_dict[index] = {"title": title, "language": language}

        return (
            source.filepath,
            audio_index,
            {source.filepath: list(subtitles_dict.keys())},
            remembered_selections,
        )

    # Check for remembered selection (by folder, not by signature)
    # Use folder_name so selections persist across episodes in the same folder
    folder_remembered = remembered_selections.get(folder_name, {})
    remembered = folder_remembered.get("multi_mkv")

    if not remembered:
        # Check config file for persisted selections
        from nadeshiko_dev_tools.common.config import get_multi_mkv_selection

        remembered = get_multi_mkv_selection(folder_name)
        if remembered:
            # Add to in-memory for this session
            if folder_name not in remembered_selections:
                remembered_selections[folder_name] = {}
            remembered_selections[folder_name]["multi_mkv"] = remembered

    if remembered and _validate_remembered_selection(remembered, matching_mkv_sources):
        logger.info(f"Using remembered multi-MKV selection for '{folder_name}'")
        # Apply remembered selection using indices
        main_mkv_index = remembered["main_mkv_index"]
        main_source = matching_mkv_sources[main_mkv_index]
        main_mkv = main_source.filepath
        audio_index = remembered["audio_index"]

        # Build subtitle_selections by applying remembered pattern to current files
        subtitle_selections = {}
        for i, source in enumerate(matching_mkv_sources):
            if i in remembered.get("subtitle_file_indices", []):
                subtitle_selections[source.filepath] = [
                    s["index"] for s in source.subtitle_streams
                ]

        return main_mkv, audio_index, subtitle_selections, remembered_selections
    else:
        logger.info("No valid remembered multi-MKV selection. Asking...")

    # Build choices for main .mkv selection
    console.print("\n[cyan]Multiple .mkv files found for this episode[/cyan]")
    choices = []
    for source in matching_mkv_sources:
        filename = os.path.basename(source.filepath)
        audio_info = ", ".join(
            [s.get("tags", {}).get("language", "und") for s in source.audio_streams]
        )
        if not audio_info:
            audio_info = "none"
        sub_info = f"{len(source.subtitle_streams)} tracks"
        label = f"{filename}\n  [dim]Audio: {audio_info} | Subtitles: {sub_info}[/dim]"
        choices.append(questionary.Choice(label, value=source.filepath))

    main_mkv = questionary.select(
        "Select MAIN .mkv file (for audio and video):",
        choices=choices,
    ).ask()

    if main_mkv is None:
        # User cancelled, return defaults
        source = matching_mkv_sources[0]
        audio_index = source.audio_streams[0]["index"] if source.audio_streams else None
        return source.filepath, audio_index, {source.filepath: []}, remembered_selections

    # Select audio track from the chosen main MKV
    main_source = next((s for s in matching_mkv_sources if s.filepath == main_mkv), None)
    if not main_source:
        logger.error(f"Could not find main source in matching files: {main_mkv}")
        main_source = matching_mkv_sources[0]

    # Try to auto-select Japanese audio
    japanese_stream = None
    for stream in main_source.audio_streams:
        lang = stream.get("tags", {}).get("language", "").lower()
        if lang in ("jpn", "ja", "japanese"):
            japanese_stream = stream
            break

    if japanese_stream:
        audio_index = japanese_stream["index"]
        lang = japanese_stream.get("tags", {}).get("language", "und")
        console.print(
            f"  [green]✓[/green] Auto-selected Japanese audio: {lang} (index {audio_index})"
        )
    elif len(main_source.audio_streams) == 1:
        # Only one audio track, use it
        audio_index = main_source.audio_streams[0]["index"]
        lang = main_source.audio_streams[0].get("tags", {}).get("language", "und")
        console.print(f"  [green]✓[/green] Using only audio track: {lang} (index {audio_index})")
    elif main_source.audio_streams:
        # Multiple audio tracks and no Japanese found - prompt user
        console.print(f"\n[yellow]Select audio track from {os.path.basename(main_mkv)}:[/yellow]")
        choices = []
        for stream in main_source.audio_streams:
            lang = stream.get("tags", {}).get("language", "und")
            title = stream.get("tags", {}).get("title", "")
            title_suffix = f" - {title}" if title else ""
            label = f"{lang}{title_suffix} [dim](index: {stream['index']})[/dim]"
            choices.append(questionary.Choice(label, value=stream["index"]))

        audio_index = questionary.select(
            "Select audio track:",
            choices=choices,
        ).ask()

        if audio_index is None:
            audio_index = main_source.audio_streams[0]["index"]

        selected_stream = next(
            (s for s in main_source.audio_streams if s["index"] == audio_index), None
        )
        if selected_stream:
            lang = selected_stream.get("tags", {}).get("language", "und")
            console.print(f"  [green]✓[/green] Selected: {lang} (index {audio_index})")
    else:
        audio_index = None
        console.print("[yellow]Warning: No audio tracks found in main MKV[/yellow]")

    # Build subtitle source selection
    subtitle_sources = []
    for source in matching_mkv_sources:
        if source.filepath == main_mkv:
            # Always include main file's subtitles
            subtitle_sources.append((source, True))
        else:
            # Ask about other files
            use_subs = questionary.confirm(
                f"Extract subtitles from {os.path.basename(source.filepath)}?",
                default=True,
            ).ask()
            if use_subs:
                subtitle_sources.append((source, True))

    # For each selected source, ask which subtitle tracks
    subtitle_selections = {}
    for source, _ in subtitle_sources:
        if source.subtitle_streams:
            choices = []
            for stream in source.subtitle_streams:
                lang = stream.get("tags", {}).get("language", "und")
                title = stream.get("tags", {}).get("title", "")
                label = f"{lang} - {title} [dim](index: {stream['index']})[/dim]"
                # Check all tracks by default for all files
                checked = True
                choices.append(questionary.Choice(label, value=stream["index"], checked=checked))

            selected = questionary.checkbox(
                f"Select subtitle tracks from {os.path.basename(source.filepath)}:",
                choices=choices,
                instruction="(↑↓ to move, Space to select)",
            ).ask()
            subtitle_selections[source.filepath] = selected if selected else []

    # Ensure main file has at least its subtitle tracks selected if user skipped
    if main_mkv not in subtitle_selections or not subtitle_selections[main_mkv]:
        main_source = next(s for s in matching_mkv_sources if s.filepath == main_mkv)
        subtitle_selections[main_mkv] = [s["index"] for s in main_source.subtitle_streams]

    # Ask to remember
    remember = questionary.confirm(
        "Save selection for remaining files in this folder?",
        default=True,
    ).ask()

    if remember:
        # Find the index of the selected main MKV
        main_mkv_index = next(
            i for i, s in enumerate(matching_mkv_sources) if s.filepath == main_mkv
        )
        # Find indices of files selected for subtitle extraction
        subtitle_file_indices = [
            i
            for i, s in enumerate(matching_mkv_sources)
            if s.filepath in subtitle_selections
        ]

        # Store by folder_name so it applies to all episodes in this folder
        if folder_name not in remembered_selections:
            remembered_selections[folder_name] = {}
        remembered_selections[folder_name]["multi_mkv"] = {
            "main_mkv_index": main_mkv_index,
            "audio_index": audio_index,
            "subtitle_file_indices": subtitle_file_indices,
        }
        # Also persist to config file for cross-run persistence
        from nadeshiko_dev_tools.common.config import save_multi_mkv_selection

        save_multi_mkv_selection(
            folder_name, main_mkv_index, audio_index, subtitle_file_indices
        )
        logger.info(f"Saved multi-MKV selection for folder '{folder_name}'")

    return main_mkv, audio_index, subtitle_selections, remembered_selections
