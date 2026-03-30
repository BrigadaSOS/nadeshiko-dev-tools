from rich.console import Console

console = Console()


def display_file_details(all_file_details: list) -> None:
    """Display file details in a formatted table, grouped by folder and episode."""
    console.print("\n[green bold]Files to process:[/green bold]")

    # Group files by (folder, episode) - keep seasons separate
    from collections import defaultdict

    grouped = defaultdict(list)
    for details in all_file_details:
        key = (details["folder_name"], details["episode"])
        grouped[key].append(details)

    # Sort by folder name, then episode number
    for (folder_name, episode_num), files in sorted(grouped.items()):
        if len(files) == 1:
            # Single file - display as before
            details = files[0]
            audio_info = ""
            if details["audio_count"] > 0:
                langs = ", ".join(details["audio_langs"])
                audio_info = f"[dim]|[/dim] Audio: {details['audio_count']} ({langs})"
            else:
                audio_info = "[dim]|[/dim] [yellow]No audio[/yellow]"

            subtitle_info = ""
            if details["subtitle_count"] > 0:
                langs = ", ".join(details["subtitle_langs"])
                subtitle_info = f"[dim]|[/dim] Subtitles: {details['subtitle_count']} ({langs})"
            else:
                subtitle_info = "[dim]|[/dim] [yellow]No subtitles[/yellow]"

            console.print(
                f"  [cyan]E{episode_num:02d}[/cyan] - {details['folder_name']} "
                f"[dim]|[/dim] {details['duration']}"
                f"{audio_info}"
                f"{subtitle_info}"
            )
        else:
            # Multiple files for same episode in same folder - group them
            total_audio = sum(f["audio_count"] for f in files)
            all_audio_langs = set()
            for f in files:
                all_audio_langs.update(f["audio_langs"])

            total_subs = sum(f["subtitle_count"] for f in files)
            all_sub_langs = set()
            for f in files:
                all_sub_langs.update(f["subtitle_langs"])

            audio_info = ""
            if total_audio > 0:
                langs = ", ".join(sorted(all_audio_langs))
                audio_info = f"[dim]|[/dim] Audio: {total_audio} ({langs})"
            else:
                audio_info = "[dim]|[/dim] [yellow]No audio[/yellow]"

            subtitle_info = ""
            if total_subs > 0:
                langs = ", ".join(sorted(all_sub_langs))
                subtitle_info = f"[dim]|[/dim] Subtitles: {total_subs} ({langs})"
            else:
                subtitle_info = "[dim]|[/dim] [yellow]No subtitles[/yellow]"

            # Use duration from first file
            duration = files[0]["duration"]

            console.print(
                f"  [cyan]E{episode_num:02d}[/cyan] - {folder_name} "
                f"([cyan]{len(files)} files[/cyan]) "
                f"[dim]|[/dim] {duration}"
                f"{audio_info}"
                f"{subtitle_info}"
            )


def display_folder_mappings(folder_mappings: dict) -> None:
    """Display mapped anime information."""
    console.print("\n[green bold]Mapped Anime Information:[/green bold]")
    for folder_name, mapping in folder_mappings.items():
        anime = mapping["anime"]
        console.print(f"  [cyan]{folder_name}[/cyan] -> {anime.title.romaji}")
        english_title = getattr(anime.title, "english", None)
        if english_title:
            console.print(f"    English: {english_title}")
        native_title = getattr(anime.title, "native", None)
        if native_title:
            console.print(f"    Native: {native_title}")
        console.print(f"    Format: {anime.format}, Status: {anime.status}")
        if anime.episodes:
            console.print(f"    Episodes: {anime.episodes}")
        console.print(f"    Files: {len(mapping['files'])}")
        console.print("")
