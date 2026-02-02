import os
from datetime import timedelta

import ffmpeg

from nadeshiko_dev_tools.media_sub_splitter.utils.text_utils import extract_anime_title_for_guessit

from .display_utils import console


def probe_files(folder_mappings: dict) -> list:
    """Probe all files in folder mappings to get stream information."""
    all_file_details = []

    for folder_name, mapping in folder_mappings.items():
        console.print(f"[cyan]Scanning files in {folder_name}...[/cyan]")
        folder_path = mapping["path"]

        for filename in mapping["files"]:
            filepath = os.path.join(folder_path, filename)
            try:
                probe = ffmpeg.probe(filepath)
                duration = float(probe["format"]["duration"])
                duration_str = str(timedelta(seconds=int(duration)))

                # Get episode info from guessit
                guessit_query = extract_anime_title_for_guessit(filepath)
                from guessit import guessit

                episode_info = guessit(guessit_query)
                episode = episode_info.get("episode", 1)

                # Count audio streams
                audio_streams = [s for s in probe["streams"] if s["codec_type"] == "audio"]
                audio_count = len(audio_streams)
                audio_langs = []
                for stream in audio_streams:
                    lang = stream.get("tags", {}).get("language", "und")
                    audio_title = stream.get("tags", {}).get("title", "")
                    audio_langs.append(f"{lang}" + (f" ({audio_title})" if audio_title else ""))

                # Count subtitle streams
                subtitle_streams = [s for s in probe["streams"] if s["codec_type"] == "subtitle"]
                subtitle_count = len(subtitle_streams)
                subtitle_langs = []
                for stream in subtitle_streams:
                    lang = stream.get("tags", {}).get("language", "und")
                    subtitle_title = stream.get("tags", {}).get("title", "")
                    subtitle_langs.append(
                        f"{lang}" + (f" ({subtitle_title})" if subtitle_title else "")
                    )

                all_file_details.append(
                    {
                        "folder_name": folder_name,
                        "filepath": filepath,
                        "episode": episode,
                        "duration": duration_str,
                        "audio_count": audio_count,
                        "audio_langs": audio_langs,
                        "audio_streams": audio_streams,
                        "subtitle_count": subtitle_count,
                        "subtitle_langs": subtitle_langs,
                        "subtitle_streams": subtitle_streams,
                        "probe": probe,
                    }
                )
            except Exception as e:
                console.print(f"[yellow]Could not probe {filepath}: {e}[/yellow]")

    return all_file_details
