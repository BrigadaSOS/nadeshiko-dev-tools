import os

from guessit import guessit

from nadeshiko_dev_tools.segment_extractor.splitter import MatchingSubtitle
from nadeshiko_dev_tools.segment_extractor.utils.subtitle_utils import load_subtitle_file


def read_input_subtitles(anime_folder_path):
    print(anime_folder_path)
    subtitles_filepaths = sorted(
        [
            os.path.join(anime_folder_path, filename)
            for filename in os.listdir(anime_folder_path)
            if filename.endswith(".ass") or filename.endswith(".srt")
        ]
    )

    current_season = 0
    current_episode = 0
    matching_subtitles = {}
    for subtitle_path in subtitles_filepaths:
        subtitle_info = guessit(subtitle_path)
        season_number = subtitle_info["season"]
        episode_number = subtitle_info["episode"]
        subtitle_language = subtitle_info["subtitle_language"].alpha2

        if season_number == current_season and episode_number == current_episode:
            matching_subtitles[subtitle_language] = MatchingSubtitle(
                origin="external",
                filepath=subtitle_path,
                data=load_subtitle_file(subtitle_path),
            )
            continue

        if matching_subtitles:
            yield matching_subtitles

        matching_subtitles = {
            subtitle_language: MatchingSubtitle(
                origin="external",
                filepath=subtitle_path,
                data=load_subtitle_file(subtitle_path),
            )
        }
        current_season = season_number
        current_episode = episode_number

    if matching_subtitles:
        yield matching_subtitles


def read_subtitles_from_folders(input_folder):
    subtitles = []
    for name in os.listdir(input_folder):
        dirpath = os.path.join(input_folder, name)
        if os.path.isdir(dirpath):
            subtitles += list(read_input_subtitles(dirpath))

    return subtitles
