import json
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest

from nadeshiko_dev_tools.common.config import ProcessingConfig
from nadeshiko_dev_tools.segment_extractor.splitter import split_video_by_subtitles

from .conftest import read_subtitles_from_folders


@dataclass
class MockAnimeTitle:
    romaji: str = "Test Anime"
    english: str | None = "Test Anime"
    native: str | None = "テストアニメ"


@dataclass
class MockAnimeData:
    id: int = 12345
    title: MockAnimeTitle = None

    def __post_init__(self):
        if self.title is None:
            self.title = MockAnimeTitle()


def segments_to_tsv(segments):
    header = (
        "ID\tSUBS_JP_IDS\tSUBS_ES_IDS\tSUBS_EN_IDS\tSTART_TIME\tEND_TIME\t"
        "NAME_AUDIO\tNAME_SCREENSHOT\tCONTENT\tCONTENT_TRANSLATION_SPANISH\t"
        "CONTENT_TRANSLATION_ENGLISH\tCONTENT_SPANISH_MT\tCONTENT_ENGLISH_MT\t"
        "ACTOR_JA\tACTOR_ES\tACTOR_EN"
    )
    lines = [header]

    for seg in segments:
        subs_jp_ids = ",".join(str(s["id"]) for s in seg.get("subtitles", {}).get("ja", []))
        subs_es_ids = ",".join(str(s["id"]) for s in seg.get("subtitles", {}).get("es", []))
        subs_en_ids = ",".join(str(s["id"]) for s in seg.get("subtitles", {}).get("en", []))

        start_time = str(timedelta(milliseconds=seg["start_ms"]))
        end_time = str(timedelta(milliseconds=seg["end_ms"]))

        files = seg.get("files") or {}
        audio = files.get("audio", "")
        screenshot = files.get("screenshot", "")

        row = [
            seg.get("id", ""),
            subs_jp_ids,
            subs_es_ids,
            subs_en_ids,
            start_time,
            end_time,
            audio,
            screenshot,
            seg.get("content_ja", "") or "",
            seg.get("content_es", "") or "",
            seg.get("content_en", "") or "",
            str(seg.get("is_mt_es", False)),
            str(seg.get("is_mt_en", False)),
            seg.get("actor_ja", "") or "",
            seg.get("actor_es", "") or "",
            seg.get("actor_en", "") or "",
        ]
        lines.append("\t".join(row))

    return "\n".join(lines)


@pytest.mark.parametrize("matching_subtitles", read_subtitles_from_folders("tests/input/"))
def test_subtitles_snapshots(snapshot, matching_subtitles):
    sample_subtitles_filepath = matching_subtitles["ja"].filepath

    snapshot.snapshot_dir = "tests/snapshots"
    tmp_output_folder = "tests/snapshots/tmp"
    filename = os.path.basename(sample_subtitles_filepath).split(".")[0]

    config = ProcessingConfig(
        input_folder=Path("."),
        output_folder=Path("."),
    )
    if "adachi-to-shimamura" in filename:
        config = ProcessingConfig(
            input_folder=Path("."),
            output_folder=Path("."),
            extra_punctuation=True,
        )

    anime_data = MockAnimeData()

    split_video_by_subtitles(
        translator=None,
        video_file=None,
        subtitles=matching_subtitles,
        episode_folder_output_path=tmp_output_folder,
        config=config,
        anime_data=anime_data,
        episode_number=1,
        duration_ms=1440000,
    )

    json_path = os.path.join(tmp_output_folder, "_data.json")
    with open(json_path) as f:
        data = json.load(f)

    tsv_content = segments_to_tsv(data.get("segments", []))

    snapshot_filename = f"{filename}.snapshot.tsv"
    snapshot.assert_match(tsv_content, snapshot_filename)
