"""AudiobookSource: how process-audiobook output becomes a media in the API."""

from nadeshiko_dev_tools.assets_uploader.uploader import (
    AudiobookSource,
    MediaInfo,
    get_media_source,
)


def _info(start_date="2019-04-12", genres=None):
    return MediaInfo(
        anilist_id="B07PPMBZHX",
        japanese_name="やはり俺の青春ラブコメはまちがっている。1",
        english_name="My Youth Romantic Comedy Is Wrong, As I Expected 1",
        romaji_name="Yahari Ore no Seishun Love Comedy wa Machigatteiru. 1",
        airing_format="AUDIOBOOK",
        airing_status="FINISHED",
        genres=genres or [],
        version="6",
        hash_salt="salt",
        cover_url=None,
        banner_url=None,
        start_date=start_date,
        end_date=None,
        media_source="audible",
        category="AUDIOBOOK",
    )


def test_audible_info_uses_the_audiobook_source():
    assert isinstance(get_media_source("audible"), AudiobookSource)
    assert isinstance(_info().source, AudiobookSource)


def test_identity_and_storage_are_keyed_by_asin():
    info = _info()
    source = info.source
    assert source.key == "asin"
    assert source.external_ids(info).to_dict() == {
        "anilist": None,
        "imdb": None,
        "tvdb": None,
        "tmdb": None,
        "youtube": None,
        "asin": "B07PPMBZHX",
    }
    assert source.storage_base_path(info) == "media/ab/B07PPMBZHX"


def test_descriptors_mark_it_as_an_audiobook_without_a_season():
    assert _info(genres=["Comedy"]).source.create_descriptors(_info(genres=["Comedy"])) == {
        "airing_format": "AUDIOBOOK",
        "airing_status": "FINISHED",
        "genres": ["Comedy"],
        "category": "AUDIOBOOK",
        "season_name": "NONE",
        "season_year": 2019,
    }


def test_chapter_folders_are_episode_numbers():
    from pathlib import Path

    source = AudiobookSource()
    assert source.episode_number(Path("out/B07PPMBZHX/2"), surrogate=1) == 2
    assert source.episode_number(Path("out/B07PPMBZHX/_nsfw_results"), surrogate=1) is None
