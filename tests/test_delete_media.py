from types import SimpleNamespace

import pytest
from nadeshiko_internal import NadeshikoError

from nadeshiko_dev_tools import delete_media
from nadeshiko_dev_tools.delete_media import (
    MediaLookupError,
    find_media,
    parse_args,
    r2_cleanup_refusal,
    remove_from_api,
    storage_path_for,
    validate_storage_path,
)


def _media(public_id="pub1", **ext):
    ids = dict.fromkeys(("anilist", "tmdb", "tvdb", "imdb", "youtube", "asin"))
    ids.update(ext)
    return SimpleNamespace(
        public_id=public_id, name_romaji="Title", external_ids=SimpleNamespace(**ids)
    )


class FakeClient:
    def __init__(self, media=(), list_error=None, episodes=(), episode_error=None):
        self._media = list(media)
        self._list_error = list_error
        self._episodes = list(episodes)
        self._episode_error = episode_error
        self.deleted = []

    def iter_list_media(self):
        yield from self._media
        if self._list_error:
            raise self._list_error

    def iter_list_episodes(self, media_public_id):
        if self._episode_error:
            raise self._episode_error
        return iter(self._episodes)

    def delete_episode(self, media_public_id, episode_number):
        self.deleted.append(("episode", episode_number))

    def delete_media(self, media_public_id):
        self.deleted.append(("media", media_public_id))


def _api_error():
    err = NadeshikoError.__new__(NadeshikoError)
    err.detail = "boom"
    return err


@pytest.mark.parametrize(
    "source, ext_id, expected",
    [
        ("anilist", "21804", "media/21804"),
        ("tmdb", "46198_1", "media/mv/46198_1"),
        ("youtube", "UCabc", "media/yt/UCabc"),
        ("asin", "B07PPMBZHX", "media/ab/B07PPMBZHX"),
        ("tvdb", "123", None),
        ("imdb", "tt1", None),
    ],
)
def test_storage_path_for_mirrors_uploader(source, ext_id, expected):
    assert storage_path_for(source, ext_id) == expected


@pytest.mark.parametrize(
    "path", ["media/21804", "media/mv/46198_1", "media/yt/UCabc", "/media/ab/B07PPMBZHX/"]
)
def test_validate_storage_path_accepts_one_media_folder(path):
    assert validate_storage_path(path) == path.strip("/")


@pytest.mark.parametrize(
    "path",
    [
        None,
        "",
        "media",
        "media/",
        "media/mv",
        "media/yt",
        "media/ab/",
        "media/zz/1",
        "media/..",
        "media/mv/..",
        "media/a/b/c",
        "other/1",
        "media/x y",
        "media/abc",
    ],
)
def test_validate_storage_path_refuses_anything_broader(path):
    with pytest.raises(ValueError):
        validate_storage_path(path)


def test_r2_only_touched_for_prod():
    assert r2_cleanup_refusal("prod") is None
    assert r2_cleanup_refusal("dev")
    assert r2_cleanup_refusal("local")


def test_parse_args_rules():
    assert parse_args(["21804", "--target", "dev"]).media_id == "21804"
    with pytest.raises(SystemExit):
        parse_args(["--target", "prod", "--r2-only"])  # no --r2-path
    with pytest.raises(SystemExit):
        parse_args(["--target", "prod"])  # no media id
    with pytest.raises(SystemExit):
        parse_args(["1", "--target", "prod", "--r2-only", "--api-only", "--r2-path", "media/1"])
    args = parse_args(["--target", "prod", "--r2-only", "--r2-path", "media/1"])
    assert args.media_id is None


def test_find_media_reports_matched_source():
    client = FakeClient([_media("a", anilist="1"), _media("b", tmdb="1_2")])
    media, source = find_media(client, "1_2")
    assert (media.public_id, source) == ("b", "tmdb")
    assert find_media(client, "999") == (None, None)


def test_find_media_api_error_is_not_not_found():
    client = FakeClient([_media("a", anilist="1")], list_error=_api_error())
    with pytest.raises(MediaLookupError):
        find_media(client, "999")


def test_remove_from_api_aborts_on_lookup_error():
    client = FakeClient(list_error=_api_error())
    assert remove_from_api(client, "1", dry_run=False) == (False, None)
    assert client.deleted == []


def test_remove_from_api_aborts_when_episode_listing_fails():
    client = FakeClient([_media("a", anilist="1")], episode_error=_api_error())
    ok, _ = remove_from_api(client, "1", dry_run=False)
    assert ok is False
    assert client.deleted == []


def test_remove_from_api_returns_derived_path():
    episodes = [SimpleNamespace(episode_number=1, segment_count=3)]
    client = FakeClient([_media("a", youtube="UCabc")], episodes=episodes)
    assert remove_from_api(client, "UCabc", dry_run=False) == (True, "media/yt/UCabc")
    assert client.deleted == [("episode", 1), ("media", "a")]


def _run_main(monkeypatch, argv, media):
    calls = []
    monkeypatch.setattr(delete_media, "get_api_client", lambda target: FakeClient(media))
    monkeypatch.setattr(
        delete_media, "remove_from_r2", lambda path, dry_run, skip_confirm=False: calls.append(path)
    )
    return delete_media.main(argv), calls


def test_main_dev_never_touches_r2(monkeypatch):
    rc, calls = _run_main(monkeypatch, ["1", "--target", "dev", "-y"], [_media(anilist="1")])
    assert rc == 0 and calls == []


def test_main_prod_cleans_derived_folder(monkeypatch):
    rc, calls = _run_main(monkeypatch, ["1", "--target", "prod", "-y"], [_media(anilist="1")])
    assert rc == 0 and calls == ["media/1"]


def test_main_prod_not_found_leaves_r2(monkeypatch):
    rc, calls = _run_main(monkeypatch, ["1", "--target", "prod", "-y"], [])
    assert rc == 0 and calls == []


def test_main_prod_tvdb_match_leaves_r2(monkeypatch):
    rc, calls = _run_main(monkeypatch, ["7", "--target", "prod", "-y"], [_media(tvdb="7")])
    assert rc == 0 and calls == []
