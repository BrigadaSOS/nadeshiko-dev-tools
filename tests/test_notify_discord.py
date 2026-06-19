import json
from types import SimpleNamespace

from nadeshiko_dev_tools import notify_discord


class FakeClient:
    def __init__(self, media):
        self._media = media

    def iter_list_media(self):
        return iter(self._media)


def _media(public_id="pub123", youtube="UCabc", anilist=None):
    return SimpleNamespace(
        public_id=public_id,
        external_ids=SimpleNamespace(youtube=youtube, anilist=anilist),
        episode_count=10,
        segment_count=2291,
        title=SimpleNamespace(
            english="Tokyo Trivia",
            romaji="Tokyo Trivia",
            native="東京限定雑学",
        ),
        cover_image_url="https://example.test/cover.webp",
        banner_image_url="https://example.test/banner.webp",
    )


def test_lookup_public_id_supports_youtube_external_id(monkeypatch):
    monkeypatch.setenv("NADESHIKO_PROD_API_KEY", "token")
    monkeypatch.setattr(notify_discord, "build_api_client", lambda target: FakeClient([_media()]))

    public_id = notify_discord.get_nadeshiko_public_id("UCabc", "prod", source="youtube")

    assert public_id == "pub123"


def test_youtube_notification_payload_uses_video_wording_and_prod_url():
    payload = notify_discord.build_webhook_payload(
        public_id="pub123",
        source="youtube",
        english_title="Tokyo Trivia",
        native_title="東京限定雑学",
        romaji_title="Tokyo Trivia",
        episodes=10,
        segments=2291,
        hours=1.9,
        cover_url="https://example.test/cover.webp",
        target="prod",
    )

    embed = payload["embeds"][0]
    assert embed["title"] == "New YouTube content on Nadeshiko!"
    assert embed["url"] == "https://nadeshiko.co/search?media=pub123"
    assert "**Videos:** 10" in embed["description"]
    assert "**Sentences:** 2,291" in embed["description"]
    assert "Generated subtitles" not in embed["description"]
    assert "Episodes" not in embed["description"]


def test_load_local_notification_info_reads_youtube_channel_name(tmp_path):
    channel = tmp_path / "UCabc"
    channel.mkdir()
    (channel / "_info.json").write_text(
        json.dumps(
            {
                "name": "東京限定雑学 / Tokyo Trivia",
                "channel_id": "UCabc",
                "cover": "cover.webp",
            }
        )
    )

    info = notify_discord.get_local_notification_info(str(channel), "pub123")

    assert info.english_title == "東京限定雑学 / Tokyo Trivia"
    assert info.romaji_title == "東京限定雑学 / Tokyo Trivia"
    assert info.cover_url == "https://cdn.nadeshiko.co/media/yt/UCabc/cover.webp"
    assert info.public_id == "pub123"


def test_compute_stats_local_reports_generated_languages_for_youtube(tmp_path):
    channel = tmp_path / "UCabc"
    video = channel / "video1"
    video.mkdir(parents=True)
    (video / "_data.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"duration_ms": 2000, "is_mt_es": True, "is_mt_en": False},
                    {"duration_ms": 3000, "is_mt_es": True, "is_mt_en": False},
                ]
            }
        )
    )

    stats = notify_discord.compute_stats_local(str(channel))

    assert stats.episodes == 1
    assert stats.segments == 2
    assert stats.hours == 5 / 3_600
    assert stats.generated_langs == ["es"]
