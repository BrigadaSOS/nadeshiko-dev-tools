import json
from types import SimpleNamespace

from nadeshiko_dev_tools import notify_discord as n


def test_audiobook_payload():
    p = n.build_webhook_payload(
        "_VKM-T05zdhD",
        "audiobook",
        "Tomozaki",
        "弱キャラ友崎くん",
        "Tomozaki",
        9,
        7024,
        9.2,
        "https://cdn.example/cover.webp",
    )
    e = p["embeds"][0]
    assert e["title"] == "New audiobook on Nadeshiko!"
    assert "**Chapters:** 9" in e["description"]
    assert "**Sentences:** 7,024" in e["description"]
    assert "**Duration:** 9.2 hours" in e["description"]
    assert p["allowed_mentions"] == {"parse": []}


def test_audiobook_asin_and_sdk_names(monkeypatch):
    m = SimpleNamespace(
        public_id="pub",
        external_ids=SimpleNamespace(asin="B08GLT57JB"),
        name_en="Tomozaki",
        name_ja="友崎",
        name_romaji="Tomozaki Lv1",
        cover_url="https://cdn.example/cover.webp",
    )
    monkeypatch.setattr(
        n, "build_api_client", lambda t: SimpleNamespace(iter_list_media=lambda: iter([m]))
    )
    assert n.get_nadeshiko_public_id("B08GLT57JB", "prod", "audiobook") == "pub"
    info = n.get_api_notification_info("B08GLT57JB", "prod", "audiobook")
    assert info is not None
    assert (
        info.english_title == "Tomozaki"
        and info.native_title == "友崎"
        and info.romaji_title == "Tomozaki Lv1"
    )
    assert n.parse_args(["--source", "audiobook", "B08GLT57JB"]).source == "audiobook"


def test_audiobook_duration_is_chapter_duration(tmp_path):
    p = tmp_path / "1"
    p.mkdir()
    (p / "_data.json").write_text(
        json.dumps(
            {
                "media": {"asin": "B08GLT57JB"},
                "metadata": {"duration_ms": 3600000},
                "segments": [{"duration_ms": 1500, "is_mt_en": True}],
            }
        )
    )
    s = n.compute_stats_local(str(tmp_path))
    assert s.hours == 1 and s.segments == 1 and s.generated_langs == ["en"]


def test_receipt_readback_and_no_duplicates(tmp_path, monkeypatch):
    payload = n.build_webhook_payload("pub", "audiobook", "Book", None, None, 9, 7024, 9.2, None)
    msg = {"id": "123", "channel_id": "456", "guild_id": "789", "embeds": payload["embeds"]}
    calls = []

    def post(url, **kw):
        calls.append(kw)
        return SimpleNamespace(status_code=200, json=lambda: msg)

    monkeypatch.setattr(n.requests, "post", post)
    monkeypatch.setattr(
        n.requests, "get", lambda *a, **kw: SimpleNamespace(status_code=200, json=lambda: msg)
    )
    path = tmp_path / "receipt.json"
    assert n.send_webhook_payload("https://discord.example/webhook", payload, str(path))
    receipt = json.loads(path.read_text())
    assert (
        receipt["read_back_verified"]
        and receipt["message_url"] == "https://discord.com/channels/789/456/123"
    )
    assert calls[0]["params"] == {"wait": "true"}
    assert not n.send_webhook_payload("https://discord.example/webhook", payload, str(path))
    assert len(calls) == 1


def test_failed_readback_preserves_receipt(tmp_path, monkeypatch):
    msg = {"id": "123", "channel_id": "456", "embeds": []}
    monkeypatch.setattr(
        n.requests, "post", lambda *a, **kw: SimpleNamespace(status_code=200, json=lambda: msg)
    )
    monkeypatch.setattr(n.requests, "get", lambda *a, **kw: SimpleNamespace(status_code=500))
    path = tmp_path / "receipt.json"
    assert not n.send_webhook_payload("https://discord.example/webhook", {"embeds": []}, str(path))
    assert json.loads(path.read_text())["message_id"] == "123"
