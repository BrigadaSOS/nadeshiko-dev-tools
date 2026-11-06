import json
import os

import pytest

from nadeshiko_dev_tools.audiobook_segment_extractor import extractor as ab
from nadeshiko_dev_tools.audiobook_segment_extractor.extractor import (
    BookInputError,
    Chapter,
    Cue,
)


def _manifest(cues, audio_files=1, kind="text-quote"):
    return {
        "schema": "honomiya.read-listen.v1",
        "sources": {
            "ebook": {"sha256": "0" * 64},
            "audioFiles": [{"index": i, "sha256": "1" * 64} for i in range(audio_files)],
        },
        "cues": [
            {
                "id": f"s-{i}",
                "text": {"kind": kind, "sectionRef": "p.xhtml", "exact": text}
                if kind == "text-quote"
                else {"kind": kind, "sectionRef": "p.xhtml", "fragmentId": "f"},
                "audioFileIndex": 0,
                "startMs": start,
                "endMs": end,
            }
            for i, (start, end, text) in enumerate(cues)
        ],
    }


class FakeTranslator:
    """Echoes the input tagged with the target language; '' for texts in `blank`."""

    def __init__(self, blank=()):
        self.blank = set(blank)
        self.calls = []

    def translate_text(self, texts, source_lang=None, target_lang=None):
        self.calls.append((source_lang, target_lang, list(texts)))

        class R:
            def __init__(self, text):
                self.text = text

        return [R("" if t in self.blank else f"{target_lang}:{t}") for t in texts]


@pytest.fixture
def fake_media(monkeypatch):
    """Replace ffmpeg calls with files that just record their arguments."""

    def fake_extract_audio(src, out, start, end, audio_index=None):
        with open(out, "w") as f:
            f.write(f"{start}-{end}")

    def fake_build_clip(image, audio, out):
        with open(out, "w") as f:
            f.write("clip")

    monkeypatch.setattr(ab, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(ab, "build_clip", fake_build_clip)


# ── load_cues ─────────────────────────────────────────────────────────────────


def test_load_cues_sorts_and_strips():
    cues = ab.load_cues(_manifest([(5000, 6000, " 二。 "), (1000, 2000, "一。")]))
    assert cues == [Cue(1000, 2000, "一。"), Cue(5000, 6000, "二。")]


def test_load_cues_rejects_other_schema():
    manifest = _manifest([(0, 1000, "a")])
    manifest["schema"] = "something.else"
    with pytest.raises(BookInputError, match="schema"):
        ab.load_cues(manifest)


def test_load_cues_rejects_multi_track():
    with pytest.raises(BookInputError, match="one audio file"):
        ab.load_cues(_manifest([(0, 1000, "a")], audio_files=2))


def test_load_cues_rejects_fragment_anchors():
    with pytest.raises(BookInputError, match="text-quote"):
        ab.load_cues(_manifest([(0, 1000, "a")], kind="fragment"))


# ── chapters ──────────────────────────────────────────────────────────────────


def test_parse_chapters_numbers_from_one():
    chapters = ab.parse_chapters(
        {
            "chapters": [
                {"start_time": "0.000000", "end_time": "206.192", "tags": {"title": "タイトル"}},
                {"start_time": "206.192", "end_time": "3305.213", "tags": {}},
            ]
        }
    )
    assert chapters == [
        Chapter(1, "タイトル", 0, 206192),
        Chapter(2, "Chapter 2", 206192, 3305213),
    ]


def test_assign_chapters_by_start():
    chapters = [Chapter(1, "a", 0, 10_000), Chapter(2, "b", 10_000, 20_000)]
    cues = [Cue(500, 1500, "x"), Cue(9_500, 10_500, "y"), Cue(10_000, 11_000, "z")]
    grouped = ab.assign_chapters(cues, chapters)
    assert [c.text for c in grouped[1]] == ["x", "y"]  # y starts in 1, crosses into 2
    assert [c.text for c in grouped[2]] == ["z"]


def test_assign_chapters_cue_past_last_chapter_goes_to_last():
    chapters = [Chapter(1, "a", 0, 10_000), Chapter(2, "b", 10_000, 20_000)]
    grouped = ab.assign_chapters([Cue(25_000, 26_000, "late")], chapters)
    assert [c.text for c in grouped[2]] == ["late"]


def test_assign_chapters_without_chapters_is_one_chapter():
    assert ab.assign_chapters([Cue(0, 1, "x")], []) == {1: [Cue(0, 1, "x")]}


# ── filtering ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("cue", "reason"),
    [
        (Cue(0, 3000, "ok"), None),
        (Cue(0, 30_803, "渡航"), "too long (30.8 s)"),
        (Cue(0, 300, "あ"), "too short (0.30 s)"),
        (Cue(0, 3000, ""), "empty text"),
    ],
)
def test_rejection_reason(cue, reason):
    assert ab.rejection_reason(cue, 500, 12_000) == reason


def test_segment_hash_is_stable_and_salted():
    a = ab.segment_hash("salt", "B07PPMBZHX", 214000, "文。")
    assert a == ab.segment_hash("salt", "B07PPMBZHX", 214000, "文。")
    assert a != ab.segment_hash("other", "B07PPMBZHX", 214000, "文。")
    assert len(a) == 10


# ── process_chapter ───────────────────────────────────────────────────────────


def _run_chapter(tmp_path, cues, translator=None):
    frame = tmp_path / "frame.webp"
    frame.write_bytes(b"webp")
    chapter = Chapter(2, "(1)とにかく比企谷八幡はくさっている。", 206_000, 3_305_000)
    count = ab.process_chapter(
        chapter,
        cues,
        audio_path="audio.m4b",
        frame_path=str(frame),
        out_folder=str(tmp_path),
        asin="B07PPMBZHX",
        hash_salt="salt",
        translator=translator or FakeTranslator(),
    )
    with open(tmp_path / "2" / "_data.json", encoding="utf-8") as f:
        return count, json.load(f)


def test_process_chapter_writes_data_json(tmp_path, fake_media):
    cues = [
        Cue(
            214_000, 220_260, "国語教師の平塚静は額に青筋を立てながら、俺の作文を大声で読み上げた。"
        ),
        Cue(222_060, 222_200, "っ"),
        Cue(239_120, 242_580, "さては、この未熟な文章について呼び出されたのか。"),
    ]
    count, data = _run_chapter(tmp_path, cues)

    assert count == 2
    assert data["metadata"] == {
        "version": "6",
        "number": 2,
        "title": "(1)とにかく比企谷八幡はくさっている。",
        "duration_ms": 3_099_000,
        "total_segments": 2,
    }
    assert data["media"] == {"asin": "B07PPMBZHX", "media_source": "audible"}

    seg = data["segments"][0]
    assert seg["segment_index"] == 0
    assert (seg["start_ms"], seg["end_ms"], seg["duration_ms"]) == (8_000, 14_260, 6_260)
    assert seg["content_en"].startswith("EN-US:")
    assert seg["content_es"].startswith("ES:")
    assert seg["is_mt_en"] is seg["is_mt_es"] is True
    h = seg["segment_hash"]
    assert seg["files"] == {"audio": f"{h}.mp3", "screenshot": f"{h}.webp", "video": f"{h}.mp4"}
    for name in seg["files"].values():
        assert os.path.exists(tmp_path / "2" / name)
    # Audio is cut from the absolute position in the m4b, not the chapter-relative one.
    assert (tmp_path / "2" / f"{h}.mp3").read_text() == "214.0-220.26"

    assert data["segments"][1]["segment_index"] == 2
    assert data["ignored_segments"] == [
        {
            "segment_index": 1,
            "start_ms": 16_060,
            "end_ms": 16_200,
            "duration_ms": 140,
            "content_ja": "っ",
            "content_en": None,
            "content_es": None,
            "reason": "too short (0.14 s)",
        }
    ]


def test_process_chapter_ignores_missing_translation(tmp_path, fake_media):
    cues = [Cue(214_000, 216_000, "一。"), Cue(217_000, 219_000, "二。")]
    count, data = _run_chapter(tmp_path, cues, FakeTranslator(blank={"二。"}))
    assert count == 1
    assert data["ignored_segments"][0]["reason"] == "missing translation: en,es"


def test_process_chapter_translates_only_kept_cues_in_one_batch(tmp_path, fake_media):
    translator = FakeTranslator()
    _run_chapter(tmp_path, [Cue(214_000, 216_000, "一。"), Cue(0, 100_000, "長")], translator)
    assert translator.calls == [("JA", "EN-US", ["一。"]), ("JA", "ES", ["一。"])]


def test_process_chapter_resumes_existing(tmp_path, fake_media):
    cues = [Cue(214_000, 216_000, "一。")]
    _run_chapter(tmp_path, cues)
    translator = FakeTranslator()
    count, _ = _run_chapter(tmp_path, cues, translator)
    assert count == 1
    assert translator.calls == []


# ── book input ────────────────────────────────────────────────────────────────


def test_load_book_requires_fields(tmp_path):
    (tmp_path / "book.json").write_text(json.dumps({"asin": "B0"}), encoding="utf-8")
    with pytest.raises(BookInputError, match="japanese_name"):
        ab.load_book(str(tmp_path))


def test_write_info_json_keeps_salt(tmp_path):
    book = {
        "asin": "B07PPMBZHX",
        "japanese_name": "やはり俺の青春ラブコメはまちがっている。1",
        "romaji_name": "Yahari Ore no Seishun Love Comedy wa Machigatteiru. 1",
        "english_name": "My Youth Romantic Comedy Is Wrong, As I Expected 1",
        "release_date": "2019-04-12",
    }
    cover = tmp_path / "in" / "cover.jpg"
    cover.parent.mkdir()
    cover.write_bytes(b"jpg")
    out = tmp_path / "out"
    out.mkdir()

    salt = ab.write_info_json(str(out), book, str(cover))
    assert ab.write_info_json(str(out), book, str(cover)) == salt
    info = json.loads((out / "_info.json").read_text(encoding="utf-8"))
    assert info["category"] == "AUDIOBOOK"
    assert info["media_source"] == "audible"
    assert info["id"] == info["asin"] == "B07PPMBZHX"
    assert info["start_date"] == "2019-04-12"
    assert (out / "cover.jpg").read_bytes() == b"jpg"


def test_parse_chapter_selection():
    from nadeshiko_dev_tools.audiobook_segment_extractor.cli import parse_chapter_selection

    assert parse_chapter_selection("2-4, 7") == {2, 3, 4, 7}
    assert parse_chapter_selection("5") == {5}


def test_load_book_defaults_content_rating_to_safe(tmp_path):
    book = {
        "asin": "B0",
        "japanese_name": "本",
        "romaji_name": "Hon",
        "english_name": "Book",
        "release_date": "2020-01-01",
    }
    (tmp_path / "book.json").write_text(json.dumps(book), encoding="utf-8")
    assert ab.load_book(str(tmp_path))["content_rating"] == "SAFE"

    (tmp_path / "book.json").write_text(
        json.dumps({**book, "content_rating": "explicit-ish"}), encoding="utf-8"
    )
    with pytest.raises(BookInputError, match="content_rating"):
        ab.load_book(str(tmp_path))


def test_segments_carry_the_rating_and_an_empty_analysis(tmp_path, fake_media):
    _, data = _run_chapter(tmp_path, [Cue(214_000, 216_000, "一。")])
    seg = data["segments"][0]
    # The uploader refuses segments without both; no classifier runs on the cover.
    assert seg["content_rating"] == "SAFE"
    assert seg["content_analysis"] == {"scores": {}, "tags": {}}
