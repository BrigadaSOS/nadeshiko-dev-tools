import json
from types import SimpleNamespace

import pytest

from nadeshiko_dev_tools.audiobook_segment_extractor import extractor as ab
from nadeshiko_dev_tools.audiobook_segment_extractor.book_translator import (
    BookTranslator,
    Glossary,
    problems,
)

GLOSSARY = Glossary(
    names={"由比ヶ浜": "Yuigahama", "雪ノ下": "Yukinoshita"}, notes=["雪ノ下: female"]
)


class FakeClient:
    """An OpenAI-shaped client; `answer(lines)` returns the {index: {src, en, es}} dict."""

    def __init__(self, answer):
        self.answer = answer
        self.calls: list[list[str]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, response_format, temperature, **_):
        lines = list(json.loads(messages[1]["content"]).values())
        self.calls.append(lines)
        out = self.answer(lines)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(out)))],
            usage=usage,
        )


def tag(ja):
    """A stable, Japanese-free stand-in translation for one line."""
    return str(sum(map(ord, ja)))


def names(ja):
    """A correct translation carries the glossary romaji of every name in the line."""
    return "".join(f" {romaji}" for name, romaji in GLOSSARY.names.items() if name in ja)


def en(ja):
    return f"EN {tag(ja)}{names(ja)}"


def es(ja):
    return f"ES {tag(ja)}{names(ja)}"


def good(lines):
    """A model that answers every line correctly, echoing each start."""
    return {str(i): {"src": ja[:5], "en": en(ja), "es": es(ja)} for i, ja in enumerate(lines)}


def translator(answer, glossary=GLOSSARY):
    return BookTranslator(FakeClient(answer), model="openai/gpt-oss-120b", glossary=glossary)


# ── Detector ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ja, text, expected",
    [
        ("おい、早くしろ", '"Hey, hurry up.""}', "debris"),
        ("「ようこそ」", "Welcome. // Note: the input had keys 0-19", "debris"),
        ("修羅場でな", "a real修罗场", "japanese-left"),
        ("「お前……」", 'You"""""""""""""""""', "repeat"),
        ("「あれ？", "«¿Eh?», «Es obvio.» «Bueno, supongo…»", "merged"),
        ("雪ノ下と由比ヶ浜がいた。", "Yukinoshita and Yui were there.", "name:Yuigahama"),
        ("", "", "empty"),
    ],
)
def test_problems_catch_what_qwen_got_wrong(ja, text, expected):
    assert expected in problems(ja, text, GLOSSARY)


@pytest.mark.parametrize(
    "ja, text",
    [
        ("雪ノ下と由比ヶ浜がいた。", "Yukinoshita and Yuigahama were there."),
        ("「………………る」", "“……ru”"),
        ("あとがき", "Afterword"),
    ],
)
def test_problems_leave_good_translations_alone(ja, text):
    assert problems(ja, text, GLOSSARY) == []


def test_glossary_reads_book_json():
    g = Glossary.from_book({"glossary": {"names": {"比企谷": "Hikigaya"}, "notes": ["n1"]}})
    assert g.names == {"比企谷": "Hikigaya"} and g.notes == ["n1"]
    assert "比企谷 = Hikigaya" in g.prompt()
    assert Glossary.from_book({}).prompt() == ""


# ── Translation, echo check and repair ────────────────────────────────────────


def test_translates_en_and_es_in_one_call_per_batch():
    t = translator(good)
    out = t.translate_book(["一つ目の文。", "二つ目の文。"])
    assert out == [
        (en("一つ目の文。"), es("一つ目の文。")),
        (en("二つ目の文。"), es("二つ目の文。")),
    ]
    assert len(t._client.calls) == 1
    assert t.usage["requests"] == 1


def test_glossary_reaches_the_prompt():
    seen = {}

    def spy(model, messages, **kw):
        seen["system"] = messages[0]["content"]
        return FakeClient(good)._create(model, messages, **kw)

    t = translator(good)
    t._client.chat.completions.create = spy
    t.translate_book(["由比ヶ浜が来た。"])
    assert "由比ヶ浜 = Yuigahama" in seen["system"]


def test_a_shifted_answer_is_rejected_and_repaired_with_context():
    """The first answer gives line 1 the translation of line 2; the echo exposes it."""
    calls = {"n": 0}

    def shifted_then_good(lines):
        calls["n"] += 1
        if calls["n"] == 1:
            out = good(lines)
            out["1"] = {"src": lines[2][:5], "en": "EN of the next line", "es": "ES siguiente"}
            return out
        return good(lines)

    t = translator(shifted_then_good)
    lines = ["最初の文です。", "二番目の文です。", "三番目の文です。"]
    out = t.translate_book(lines)
    assert out[1] == (en("二番目の文です。"), es("二番目の文です。"))
    assert t.review == []


def test_flagged_lines_are_retranslated_and_the_better_answer_kept():
    calls = {"n": 0}

    def yui_then_full_name(lines):
        calls["n"] += 1
        out = good(lines)
        for i, ja in enumerate(lines):
            if "由比ヶ浜" in ja:
                name = "Yui" if calls["n"] == 1 else "Yuigahama"
                out[str(i)].update(en=f"{name} came.", es=f"Vino {name}.")
        return out

    t = translator(yui_then_full_name)
    out = t.translate_book(["由比ヶ浜が来た。"])
    assert out == [("Yuigahama came.", "Vino Yuigahama.")]


def test_what_cannot_be_fixed_is_kept_and_reported():
    def always_yui(lines):
        out = good(lines)
        for i, ja in enumerate(lines):
            if "由比ヶ浜" in ja:
                out[str(i)].update(en="Yui came.", es="Vino Yui.")
        return out

    t = translator(always_yui)
    out = t.translate_book(["由比ヶ浜が来た。"])
    assert out == [("Yui came.", "Vino Yui.")]
    assert t.review and "name:Yuigahama" in t.review[0]["problems"]


def test_two_lines_with_the_same_translation_are_both_rechecked():
    calls = {"n": 0}

    def duplicate_then_good(lines):
        calls["n"] += 1
        out = good(lines)
        if calls["n"] == 1:
            out["1"] = dict(out["1"], en=out["0"]["en"])
        return out

    t = translator(duplicate_then_good)
    out = t.translate_book(["一行目の台詞です。", "二行目の台詞です。"])
    assert out[1][0] == en("二行目の台詞です。")


def test_a_failed_request_leaves_none_for_the_pipeline_to_skip(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)

    def broken(lines):
        raise RuntimeError("down")

    t = translator(broken)
    assert t.translate_book(["文。"]) == [None]


# ── Wiring into the extractor ─────────────────────────────────────────────────


def test_translate_all_uses_the_book_translator_and_fills_the_cache():
    t = translator(good)
    cache = ab.TranslationCache()
    out = ab.translate_all(t, ["一。", "二。", "一。"], cache)
    assert out["en"] == [en("一。"), en("二。"), en("一。")]
    assert cache.get("二。", "es") == es("二。")
    # Cached sentences are not sent again.
    ab.translate_all(t, ["一。", "二。"], cache)
    assert len(t._client.calls) == 1


def test_repair_chapter_translations_rewrites_only_the_text(tmp_path):
    chapter = tmp_path / "8"
    chapter.mkdir()
    segments = [
        {
            "segment_hash": "a",
            "content_ja": "由比ヶ浜が来た。",
            "content_en": "Yui came.",
            "content_es": "Vino Yui.",
        },
        {
            "segment_hash": "b",
            "content_ja": "雨だ。",
            "content_en": en("雨だ。"),
            "content_es": es("雨だ。"),
        },
    ]
    (chapter / "_data.json").write_text(json.dumps({"segments": segments}), encoding="utf-8")
    cache = ab.TranslationCache()

    changed = ab.repair_chapter_translations(str(chapter), translator(good), cache)

    data = json.loads((chapter / "_data.json").read_text(encoding="utf-8"))
    assert changed == 1
    assert data["segments"][0]["content_en"] == en("由比ヶ浜が来た。")
    assert data["segments"][0]["segment_hash"] == "a"
    assert data["segments"][1]["content_en"] == en("雨だ。")
    assert cache.get("由比ヶ浜が来た。", "es") == es("由比ヶ浜が来た。")
