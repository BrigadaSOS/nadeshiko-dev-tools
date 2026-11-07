import json
from types import SimpleNamespace

import httpx
import openai

from nadeshiko_dev_tools.common.llm_translator import OpenAITranslator


def test_sentence_starts_chunks_large_cue_sets():
    """Large cue sets must not create a single oversized structured response."""
    translator = OpenAITranslator.__new__(OpenAITranslator)
    windows = []

    def complete(_system, user, **_kwargs):
        windows.append(len(json.loads(user)["cues"]))
        return json.dumps({"sentence_starts": [0]})

    translator._complete = complete

    starts = translator.sentence_starts(["あ" for _ in range(306)])

    assert windows == [80, 80, 80, 80, 26]
    # A window's forced first cue must not become a cut; only the real index 0 does.
    assert starts == [0]


def test_sentence_starts_keeps_cuts_inside_each_region():
    """Cuts are taken once, in global indices, from the window that owns them."""
    translator = OpenAITranslator.__new__(OpenAITranslator)
    replies = iter([[0, 30, 75], [0, 5, 12, 40]])  # window 2 covers cues 70..149

    translator._complete = lambda *_a, **_k: json.dumps({"sentence_starts": next(replies)})

    starts = translator.sentence_starts(["あ" for _ in range(150)])

    # 75 from window 1; window 2: 70+5 ignored (overlap), 70+12=82 and 70+40=110 kept.
    assert starts == [0, 30, 75, 82, 110]


def test_complete_falls_back_when_structured_output_exhausts_tokens():
    """A proxy JSON-schema token-limit rejection must retry as json_object."""
    formats = []

    class Completions:
        def create(self, **kwargs):
            formats.append(kwargs["response_format"]["type"])
            if len(formats) == 1:
                request = httpx.Request("POST", "https://example.invalid")
                response = httpx.Response(400, request=request)
                raise openai.BadRequestError(
                    "json_validate_failed: max completion tokens reached",
                    response=response,
                    body={},
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"0":"ok"}'))]
            )

    translator = OpenAITranslator.__new__(OpenAITranslator)
    translator._client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    translator._model = "qwen/qwen3.8-27b"
    translator._supports_schema = True

    assert translator._complete("system", "user", schema={"type": "object"}) == '{"0":"ok"}'
    assert formats == ["json_schema", "json_object"]
    assert translator._supports_schema is False


def test_latam_variety_only_applies_to_spanish():
    from nadeshiko_dev_tools.common.llm_translator import _translation_system_prompt

    assert "neutral Latin American Spanish" in _translation_system_prompt("Japanese", "Spanish")
    english = _translation_system_prompt("Japanese", "English")
    assert "Latin American" not in english
    assert "colloquial English" in english


def _failed_generation():
    request = httpx.Request("POST", "https://example.invalid")
    return openai.BadRequestError(
        "Error code: 400 - json_validate_failed: Failed to generate JSON",
        response=httpx.Response(400, request=request),
        body={},
    )


def test_a_batch_with_invalid_json_is_halved_instead_of_failing():
    """One bad batch must not lose a whole chapter: it is split and retried."""
    translator = OpenAITranslator.__new__(OpenAITranslator)
    sizes = []

    def complete(_system, user, **_kwargs):
        items = json.loads(user)
        sizes.append(len(items))
        if len(items) == 4:  # the provider chokes on the full batch only
            raise _failed_generation()
        return json.dumps({k: f"t:{v}" for k, v in items.items()})

    translator._complete = complete

    out = translator._translate_chunk(["a", "b", "c", "d"], "Japanese", "English")

    assert out == ["t:a", "t:b", "t:c", "t:d"]
    assert sizes == [4, 2, 2]


def test_a_line_that_never_translates_comes_back_empty_not_as_source():
    translator = OpenAITranslator.__new__(OpenAITranslator)
    calls = []

    def complete(*_args, **_kwargs):
        calls.append(1)
        raise _failed_generation()

    translator._complete = complete

    assert translator._translate_chunk(["文"], "Japanese", "Spanish") == [""]
    assert len(calls) == 1 + 3  # the batch, then three single-line attempts


def test_other_bad_requests_still_raise():
    translator = OpenAITranslator.__new__(OpenAITranslator)
    request = httpx.Request("POST", "https://example.invalid")

    def complete(*_args, **_kwargs):
        raise openai.BadRequestError(
            "model not found", response=httpx.Response(400, request=request), body={}
        )

    translator._complete = complete
    import pytest

    with pytest.raises(openai.BadRequestError):
        translator._translate_chunk(["a", "b"], "Japanese", "English")


def test_usage_adds_up_what_the_provider_reports():
    translator = OpenAITranslator.__new__(OpenAITranslator)
    translator.usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}

    translator._count_usage(
        SimpleNamespace(usage=SimpleNamespace(prompt_tokens=700, completion_tokens=560))
    )
    translator._count_usage(SimpleNamespace(usage=None))

    assert translator.usage == {"requests": 2, "prompt_tokens": 700, "completion_tokens": 560}
