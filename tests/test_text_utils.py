"""Tests for the ASS subtitle style/name filter in process_subtitle_line."""

from dataclasses import dataclass

import pytest

from nadeshiko_dev_tools.segment_extractor.utils.text_utils import process_subtitle_line


@dataclass
class FakeLine:
    """Minimal stand-in for a pysubs2 SSAEvent."""

    text: str = "Hello world"
    plaintext: str = "Hello world"
    type: str = "Dialogue"
    name: str = ""
    style: str = "Default"


# --- Style filter tests ---

_DIALOGUE_STYLES = [
    "Default",
    "DefaultItalics",
    "Flashback",
    "FlashbackItalics",
    "Gen_Main",
    "Gen_Italics",
    "Gen_Main_Up",
    "Italics",
    "B1",
    "DefaultLow",
    "Congrats",
    "BorderAlpha",
    "BText",
]

_FILTERED_STYLES = [
    # Existing filters
    "On Top",
    "On Top Italic",
    "Default - On top",
    "FlashbackTop",
    "DefaultTop",
    "FlashbackItalicsTop",
    "Signs",
    "Box Signs",
    "Gen_Italics_top",
    "Cart_A_Tre",
    "Cart_C_Tre",
    "Cart_A_Ari",
    "Cart_TitleSeries",
    "Cart_EpiTitle",
    "tipo tv",
    # New: lyrics / song patterns
    "Lyrics-Jap",
    "Lyrics-Eng",
    "Songs_OP",
    "Songs_ED",
    "Songs_ED2",
    "Songs_ED2b",
    "Songs_ED3",
    "Songs_ED_Basic",
    "Songs_Insert",
    # New: ep/title/next patterns
    "Ep Titles",
    "Next Ep",
]


@pytest.mark.parametrize("style", _DIALOGUE_STYLES)
def test_dialogue_styles_pass_through(style):
    line = FakeLine(style=style)
    result = process_subtitle_line(line)
    assert result != "", f"Style '{style}' should NOT be filtered"


@pytest.mark.parametrize("style", _FILTERED_STYLES)
def test_non_dialogue_styles_filtered(style):
    line = FakeLine(style=style)
    result = process_subtitle_line(line)
    assert result == "", f"Style '{style}' SHOULD be filtered"


# --- Name/actor filter tests ---

_FILTERED_NAMES = [
    "Sign",
    "sign",
    "OP",
    "op",
    "ED",
    "ed",
    "_ed",
    "op_",
]


@pytest.mark.parametrize("name", _FILTERED_NAMES)
def test_non_dialogue_names_filtered(name):
    line = FakeLine(name=name)
    result = process_subtitle_line(line)
    assert result == "", f"Name '{name}' SHOULD be filtered"


def test_regular_actor_name_passes():
    line = FakeLine(name="Takeshi")
    result = process_subtitle_line(line)
    assert result != ""


# --- Type filter ---


def test_comment_type_filtered():
    line = FakeLine(type="Comment")
    result = process_subtitle_line(line)
    assert result == ""


# --- Positioning filter ---


def test_pos_tag_filtered():
    line = FakeLine(text=r"{\pos(320,50)}Some sign text", plaintext="Some sign text")
    result = process_subtitle_line(line)
    assert result == ""


def test_move_tag_filtered():
    line = FakeLine(text=r"{\move(0,0,320,50)}Moving text", plaintext="Moving text")
    result = process_subtitle_line(line)
    assert result == ""
