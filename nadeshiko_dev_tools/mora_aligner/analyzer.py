"""Text analysis: sudachipy tokenization + pyopenjtalk phoneme/accent extraction.

Produces word-level mora breakdown with accent H/L predictions and
relative duration ratios from OpenJTalk's HTS labels.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field


@dataclass
class MoraUnit:
    """A single mora from HTS label parsing."""

    kana: str
    phonemes: list[str]
    duration: float  # predicted duration from HTS (arbitrary units)
    accent_phrase_idx: int
    mora_position: int  # 1-based position within accent phrase
    accent_phrase_len: int
    accent_nucleus: int  # 0 = 平板
    accent: str = ""  # H or L, assigned after grouping


@dataclass
class WordMora:
    """A word with its mora breakdown."""

    surface: str
    reading: str
    pos: str
    mora: list[MoraUnit] = field(default_factory=list)


# Phoneme-to-kana mapping for mora grouping
# Maps (consonant, vowel) pairs and standalone vowels to kana
_VOWELS = {"a", "i", "u", "e", "o"}
_SPECIAL_MORA = {"N": "ン", "cl": "ッ"}

_CV_TO_KANA: dict[tuple[str, str], str] = {
    ("k", "a"): "カ", ("k", "i"): "キ", ("k", "u"): "ク", ("k", "e"): "ケ", ("k", "o"): "コ",
    ("s", "a"): "サ", ("sh", "i"): "シ", ("s", "u"): "ス", ("s", "e"): "セ", ("s", "o"): "ソ",
    ("t", "a"): "タ", ("ch", "i"): "チ", ("ts", "u"): "ツ", ("t", "e"): "テ", ("t", "o"): "ト",
    ("n", "a"): "ナ", ("n", "i"): "ニ", ("n", "u"): "ヌ", ("n", "e"): "ネ", ("n", "o"): "ノ",
    ("h", "a"): "ハ", ("h", "i"): "ヒ", ("f", "u"): "フ", ("h", "e"): "ヘ", ("h", "o"): "ホ",
    ("m", "a"): "マ", ("m", "i"): "ミ", ("m", "u"): "ム", ("m", "e"): "メ", ("m", "o"): "モ",
    ("y", "a"): "ヤ", ("y", "u"): "ユ", ("y", "o"): "ヨ",
    ("r", "a"): "ラ", ("r", "i"): "リ", ("r", "u"): "ル", ("r", "e"): "レ", ("r", "o"): "ロ",
    ("w", "a"): "ワ", ("w", "o"): "ヲ",
    ("g", "a"): "ガ", ("g", "i"): "ギ", ("g", "u"): "グ", ("g", "e"): "ゲ", ("g", "o"): "ゴ",
    ("z", "a"): "ザ", ("j", "i"): "ジ", ("z", "u"): "ズ", ("z", "e"): "ゼ", ("z", "o"): "ゾ",
    ("d", "a"): "ダ", ("d", "i"): "ヂ", ("d", "u"): "ヅ", ("d", "e"): "デ", ("d", "o"): "ド",
    ("b", "a"): "バ", ("b", "i"): "ビ", ("b", "u"): "ブ", ("b", "e"): "ベ", ("b", "o"): "ボ",
    ("p", "a"): "パ", ("p", "i"): "ピ", ("p", "u"): "プ", ("p", "e"): "ペ", ("p", "o"): "ポ",
    # Palatalized (拗音) - these form single mora with y-glide
    ("ky", "a"): "キャ", ("ky", "u"): "キュ", ("ky", "o"): "キョ",
    ("sh", "a"): "シャ", ("sh", "u"): "シュ", ("sh", "o"): "ショ",
    ("ch", "a"): "チャ", ("ch", "u"): "チュ", ("ch", "o"): "チョ",
    ("ny", "a"): "ニャ", ("ny", "u"): "ニュ", ("ny", "o"): "ニョ",
    ("hy", "a"): "ヒャ", ("hy", "u"): "ヒュ", ("hy", "o"): "ヒョ",
    ("my", "a"): "ミャ", ("my", "u"): "ミュ", ("my", "o"): "ミョ",
    ("ry", "a"): "リャ", ("ry", "u"): "リュ", ("ry", "o"): "リョ",
    ("gy", "a"): "ギャ", ("gy", "u"): "ギュ", ("gy", "o"): "ギョ",
    ("j", "a"): "ジャ", ("j", "u"): "ジュ", ("j", "o"): "ジョ",
    ("by", "a"): "ビャ", ("by", "u"): "ビュ", ("by", "o"): "ビョ",
    ("py", "a"): "ピャ", ("py", "u"): "ピュ", ("py", "o"): "ピョ",
    ("dy", "a"): "ヂャ", ("dy", "u"): "ヂュ", ("dy", "o"): "ヂョ",
    # Additional combinations
    ("ts", "a"): "ツァ", ("t", "i"): "ティ",
    ("f", "a"): "ファ", ("f", "i"): "フィ", ("f", "e"): "フェ", ("f", "o"): "フォ",
    ("w", "i"): "ウィ", ("w", "e"): "ウェ",
    ("v", "a"): "ヴァ", ("v", "i"): "ヴィ", ("v", "u"): "ヴ",
    ("v", "e"): "ヴェ", ("v", "o"): "ヴォ",
}

_V_TO_KANA: dict[str, str] = {
    "a": "ア", "i": "イ", "u": "ウ", "e": "エ", "o": "オ",
}

# Phoneme-class relative durations from Japanese TTS research.
# Vowels are longer, consonants shorter, special mora (ッ/ン) shorter still.
# These give relative proportions — they're scaled to fit actual subtitle timing.
_PHONE_DURATION: dict[str, float] = {
    # Vowels
    "a": 1.0, "i": 0.85, "u": 0.75, "e": 0.9, "o": 0.9,
    # Special mora
    "N": 0.65, "cl": 0.45,
    # Plosives
    "k": 0.35, "g": 0.3, "t": 0.3, "d": 0.25, "p": 0.3, "b": 0.25,
    # Fricatives / affricates
    "s": 0.45, "sh": 0.5, "z": 0.35, "j": 0.35,
    "ch": 0.4, "ts": 0.4, "h": 0.25, "f": 0.3,
    # Nasals
    "n": 0.3, "m": 0.3, "ny": 0.3, "my": 0.3,
    # Approximants
    "y": 0.2, "r": 0.25, "w": 0.2,
    # Palatalized
    "ky": 0.35, "gy": 0.3, "by": 0.25, "py": 0.3,
    "hy": 0.25, "ry": 0.25, "dy": 0.25,
    # Rare
    "v": 0.3,
}

# HTS label regex: extracts phoneme identity and context fields
_LABEL_RE = re.compile(
    r"^(?P<p1>[^\^]+)\^(?P<p2>[^-]+)-(?P<phone>[^+]+)\+(?P<p4>[^=]+)="
    r"(?P<p5>[^/]+)/A:(?P<A>[^/]+)/B:(?P<B>[^/]+)/C:(?P<C>[^/]+)"
    r"/D:(?P<D>[^/]+)/E:(?P<E>[^/]+)/F:(?P<F>[^/]+)/G:(?P<G>[^/]+)"
    r"/H:(?P<H>[^/]+)/I:(?P<I>[^/]+)"
)


def _parse_fullcontext(labels: list[str]) -> list[MoraUnit]:
    """Parse HTS full-context labels into mora units.

    Each label corresponds to one phoneme. We group phonemes into mora,
    extract accent phrase info, and compute H/L for each mora.
    """
    phonemes: list[dict] = []

    for label in labels:
        m = _LABEL_RE.match(label)
        if not m:
            continue

        phone = m.group("phone")
        if phone in ("sil", "pau"):
            continue

        # /A: accent difference, mora position in phrase, mora count to accent
        a_parts = m.group("A").split("+")
        # /F: mora count in accent phrase, accent nucleus position
        f_parts = m.group("F").split("_")

        # Parse mora position within accent phrase (from /A:x+y+z, y = position)
        mora_pos = int(a_parts[1]) if len(a_parts) >= 2 and a_parts[1] != "xx" else 0

        # Parse accent phrase info from /F:x_y#... (x = total mora, y = accent pos)
        f_mora_count = 0
        f_accent_pos = 0
        if len(f_parts) >= 2:
            f_mora_str = f_parts[0]
            # Second part may have additional # fields
            f_accent_str = f_parts[1].split("#")[0] if "#" in f_parts[1] else f_parts[1]
            with contextlib.suppress(ValueError):
                f_mora_count = int(f_mora_str)
            with contextlib.suppress(ValueError):
                f_accent_pos = int(f_accent_str)

        # Determine accent phrase index from /I: field (phrase index)
        i_parts = m.group("I").split("_")
        phrase_idx = 0
        with contextlib.suppress(ValueError):
            phrase_idx = int(i_parts[0])

        phonemes.append({
            "phone": phone,
            "mora_pos": mora_pos,
            "phrase_idx": phrase_idx,
            "phrase_len": f_mora_count,
            "accent_nucleus": f_accent_pos,
            "duration": _PHONE_DURATION.get(phone, 0.5),
        })

    # Group phonemes into mora
    mora_list: list[MoraUnit] = []
    i = 0
    while i < len(phonemes):
        p = phonemes[i]
        phone = p["phone"]

        if phone in _SPECIAL_MORA:
            # N (moraic nasal) or cl (geminate)
            mora_list.append(MoraUnit(
                kana=_SPECIAL_MORA[phone],
                phonemes=[phone],
                duration=p["duration"],
                accent_phrase_idx=p["phrase_idx"],
                mora_position=p["mora_pos"],
                accent_phrase_len=p["phrase_len"],
                accent_nucleus=p["accent_nucleus"],
            ))
            i += 1
        elif phone in _VOWELS:
            # Standalone vowel mora
            mora_list.append(MoraUnit(
                kana=_V_TO_KANA.get(phone, phone),
                phonemes=[phone],
                duration=p["duration"],
                accent_phrase_idx=p["phrase_idx"],
                mora_position=p["mora_pos"],
                accent_phrase_len=p["phrase_len"],
                accent_nucleus=p["accent_nucleus"],
            ))
            i += 1
        else:
            # Consonant: look ahead for vowel
            if i + 1 < len(phonemes) and phonemes[i + 1]["phone"] in _VOWELS:
                vowel_p = phonemes[i + 1]
                vowel = vowel_p["phone"]
                kana = _CV_TO_KANA.get((phone, vowel), phone + vowel)
                mora_list.append(MoraUnit(
                    kana=kana,
                    phonemes=[phone, vowel],
                    duration=p["duration"] + vowel_p["duration"],
                    accent_phrase_idx=vowel_p["phrase_idx"],
                    mora_position=vowel_p["mora_pos"],
                    accent_phrase_len=vowel_p["phrase_len"],
                    accent_nucleus=vowel_p["accent_nucleus"],
                ))
                i += 2
            else:
                # Lone consonant — skip (shouldn't normally happen)
                i += 1

    # Assign H/L accent to each mora
    for mora in mora_list:
        n = mora.accent_nucleus  # accent position (0 = heiban)
        pos = mora.mora_position

        if pos == 0:
            # No position info available
            mora.accent = "L"
        elif n == 0:
            # 平板型 (heiban): first mora L, rest H
            mora.accent = "L" if pos == 1 else "H"
        elif n == 1:
            # 頭高型 (atamadaka): first mora H, rest L
            mora.accent = "H" if pos == 1 else "L"
        else:
            # 中高型/尾高型: mora 1 is L, 2..N are H, after N are L
            if pos == 1:
                mora.accent = "L"
            elif pos <= n:
                mora.accent = "H"
            else:
                mora.accent = "L"

    return mora_list


def _mora_to_reading(mora_list: list[MoraUnit]) -> str:
    """Concatenate mora kana to form a reading string."""
    return "".join(m.kana for m in mora_list)


def _katakana_to_mora_count(reading: str) -> int:
    """Count mora in a katakana string (small kana are not separate mora)."""
    small = set("ァィゥェォャュョヮ")
    return sum(1 for ch in reading if ch not in small)


# POS categories that produce no phonemes in pyopenjtalk
_SKIP_POS = {"空白", "補助記号", "記号"}


def _is_phonetic_word(word: dict) -> bool:
    """Check if a word contributes phonemes (is not whitespace/punctuation/symbol)."""
    if word["pos"] in _SKIP_POS:
        return False
    # Also skip if reading is all non-kana (punctuation with non-empty reading)
    reading = word["reading"]
    if not reading:
        return False
    # Check for at least one katakana character
    return any("\u30A0" <= ch <= "\u30FF" for ch in reading)


def _reading_to_mora_kana(reading: str) -> list[str]:
    """Split a katakana reading string into individual mora kana."""
    small = set("ァィゥェォャュョヮ")
    result = []
    for ch in reading:
        if ch in small and result:
            result[-1] += ch
        elif "\u30A0" <= ch <= "\u30FF":
            result.append(ch)
    return result


def _align_mora_to_words(
    mora_list: list[MoraUnit], sudachi_words: list[dict]
) -> list[WordMora]:
    """Match mora sequence to sudachipy word boundaries.

    Uses kana matching: for each word, consumes mora whose kana matches
    the word's reading. Handles pyopenjtalk dropping devoiced vowels
    (e.g., ス in です) by allowing partial matches.
    """
    result: list[WordMora] = []
    mora_idx = 0

    for word in sudachi_words:
        surface = word["surface"]
        reading = word["reading"]
        pos = word["pos"]

        if not _is_phonetic_word(word):
            result.append(WordMora(surface=surface, reading=reading, pos=pos, mora=[]))
            continue

        word_reading_mora = _reading_to_mora_kana(reading)
        if not word_reading_mora:
            result.append(WordMora(surface=surface, reading=reading, pos=pos, mora=[]))
            continue

        # Greedily match mora from the list against this word's reading mora.
        # pyopenjtalk may skip some mora (devoiced vowels), so we track
        # which reading mora we've matched and allow skipping.
        word_mora: list[MoraUnit] = []
        reading_pos = 0

        while reading_pos < len(word_reading_mora) and mora_idx < len(mora_list):
            mora_kana = mora_list[mora_idx].kana
            expected_kana = word_reading_mora[reading_pos]

            # Handle は→ワ (particle reading)
            if _kana_matches(mora_kana, expected_kana):
                word_mora.append(mora_list[mora_idx])
                mora_idx += 1
                reading_pos += 1
            else:
                # pyopenjtalk may have dropped this mora (devoiced)
                # Skip this reading position and try next
                reading_pos += 1

        result.append(WordMora(
            surface=surface,
            reading=reading,
            pos=pos,
            mora=word_mora,
        ))

    # Leftover mora: attach to last phonetic word
    if mora_idx < len(mora_list) and result:
        for w in reversed(result):
            if w.mora:
                w.mora.extend(mora_list[mora_idx:])
                break

    return result


def _kana_matches(mora_kana: str, reading_kana: str) -> bool:
    """Check if a mora kana matches a reading kana, with common equivalences."""
    if mora_kana == reading_kana:
        return True
    # Common substitutions: ワ↔ハ (particle は), ヲ↔オ, ヂ↔ジ, ヅ↔ズ
    equivalences = {
        ("ワ", "ハ"), ("ハ", "ワ"),
        ("ヲ", "オ"), ("オ", "ヲ"),
        ("ヂ", "ジ"), ("ジ", "ヂ"),
        ("ヅ", "ズ"), ("ズ", "ヅ"),
    }
    return (mora_kana, reading_kana) in equivalences


def analyze_text(text: str) -> list[WordMora]:
    """Full analysis pipeline: text → words with mora and accent H/L.

    Uses sudachipy for word segmentation and pyopenjtalk for phoneme/accent analysis.
    """
    import pyopenjtalk
    from sudachipy import Dictionary, SplitMode

    # Strip whitespace/punctuation edge cases
    text = text.strip()
    if not text:
        return []

    # 1. sudachipy tokenization (mode C = short unit)
    tokenizer = Dictionary().create()
    tokens = tokenizer.tokenize(text, SplitMode.C)

    sudachi_words: list[dict] = []
    for token in tokens:
        surface = token.surface()
        reading = token.reading_form()  # katakana
        pos_parts = token.part_of_speech()
        pos = pos_parts[0] if pos_parts else ""

        sudachi_words.append({
            "surface": surface,
            "reading": reading,
            "pos": pos,
        })

    # 2. pyopenjtalk full-context labels
    labels = pyopenjtalk.extract_fullcontext(text)

    # 3. Parse labels into mora with accent
    mora_list = _parse_fullcontext(labels)

    # 4. Align mora to words
    words = _align_mora_to_words(mora_list, sudachi_words)

    return words
