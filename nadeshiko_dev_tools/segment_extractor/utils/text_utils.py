import re

import jaconvV2

_EMOJI_RE = re.compile(
    "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f1e0-\U0001f1ff]+",
    re.UNICODE,
)
_SIGN_NAME_RE = re.compile(r"\bsign\b")
_NON_DIALOGUE_STYLE_RE = re.compile(
    r"sign|top|tipo tv|block|cart|postal|carta|pizarra"  # on-screen text
    r"|tabla|contrato|abanico|electr|taxi"  # on-screen props
    r"|lyric|song|romaji|kara|engsub|\benglish\b"  # lyrics (EN)
    r"|español|japka|japro|\besp\b|\bfin\b|espanis"  # lyrics (ES/JA-romaji)
    r"|\bop\b|\bed\b"  # OP/ED (word boundary to avoid matching "Default")
    r"|note|sala"  # translator notes / room labels
    r"|\btitle\b|\bep\b|\bnext\b"  # episode titles
)
_POS_MOVE_RE = re.compile(r"pos\(.*?\)|move\(.*?\)")
_AN_NON_DEFAULT_RE = re.compile(r"\{\\an[1345679]\}")
_AN8_RE = re.compile(r"\{\\an8\}")
_WHITESPACE_RE = re.compile(r"\r?\n|\t")
_SPECIAL_CHARS_RE = re.compile(r"⚟|⚞|<|>|=|●|→|ー?♪ー?|\u202a|\u202c|➡|&lrm;")
_ASS_DRAWING_RE = re.compile(r"^{[^}]*}m\s+\d")
_KARAOKE_TAG_RE = re.compile(r"\\[kK][fo]?\d")
_HAS_LATIN_RE = re.compile(r"[A-Z]")
_INVALID_QUOTES_RE = re.compile(r"``|''")
_REDUNDANT_SYMBOLS_RE = re.compile(
    r"(?<=\.\.\.)-|(?<=\?)-|(?<=!)-|(?<=\.)-|(?<=,)-|(?<=ー)-|(?<=-)-|(?<=。)\s|^-|(?<=\s)+\s|(?<=\.\.\.)。"
)


def process_subtitle_line(line):
    if line.type != "Dialogue":
        return ""

    if line.name and _SIGN_NAME_RE.search(line.name.lower()):
        return ""

    if line.style and _NON_DIALOGUE_STYLE_RE.search(line.style.lower()):
        return ""

    if _POS_MOVE_RE.search(line.text):
        return ""

    # ASS drawing commands (used for signs/shapes, not dialogue)
    if _ASS_DRAWING_RE.search(line.text):
        return ""

    # Karaoke tags (\k, \K, \kf, \ko) — lyrics timing
    if _KARAOKE_TAG_RE.search(line.text):
        return ""

    # Non-default alignment: positioned signs
    # {\an8} is used for both signs and top-positioned narration, so only filter if
    # the plaintext is entirely uppercase (sign/title text, not dialogue)
    if _AN_NON_DEFAULT_RE.search(line.text):
        return ""
    if _AN8_RE.search(line.text):
        plaintext = line.plaintext.strip()
        if plaintext and plaintext == plaintext.upper() and _HAS_LATIN_RE.search(plaintext):
            return ""

    processed_sentence = jaconvV2.normalize(line.plaintext, "NFKC")
    processed_sentence = _WHITESPACE_RE.sub(" ", processed_sentence)
    processed_sentence = remove_nested_parenthesis(processed_sentence)
    processed_sentence = _SPECIAL_CHARS_RE.sub("", processed_sentence)
    processed_sentence = _EMOJI_RE.sub("", processed_sentence)

    return processed_sentence.strip()


def remove_nested_parenthesis(sentence):
    nb_rep = 1
    while nb_rep:
        sentence, nb_rep = re.subn(
            r"\([^\(\)（）\[\]\{\}《》【】]*\)|\[[^\(\)（）\[\]\{\}《》【】]*\]",
            "",
            sentence,
        )

    return sentence


def join_sentences_to_segment(sentences, ln):
    join_symbol = "　" if ln == "ja" else " "
    joined_sentence = join_symbol.join(x["sentence"].strip() for x in sentences)

    joined_sentence = _INVALID_QUOTES_RE.sub('"', joined_sentence)
    joined_sentence = _REDUNDANT_SYMBOLS_RE.sub("", joined_sentence)

    actor_sentence = ",".join(sorted({x["actor"].replace("\t", "").strip() for x in sentences}))

    subs_details = [
        {
            "id": s["sub_id"],
            "text": s["sentence"],
            "start_ms": s["start"],
            "end_ms": s["end"],
            "actor": s["actor"].replace("\t", "").strip() or None,
        }
        for s in sentences
    ]

    return joined_sentence, actor_sentence, subs_details
