import re

import jaconvV2

emoji = re.compile(
    "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f1e0-\U0001f1ff]+",
    re.UNICODE,
)


def process_subtitle_line(line):
    if line.type != "Dialogue":
        return ""

    if line.name and re.search(r"sign|[_\-\s]?ed|op[_\-\s]?", line.name.lower()):
        return ""

    if line.style and re.search(
        r"top|sign|tipo tv|block|alt|cart|lyric|song|\btitle\b|\bep\b|\bnext\b",
        line.style.lower(),
    ):
        return ""

    if re.search(r"pos\(.*?\)|move\(.*?\)", line.text):
        return ""

    # Filter ASS lines that are positioned signs: non-default alignment + ALL CAPS text
    # {\an8} is used for both signs and top-positioned narration, so only filter if
    # the plaintext is entirely uppercase (sign/title text, not dialogue)
    if re.search(r"\{\\an[1345679]\}", line.text):
        return ""
    if re.search(r"\{\\an8\}", line.text):
        plaintext = line.plaintext.strip()
        if plaintext and plaintext == plaintext.upper() and re.search(r"[A-Z]", plaintext):
            return ""

    processed_sentence = jaconvV2.normalize(line.plaintext, "NFKC")
    processed_sentence = re.sub("\r?\n|\t", " ", processed_sentence)
    processed_sentence = remove_nested_parenthesis(processed_sentence)

    special_chars = r"⚟|⚞|<|>|=|●|→|ー?♪ー?|\u202a|\u202c|➡|&lrm;"
    processed_sentence = re.sub(special_chars, "", processed_sentence)

    processed_sentence = emoji.sub("", processed_sentence)

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

    invalid_quotes = r"``|''"
    joined_sentence = re.sub(invalid_quotes, '"', joined_sentence)

    remove_redundant_symbols = [
        r"(?<=\.\.\.)-",
        r"(?<=\?)-",
        r"(?<=!)-",
        r"(?<=\.)-",
        r"(?<=,)-",
        r"(?<=ー)-",
        r"(?<=-)-",
        r"(?<=。)\s",
        r"^-",
        r"(?<=\s)+\s",
        r"(?<=\.\.\.)。",
    ]

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

    return (
        re.sub(rf"{'|'.join(remove_redundant_symbols)}", "", joined_sentence),
        actor_sentence,
        subs_details,
    )
