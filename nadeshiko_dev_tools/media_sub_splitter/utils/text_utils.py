import re
import string

import jaconvV2

emoji = re.compile(
    "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f1e0-\U0001f1ff]+",
    re.UNICODE,
)


def process_subtitle_line(line, config):
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

    processed_sentence = jaconvV2.normalize(line.plaintext, "NFKC")
    processed_sentence = re.sub("\r?\n|\t", " ", processed_sentence)

    if hasattr(config, "extra_punctuation") and config.extra_punctuation:
        processed_sentence = processed_sentence.replace("・", " ")

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


def extract_anime_title_for_guessit(episode_filepath):
    return re.sub(
        r"\[.*?\]|1080p|720p|BDRip|Dual\s?Audio|x?26[4|5]-?|HEVC|10\sbits|EMBER",
        "",
        " ".join(episode_filepath.split("/")[-2:]),
    )


def map_anime_title_to_media_folder(anime_title):
    return "-".join(
        anime_title.lower().translate(str.maketrans("", "", string.punctuation)).split()
    )
