import pysubs2
from langdetect import detect

SUPPORTED_LANGUAGES = ["en", "ja", "es"]


def load_subtitle_file(filepath: str) -> pysubs2.SSAFile:
    return pysubs2.load(filepath)


def detect_subtitle_language(subtitle_data: pysubs2.SSAFile) -> str:
    subtitle_text = " ".join([event.text for event in subtitle_data])
    return detect(subtitle_text)


def validate_subtitle_language(language: str) -> bool:
    return language in SUPPORTED_LANGUAGES
