import pysubs2

SUPPORTED_LANGUAGES = ["en", "ja", "es"]


def load_subtitle_file(filepath: str) -> pysubs2.SSAFile:
    return pysubs2.load(filepath)
