"""Japanese tokenizers: sudachipy and fugashi (UniDic).

Provides word-level tokenization with POS tags, readings, and normalized forms.

UniDic dictionary resolution order:
  1. UNIDIC_DIR env var — path to an extracted UniDic-CWJ directory
  2. `unidic` Python package (pip install unidic && python -m unidic download)

The latest UniDic-CWJ (2025.12) can be downloaded from NINJAL:
  https://clrd.ninjal.ac.jp/unidic/download.html
Extract the zip and set UNIDIC_DIR to the extracted directory path.
"""

from __future__ import annotations

import os

# POS categories to filter out (whitespace, symbols, supplementary symbols)
_SKIP_POS = frozenset({"空白", "補助記号", "記号"})


def _resolve_unidic_dicdir() -> tuple[str | None, str | None]:
    """Resolve the UniDic dictionary directory path and version.

    Returns (path_string, version_string) or (None, None) if not found.
    """
    # 1. Explicit env var
    env_dir = os.environ.get("UNIDIC_DIR")
    if env_dir:
        # Expand both ~ and $HOME/VARS
        dicdir = os.path.expandvars(os.path.expanduser(env_dir))
        if os.path.isdir(dicdir):
            version = _read_unidic_version(dicdir)
            return dicdir, version
        raise FileNotFoundError(f"UNIDIC_DIR={env_dir} does not exist or is not a directory")

    # 2. unidic Python package
    try:
        import unidic

        if os.path.isdir(unidic.DICDIR):
            version = _read_unidic_version(unidic.DICDIR)
            return unidic.DICDIR, version
    except ImportError:
        pass

    return None, None


def _read_unidic_version(dicdir: str) -> str | None:
    """Read UniDic version from README.md or version file in dicdir."""
    # First try README.md (for manual downloads like 2025.12)
    readme_file = os.path.join(dicdir, "README.md")
    if os.path.isfile(readme_file):
        try:
            import re

            with open(readme_file) as f:
                content = f.read()
            # Look for "ver.2025.12" or "2025.12" pattern
            match = re.search(r"ver\.?(\d{4}\.\d{2})", content)
            if match:
                return match.group(1).replace(".", "-")  # Convert to YYYY-MM
        except Exception:
            pass

    # Try old-style version file
    version_file = os.path.join(dicdir, "version")
    if os.path.isfile(version_file):
        try:
            with open(version_file) as f:
                content = f.read().strip()
            # Format: unidic-3.1.0+2021-08-31 or just date
            if "+" in content:
                return content.split("+", 1)[1]  # Return date part
            return content
        except Exception:
            pass

    # Try to infer from directory name
    dir_name = os.path.basename(dicdir)
    match = _extract_version_from_path(dir_name)
    if match:
        return match

    return None


def _extract_version_from_path(path: str) -> str | None:
    """Extract version from path (e.g., unidic-cwj-202512 -> 2025-12)."""
    import re

    # Match 202512 or 2025.12 or 2025-12 patterns
    match = re.search(r"20(\d{2})[.-]?(\d{2})", path)
    if match:
        return f"20{match.group(1)}-{match.group(2)}"
    return None


class JapaneseTokenizer:
    """Wraps sudachipy for Japanese text tokenization."""

    def __init__(self):
        from sudachipy import Dictionary

        self._tokenizer = Dictionary(dict="full").create()
        self._sudachi_version, self._dict_version = self._get_versions()

    def _get_versions(self) -> tuple[str, str]:
        """Get Sudachi library and dictionary versions."""
        # Library version
        try:
            from sudachipy import __version__ as sudachi_version
        except Exception:
            sudachi_version = "unknown"

        # Dictionary version - use package metadata
        dict_version = "unknown"
        try:
            import importlib.metadata

            dict_version = importlib.metadata.version("sudachidict-full")
            # Format YYYYMMDD as YYYY-MM-DD for readability
            if len(dict_version) == 8 and dict_version.isdigit():
                dict_version = f"{dict_version[:4]}-{dict_version[4:6]}-{dict_version[6:8]}"
        except Exception:
            pass

        return sudachi_version, dict_version

    @property
    def info(self) -> str:
        """Return tokenizer version string: lib<version>.dic<version>."""
        return f"lib{self._sudachi_version}.dic{self._dict_version}"

    def tokenize(self, text: str) -> list[dict]:
        """Tokenize Japanese text into word dicts.

        Returns core sudachipy fields:
          surface, reading, normalized_form, dictionary_form, pos,
          begin, end.
        """
        from sudachipy import SplitMode

        text = text.strip()
        if not text:
            return []

        tokens = self._tokenizer.tokenize(text, SplitMode.C)
        result = []

        for token in tokens:
            pos_parts = list(token.part_of_speech())
            top_pos = pos_parts[0] if pos_parts else ""

            if top_pos in _SKIP_POS:
                continue

            # Pad POS hierarchy to 6 elements
            while len(pos_parts) < 6:
                pos_parts.append("*")

            result.append(
                {
                    "surface": token.surface(),
                    "reading": token.reading_form(),
                    "normalized_form": token.normalized_form(),
                    "dictionary_form": token.dictionary_form(),
                    "pos": pos_parts[:6],
                    "begin": token.begin(),
                    "end": token.end(),
                }
            )

        return result


class UnidicTokenizer:
    """Wraps fugashi (MeCab) with UniDic for Japanese text tokenization.

    Dictionary is resolved via UNIDIC_DIR env var or the `unidic` Python package.
    """

    def __init__(self):
        import fugashi

        dicdir, version = _resolve_unidic_dicdir()
        if dicdir is None:
            raise RuntimeError(
                "No UniDic dictionary found. Either set UNIDIC_DIR env var "
                "or install the unidic package (pip install unidic && "
                "python -m unidic download)"
            )

        self._tagger = fugashi.Tagger(f"-d {dicdir}")
        self._dicdir = dicdir
        self._version = version or "unknown"

    @property
    def info(self) -> str:
        """Return tokenizer version string: dic<version>."""
        return f"dic{self._version}"

    @property
    def dicdir(self) -> str:
        """Path to the UniDic dictionary directory in use."""
        return self._dicdir

    @property
    def version(self) -> str:
        """UniDic dictionary version."""
        return self._version

    @staticmethod
    def _f(feat, name: str, default: str = "*") -> str:
        """Read a UniDic feature field, coercing None to a default."""
        val = getattr(feat, name, default)
        return val if val is not None else default

    def tokenize(self, text: str) -> list[dict]:
        """Tokenize Japanese text into word dicts using UniDic.

        Returns UniDic fields grouped logically:
          surface, reading/pronunciation, lemma, POS, conjugation,
          accent, word origin, initial/final sound changes.
        """
        text = text.strip()
        if not text:
            return []

        words = self._tagger(text)
        result = []

        for word in words:
            if word.feature is None:
                continue

            feat = word.feature
            pos1 = self._f(feat, "pos1")
            if pos1 in _SKIP_POS:
                continue

            result.append(
                {
                    "surface": word.surface,
                    # Reading (kana)
                    "kana": self._f(feat, "kana"),
                    "kanaBase": self._f(feat, "kanaBase"),
                    # Word form reading
                    "form": self._f(feat, "form"),
                    "formBase": self._f(feat, "formBase"),
                    # Pronunciation (actual spoken form, e.g. は→ワ)
                    "pron": self._f(feat, "pron"),
                    "pronBase": self._f(feat, "pronBase"),
                    # Lemma
                    "lemma": self._f(feat, "lemma"),
                    "lForm": self._f(feat, "lForm"),
                    # POS (4-level hierarchy)
                    "pos": [
                        pos1,
                        self._f(feat, "pos2"),
                        self._f(feat, "pos3"),
                        self._f(feat, "pos4"),
                    ],
                    # Conjugation
                    "cType": self._f(feat, "cType"),
                    "cForm": self._f(feat, "cForm"),
                    # Word origin
                    "goshu": self._f(feat, "goshu"),
                    # Accent
                    "aType": self._f(feat, "aType"),
                    "aConType": self._f(feat, "aConType"),
                    "aModType": self._f(feat, "aModType"),
                    # Initial sound change (連濁 etc.)
                    "iType": self._f(feat, "iType"),
                    "iForm": self._f(feat, "iForm"),
                    "iConType": self._f(feat, "iConType"),
                    # Final sound change (撥音化 etc.)
                    "fType": self._f(feat, "fType"),
                    "fForm": self._f(feat, "fForm"),
                    "fConType": self._f(feat, "fConType"),
                }
            )

        return result


def get_available_tokenizers() -> dict[str, dict]:
    """Return info about available tokenizers.

    Returns {"sudachi": {"version": "libX.YZ.dicYYYY-MM-DD"}, "unidic": {...}}
    with missing engines omitted or having "error" key.
    """
    result = {}

    # Try Sudachi
    try:
        tok = JapaneseTokenizer()
        result["sudachi"] = {"version": tok.info}
    except Exception as e:
        result["sudachi"] = {"error": str(e)}

    # Try UniDic
    try:
        tok = UnidicTokenizer()
        result["unidic"] = {"version": tok.info}
    except Exception as e:
        result["unidic"] = {"error": str(e)}

    return result
