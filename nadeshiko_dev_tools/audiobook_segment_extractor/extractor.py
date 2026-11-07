"""Turn a Honomiya read-listen alignment into segment data.

Honomiya aligns an ebook with its audiobook and emits one cue per ebook sentence,
carrying the sentence's exact text and its audio interval. Each cue is already a
clean Nadeshiko segment, so unlike the anime/YouTube pipelines there is no subtitle
parsing or sentence grouping here.

Input folder (one per book):

    audio.m4b        chapters embedded in the file (read with ffprobe)
    alignment.json   Honomiya manifest, schema honomiya.read-listen.v1
    cover.(webp|jpg|jpeg|png)
    book.json        {asin, japanese_name, romaji_name, english_name, release_date}

Output (same contract as process-media / process-youtube, so tokenize-media,
quality-check and assets-uploader consume it unchanged):

    <out>/<asin>/_info.json, cover.<ext>
    <out>/<asin>/<chapter>/_data.json + {hash}.mp3 / .webp / .mp4

Chapter folders are numbered from 1 in m4b order. Segment times in _data.json are
relative to the chapter start, like episode times for anime.
"""

import hashlib
import json
import logging
import os
import secrets
import shutil
import subprocess
from dataclasses import dataclass

from nadeshiko_dev_tools.common.file_utils import atomic_write_json
from nadeshiko_dev_tools.common.translator import LANG_CODES
from nadeshiko_dev_tools.segment_extractor.utils.media import build_clip, extract_audio

logger = logging.getLogger(__name__)

# v2 adds per-word timings (Honomiya --words); cue fields are the same, words are ignored.
MANIFEST_SCHEMAS = ("honomiya.read-listen.v1", "honomiya.read-listen.v2")
TRANSLATIONS_FILE = "_translations.json"
AUDIO_FILE = "audio.m4b"
ALIGNMENT_FILE = "alignment.json"
BOOK_FILE = "book.json"
COVER_NAMES = ("cover.webp", "cover.jpg", "cover.jpeg", "cover.png")
BOOK_REQUIRED_FIELDS = ("asin", "japanese_name", "romaji_name", "english_name", "release_date")
# Nadeshiko's content rating is about the image, and every audiobook image is the cover, so SAFE
# unless book.json says otherwise.
CONTENT_RATINGS = ("SAFE", "SUGGESTIVE", "QUESTIONABLE", "EXPLICIT")
DEFAULT_CONTENT_RATING = "SAFE"

# Cues outside these bounds are alignment artefacts: a long one is usually a gap the
# aligner bridged (e.g. the author's name spanning 30 s of title narration), a very
# short one has no audible speech to study.
DEFAULT_MIN_DURATION_MS = 500
DEFAULT_MAX_DURATION_MS = 12_000

# Every segment's screenshot is the book cover, letterboxed to the size the anime
# screenshots use, so the search card and the Anki export need no special case.
_FRAME_FILE = "_segment_frame.webp"
_TARGET_LANGS = ("en", "es")


@dataclass(frozen=True)
class Cue:
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class Chapter:
    number: int  # 1-based, m4b order; also the output folder and the episode number
    title: str
    start_ms: int
    end_ms: int


class BookInputError(ValueError):
    """The input folder is incomplete or inconsistent."""


# ── Input ─────────────────────────────────────────────────────────────────────


def find_cover(book_folder: str) -> str | None:
    for name in COVER_NAMES:
        path = os.path.join(book_folder, name)
        if os.path.exists(path):
            return path
    return None


def load_book(book_folder: str) -> dict:
    """Read and validate book.json."""
    path = os.path.join(book_folder, BOOK_FILE)
    if not os.path.exists(path):
        raise BookInputError(f"Missing {BOOK_FILE} in {book_folder}")
    with open(path, encoding="utf-8") as f:
        book = json.load(f)
    missing = [k for k in BOOK_REQUIRED_FIELDS if not book.get(k)]
    if missing:
        raise BookInputError(f"{BOOK_FILE} is missing: {', '.join(missing)}")
    rating = book.setdefault("content_rating", DEFAULT_CONTENT_RATING)
    if rating not in CONTENT_RATINGS:
        raise BookInputError(
            f"{BOOK_FILE}: content_rating must be one of {', '.join(CONTENT_RATINGS)}"
        )
    return book


def load_cues(manifest: dict) -> list[Cue]:
    """Extract (start, end, text) cues from a Honomiya manifest.

    Only single-audio-file manifests with text-quote anchors are supported: those
    carry the sentence text inline, so the ebook itself is never needed.
    """
    if manifest.get("schema") not in MANIFEST_SCHEMAS:
        raise BookInputError(f"Unsupported alignment schema: {manifest.get('schema')!r}")
    audio_files = manifest.get("sources", {}).get("audioFiles", [])
    if len(audio_files) != 1:
        raise BookInputError(f"Expected an alignment for one audio file, found {len(audio_files)}")

    cues = []
    for raw in manifest.get("cues", []):
        anchor = raw.get("text", {})
        if anchor.get("kind") != "text-quote":
            raise BookInputError(
                f"Cue {raw.get('id')} uses a {anchor.get('kind')!r} anchor; "
                "only text-quote anchors carry their text"
            )
        cues.append(Cue(int(raw["startMs"]), int(raw["endMs"]), anchor.get("exact", "").strip()))
    cues.sort(key=lambda c: c.start_ms)
    return cues


def probe_chapters(audio_path: str) -> list[Chapter]:
    """Read the chapters embedded in an m4b with ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_chapters", "-print_format", "json", audio_path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return parse_chapters(json.loads(result.stdout))


def parse_chapters(ffprobe_output: dict) -> list[Chapter]:
    chapters = []
    for i, ch in enumerate(ffprobe_output.get("chapters", []), start=1):
        chapters.append(
            Chapter(
                number=i,
                title=(ch.get("tags", {}).get("title") or f"Chapter {i}").strip(),
                start_ms=round(float(ch["start_time"]) * 1000),
                end_ms=round(float(ch["end_time"]) * 1000),
            )
        )
    return chapters


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ── Pure transforms ───────────────────────────────────────────────────────────


def assign_chapters(cues: list[Cue], chapters: list[Chapter]) -> dict[int, list[Cue]]:
    """Group cues by the chapter that contains their start.

    A book without chapter metadata becomes a single chapter 1.
    """
    if not chapters:
        return {1: list(cues)} if cues else {}
    grouped: dict[int, list[Cue]] = {ch.number: [] for ch in chapters}
    for cue in cues:
        # The last chapter starting at or before the cue; chapters are contiguous.
        chapter = chapters[0]
        for ch in chapters:
            if ch.start_ms > cue.start_ms:
                break
            chapter = ch
        grouped[chapter.number].append(cue)
    return grouped


def rejection_reason(cue: Cue, min_ms: int, max_ms: int) -> str | None:
    """Why a cue can't be a segment, or None if it can."""
    duration = cue.end_ms - cue.start_ms
    if not cue.text:
        return "empty text"
    if duration > max_ms:
        return f"too long ({duration / 1000:.1f} s)"
    if duration < min_ms:
        return f"too short ({duration / 1000:.2f} s)"
    return None


def segment_hash(salt: str, asin: str, start_ms: int, text: str) -> str:
    payload = f"{salt}:{asin}:{start_ms}:{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


class TranslationCache:
    """EN/ES translations keyed by the exact Japanese sentence, kept in the output folder.

    Re-cutting a book (a new alignment, other duration limits) produces the same ebook sentences
    with new timings, so their translations can be reused instead of paid for again.
    """

    def __init__(self, path: str | None = None):
        self.path = path
        self.entries: dict[str, dict[str, str]] = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.entries = json.load(f)

    def get(self, ja: str, lang: str) -> str:
        return self.entries.get(ja, {}).get(lang, "")

    def put(self, ja: str, lang: str, text: str) -> None:
        if ja and text:
            self.entries.setdefault(ja, {})[lang] = text

    def save(self) -> None:
        if self.path:
            atomic_write_json(self.path, self.entries)

    def seed_from_output(self, folder: str) -> int:
        """Take translations from an earlier process-audiobook output; returns how many."""
        before = sum(len(v) for v in self.entries.values())
        other = os.path.join(folder, TRANSLATIONS_FILE)
        if os.path.exists(other):
            with open(other, encoding="utf-8") as f:
                for ja, langs in json.load(f).items():
                    for lang, text in langs.items():
                        if not self.get(ja, lang):
                            self.put(ja, lang, text)
        for entry in sorted(os.listdir(folder)):
            data_path = os.path.join(folder, entry, "_data.json")
            if not os.path.exists(data_path):
                continue
            with open(data_path, encoding="utf-8") as f:
                for seg in json.load(f).get("segments", []):
                    for lang in _TARGET_LANGS:
                        if not self.get(seg.get("content_ja", ""), lang):
                            self.put(
                                seg.get("content_ja", ""), lang, seg.get(f"content_{lang}") or ""
                            )
        return sum(len(v) for v in self.entries.values()) - before


def translate_all(
    translator, texts: list[str], cache: TranslationCache | None = None
) -> dict[str, list[str]]:
    """Translate JA texts to every target language, one batch call per language.

    Texts already in the cache are not sent; new translations are added to it. With no
    translator, only cached translations are returned and the rest stay empty.
    """
    cache = cache or TranslationCache()
    out: dict[str, list[str]] = {}
    for lang in _TARGET_LANGS:
        missing = list(dict.fromkeys(t for t in texts if not cache.get(t, lang)))
        if missing and translator is not None:
            results = translator.translate_text(
                missing, source_lang="JA", target_lang=LANG_CODES[lang]
            )
            for ja, r in zip(missing, results, strict=True):
                cache.put(ja, lang, (r.text or "").strip())
        out[lang] = [cache.get(t, lang) for t in texts]
    return out


# ── Output ────────────────────────────────────────────────────────────────────


def write_info_json(out_folder: str, book: dict, cover_path: str | None) -> str:
    """Create _info.json (once) and return the media's hash salt."""
    info_path = os.path.join(out_folder, "_info.json")
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            return json.load(f)["hash_salt"]

    if cover_path:
        shutil.copyfile(cover_path, os.path.join(out_folder, os.path.basename(cover_path)))

    hash_salt = secrets.token_hex(16)
    atomic_write_json(
        info_path,
        {
            "id": book["asin"],
            "asin": book["asin"],
            "media_source": "audible",
            "category": "AUDIOBOOK",
            "version": "6",
            "name": book["romaji_name"],
            "japanese_name": book["japanese_name"],
            "romaji_name": book["romaji_name"],
            "english_name": book["english_name"],
            "airing_format": "AUDIOBOOK",
            "airing_status": "FINISHED",
            "genres": book.get("genres", []),
            "start_date": book["release_date"],
            "hash_salt": hash_salt,
        },
    )
    return hash_salt


def prepare_frame(cover_path: str, out_folder: str) -> str:
    """Letterbox the cover to 960x540 webp once; every segment reuses it."""
    frame = os.path.join(out_folder, _FRAME_FILE)
    if not os.path.exists(frame):
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", cover_path,
                "-vf",
                "scale=960:540:force_original_aspect_ratio=decrease,"
                "pad=960:540:(ow-iw)/2:(oh-ih)/2:black",
                "-c:v", "libwebp", "-quality", "85", frame,
            ],
            capture_output=True,
            check=True,
        )  # fmt: skip
    return frame


def _ignored(index: int, cue: Cue, chapter: Chapter, reason: str) -> dict:
    return {
        "segment_index": index,
        "start_ms": cue.start_ms - chapter.start_ms,
        "end_ms": cue.end_ms - chapter.start_ms,
        "duration_ms": cue.end_ms - cue.start_ms,
        "content_ja": cue.text,
        "content_en": None,
        "content_es": None,
        "reason": reason,
    }


def process_chapter(
    chapter: Chapter,
    cues: list[Cue],
    *,
    audio_path: str,
    frame_path: str,
    out_folder: str,
    asin: str,
    hash_salt: str,
    translator,
    min_ms: int = DEFAULT_MIN_DURATION_MS,
    max_ms: int = DEFAULT_MAX_DURATION_MS,
    content_rating: str = DEFAULT_CONTENT_RATING,
    cache: TranslationCache | None = None,
) -> int:
    """Build <out>/<chapter>/_data.json and its media. Returns the segment count.

    A chapter whose _data.json already exists is skipped, so a run can resume.
    """
    chapter_dir = os.path.join(out_folder, str(chapter.number))
    data_path = os.path.join(chapter_dir, "_data.json")
    if os.path.exists(data_path):
        with open(data_path, encoding="utf-8") as f:
            return len(json.load(f).get("segments", []))
    os.makedirs(chapter_dir, exist_ok=True)

    ignored: list[dict] = []
    kept: list[tuple[int, Cue]] = []
    for idx, cue in enumerate(cues):
        reason = rejection_reason(cue, min_ms, max_ms)
        if reason:
            ignored.append(_ignored(idx, cue, chapter, reason))
        else:
            kept.append((idx, cue))

    translations = translate_all(translator, [cue.text for _, cue in kept], cache)
    if cache is not None:
        cache.save()

    segments: list[dict] = []
    for pos, (idx, cue) in enumerate(kept):
        en, es = translations["en"][pos], translations["es"][pos]
        missing = [lang for lang, t in (("en", en), ("es", es)) if not t]
        if missing:
            ignored.append(_ignored(idx, cue, chapter, f"missing translation: {','.join(missing)}"))
            continue

        h = segment_hash(hash_salt, asin, cue.start_ms, cue.text)
        audio_name, image_name, video_name = f"{h}.mp3", f"{h}.webp", f"{h}.mp4"
        try:
            audio_out = os.path.join(chapter_dir, audio_name)
            extract_audio(audio_path, audio_out, cue.start_ms / 1000, cue.end_ms / 1000)
            image_out = os.path.join(chapter_dir, image_name)
            shutil.copyfile(frame_path, image_out)
            build_clip(image_out, audio_out, os.path.join(chapter_dir, video_name))
        except Exception:
            logger.error(f"[{chapter.number}] media extraction failed for cue {idx}", exc_info=True)
            ignored.append(_ignored(idx, cue, chapter, "media extraction failed"))
            continue

        segments.append(
            {
                "segment_hash": h,
                "segment_index": idx,
                "start_ms": cue.start_ms - chapter.start_ms,
                "end_ms": cue.end_ms - chapter.start_ms,
                "duration_ms": cue.end_ms - cue.start_ms,
                "content_ja": cue.text,
                "content_en": en,
                "content_es": es,
                "is_mt_en": True,
                "is_mt_es": True,
                "actor_ja": None,
                "actor_en": None,
                "actor_es": None,
                "files": {"audio": audio_name, "screenshot": image_name, "video": video_name},
                "content_rating": content_rating,
                # No classifier runs on the cover; this is the backend's own "no analysis" value.
                "content_analysis": {"scores": {}, "tags": {}},
                "pos_analysis": None,
            }
        )

    ignored.sort(key=lambda s: s["segment_index"])
    atomic_write_json(
        data_path,
        {
            "metadata": {
                "version": "6",
                "number": chapter.number,
                "title": chapter.title,
                "duration_ms": chapter.end_ms - chapter.start_ms,
                "total_segments": len(segments),
            },
            "media": {"asin": asin, "media_source": "audible"},
            "segments": segments,
            "ignored_segments": ignored,
        },
    )
    return len(segments)


@dataclass
class PreparedBook:
    book: dict
    audio_path: str
    cover_path: str
    out_folder: str
    hash_salt: str
    frame_path: str
    chapters: list[Chapter]
    cues_by_chapter: dict[int, list[Cue]]


def prepare_book(book_folder: str, out_root: str, *, verify_audio: bool = True) -> PreparedBook:
    """Validate the input folder, write _info.json and the cover frame."""
    book = load_book(book_folder)
    audio_path = os.path.join(book_folder, AUDIO_FILE)
    if not os.path.exists(audio_path):
        raise BookInputError(f"Missing {AUDIO_FILE} in {book_folder}")
    cover_path = find_cover(book_folder)
    if not cover_path:
        raise BookInputError(f"Missing cover ({' / '.join(COVER_NAMES)}) in {book_folder}")
    alignment_path = os.path.join(book_folder, ALIGNMENT_FILE)
    if not os.path.exists(alignment_path):
        raise BookInputError(f"Missing {ALIGNMENT_FILE} in {book_folder}")

    with open(alignment_path, encoding="utf-8") as f:
        manifest = json.load(f)
    cues = load_cues(manifest)

    if verify_audio:
        expected = manifest["sources"]["audioFiles"][0].get("sha256")
        if expected and sha256_file(audio_path) != expected:
            raise BookInputError(
                f"{AUDIO_FILE} is not the audio this alignment was made for (sha256 mismatch)"
            )

    chapters = probe_chapters(audio_path)
    if not chapters:
        duration = manifest["sources"]["audioFiles"][0].get("durationMs") or max(
            c.end_ms for c in cues
        )
        chapters = [Chapter(1, book["japanese_name"], 0, duration)]

    out_folder = os.path.join(out_root, book["asin"])
    os.makedirs(out_folder, exist_ok=True)
    hash_salt = write_info_json(out_folder, book, cover_path)
    frame_path = prepare_frame(cover_path, out_folder)

    return PreparedBook(
        book=book,
        audio_path=audio_path,
        cover_path=cover_path,
        out_folder=out_folder,
        hash_salt=hash_salt,
        frame_path=frame_path,
        chapters=chapters,
        cues_by_chapter=assign_chapters(cues, chapters),
    )
