import glob
import json
import logging
import os
import secrets
from dataclasses import dataclass

import yt_dlp

from nadeshiko_dev_tools.common.file_utils import atomic_write_json
from nadeshiko_dev_tools.segment_extractor.utils.subtitle_utils import load_subtitle_file

logger = logging.getLogger(__name__)

# Silence yt-dlp's internal Python logger — all relevant output is routed through
# _YdlLogger instead. Without this, yt-dlp sub-modules (cookies, JS challenge solver,
# extractor) emit directly to the "yt_dlp" logger and bypass our filters entirely.
logging.getLogger("yt_dlp").setLevel(logging.ERROR)

DEEPL_LANG = {"en": "EN-US", "es": "ES"}

# Language prefixes matched against YouTube subtitle track keys
_LANG_PREFIXES = ["ja", "en", "es"]

# yt-dlp warning substrings that are irrelevant when only fetching subtitles
_SUPPRESSED_WARNINGS = (
    "cookie file entry",            # malformed/binary browser cookie entries (any form)
    "challenge solving failed",     # JS challenge for video formats (we don't download video)
    "signature solving failed",     # same
    "only images are available",    # community posts / image-only content
    "requested format is not available",  # format errors (we set ignore_no_formats_error)
    "javascript runtime",           # JS runtime (not needed for subtitle-only fetching)
)


@dataclass
class VideoMeta:
    video_id: str
    channel_id: str
    channel_name: str
    title: str
    duration_ms: int
    published_at: str | None
    available_manual_langs: list[str]


def _resolve_lang(subtitles: dict, prefix: str) -> str | None:
    """Return the first subtitle key matching a language prefix (e.g. 'ja' matches 'ja-JP')."""
    for key in subtitles:
        if key == prefix or key.startswith(f"{prefix}-"):
            return key
    return None


def _manual_lang_codes(info: dict) -> list[str]:
    """Return normalised language codes ('ja', 'en', 'es') that have manual subtitles."""
    subtitles = info.get("subtitles", {})
    found = []
    for prefix in _LANG_PREFIXES:
        if _resolve_lang(subtitles, prefix):
            found.append(prefix)
    return found


def _pick_thumbnail(thumbnails: list[dict], preference: list[str]) -> str | None:
    """Pick the best thumbnail URL by matching id keywords in order of preference."""
    if not thumbnails:
        return None
    for keyword in preference:
        for t in thumbnails:
            if keyword in (t.get("id") or "").lower():
                return t.get("url")
    # Fallback: largest by resolution
    sized = [t for t in thumbnails if t.get("width") and t.get("height")]
    if sized:
        return max(sized, key=lambda t: t["width"] * t["height"])["url"]
    return thumbnails[-1].get("url")


def _channel_videos_url(url: str) -> str:
    """Redirect channel/user URLs to their /videos tab to get individual video entries."""
    stripped = url.rstrip("/")
    is_channel = any(p in stripped for p in ["/@", "/channel/", "/c/", "/user/"])
    if is_channel and not stripped.endswith("/videos"):
        return stripped + "/videos"
    return url


_YT_COOKIE_DOMAINS = (".youtube.com", ".google.com", "accounts.google.com")


def export_cookies(browser: str) -> str:
    """Export YouTube-relevant browser cookies to a temp Netscape-format file.

    Only cookies for YouTube/Google domains are kept — other sites produce binary
    cookie values that pollute the log output.
    """
    import tempfile

    # Export all cookies to a raw temp file first.
    with tempfile.NamedTemporaryFile(
        suffix=".txt", delete=False, prefix="yt_cookies_raw_", mode="w"
    ) as raw:
        raw.write("# Netscape HTTP Cookie File\n")
        raw_name = raw.name

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "cookiesfrombrowser": (browser,),
        "cookiefile": raw_name,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        _ = ydl.cookiejar  # accessing the property triggers browser cookie export

    # Filter to YouTube/Google cookies only and write the final file.
    with tempfile.NamedTemporaryFile(
        suffix=".txt", delete=False, prefix="yt_cookies_", mode="w", encoding="utf-8"
    ) as out:
        out.write("# Netscape HTTP Cookie File\n")
        out_name = out.name
        try:
            with open(raw_name, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    if any(line.startswith(d) for d in _YT_COOKIE_DOMAINS):
                        out.write(line)
        finally:
            os.remove(raw_name)

    logger.info(f"Cookies exported from {browser} to {out_name} (YouTube/Google only)")
    return out_name


class _YdlLogger:
    """Route yt-dlp log output and suppress noise irrelevant to subtitle-only fetching."""

    def debug(self, msg: str) -> None:
        if msg.startswith("[debug]"):
            logger.debug(msg)

    def info(self, msg: str) -> None:
        logger.debug(msg)

    def warning(self, msg: str) -> None:
        if "\x00" in msg:
            return
        msg_lower = msg.lower()
        if not any(pat in msg_lower for pat in _SUPPRESSED_WARNINGS):
            logger.warning(msg)

    def error(self, msg: str) -> None:
        logger.error(msg)


def _base_ydl_opts(cookies_file: str | None = None) -> dict:
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignore_no_formats_error": True,
        "logger": _YdlLogger(),
    }
    if cookies_file:
        opts["cookiefile"] = cookies_file
    return opts


def _fetch_channel_thumbnails(
    channel_id: str, cookies_file: str | None = None
) -> tuple[str | None, str | None]:
    """Resolve avatar + banner for a channel by hitting its main page."""
    if not channel_id:
        return None, None
    url = f"https://www.youtube.com/channel/{channel_id}"
    ydl_opts = {**_base_ydl_opts(cookies_file), "extract_flat": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        logger.warning(f"Could not resolve channel thumbnails for {channel_id}: {e}")
        return None, None
    thumbnails = info.get("thumbnails", [])
    avatar = _pick_thumbnail(thumbnails, ["avatar_uncropped", "avatar"])
    banner = _pick_thumbnail(thumbnails, ["banner_uncropped", "banner"])
    return avatar, banner


def get_channel_metadata(
    url: str, cookies_file: str | None = None
) -> tuple[str, str, list[str], str | None, str | None]:
    """Return (channel_id, channel_name, [video_ids], avatar_url, banner_url).

    Uses extract_flat to avoid fetching per-video subtitle info at this stage.
    For single-video URLs, avatar/banner are resolved against the channel
    page so they always reflect the channel (not the video thumbnail).
    """
    resolved_url = _channel_videos_url(url)

    ydl_opts = {**_base_ydl_opts(cookies_file), "extract_flat": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(resolved_url, download=False)

    entries = info.get("entries")
    if entries:
        # Channel / playlist — info already contains channel-level thumbnails
        thumbnails = info.get("thumbnails", [])
        avatar_url = _pick_thumbnail(thumbnails, ["avatar_uncropped", "avatar"])
        banner_url = _pick_thumbnail(thumbnails, ["banner_uncropped", "banner"])
        channel_id = info.get("channel_id") or info.get("id", "")
        channel_name = info.get("channel") or info.get("uploader") or info.get("title", "")
        video_ids = [
            e["id"] for e in entries
            if e and "id" in e and not e["id"].startswith("UC") and len(e["id"]) == 11
        ]
    else:
        # Single video — info has the video's thumbnails, not the channel's.
        # Hit the channel page directly to get avatar + banner.
        channel_id = info.get("channel_id", "")
        channel_name = info.get("channel") or info.get("uploader", "")
        video_ids = [info["id"]]
        avatar_url, banner_url = _fetch_channel_thumbnails(channel_id, cookies_file)

    return channel_id, channel_name, video_ids, avatar_url, banner_url


def get_video_info(video_id: str, cookies_file: str | None = None) -> VideoMeta | None:
    """Fetch full metadata for a single video.

    Returns None if the video is unavailable or has no manual Japanese subtitles.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = _base_ydl_opts(cookies_file)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        logger.debug(f"Video {video_id} unavailable: {e}")
        return None

    manual_langs = _manual_lang_codes(info)
    if "ja" not in manual_langs:
        return None

    upload_date = info.get("upload_date")  # YYYYMMDD
    published_at = (
        f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}" if upload_date else None
    )

    return VideoMeta(
        video_id=info["id"],
        channel_id=info.get("channel_id", ""),
        channel_name=info.get("channel") or info.get("uploader", ""),
        title=info.get("title", ""),
        duration_ms=int(info.get("duration", 0) * 1000),
        published_at=published_at,
        available_manual_langs=manual_langs,
    )


def download_subtitles(
    video_id: str, video_folder: str, langs: list[str], cookies_file: str | None = None
) -> dict[str, str]:
    """Download manual subtitles for the given video. Returns {lang: filepath}."""
    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        **_base_ydl_opts(cookies_file),
        "writesubtitles": True,
        "subtitleslangs": langs,
        "subtitlesformat": "vtt",
        "outtmpl": os.path.join(video_folder, f"{video_id}.%(ext)s"),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    result = {}
    for lang in langs:
        # yt-dlp names files like {video_id}.ja.vtt or {video_id}.ja-JP.vtt —
        # the wildcard pattern covers both plain and variant codes.
        matches = glob.glob(os.path.join(video_folder, f"{video_id}*.{lang}*.vtt"))
        if not matches:
            logger.warning(f"Subtitle file not found for lang={lang} video={video_id}")
            continue

        src = matches[0]
        dst = os.path.join(video_folder, f"subs.{lang}.vtt")
        if src != dst:
            os.rename(src, dst)
        result[lang] = dst

    return result


def translate_subtitle(src_path: str, target_lang: str, translator) -> str:
    """Translate a VTT subtitle file to target_lang using DeepL. Returns output path."""
    subs = load_subtitle_file(src_path)

    texts = [event.text for event in subs]
    non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]

    if not non_empty:
        logger.warning(f"No text found in {src_path}, skipping translation")
        return src_path

    indices, batch = zip(*non_empty, strict=True)
    deepl_lang = DEEPL_LANG.get(target_lang, target_lang.upper())

    # DeepL accepts lists; translate in one call (library handles chunking internally)
    results = translator.translate_text(list(batch), target_lang=deepl_lang)

    translated = list(texts)
    for i, result in zip(indices, results, strict=True):
        translated[i] = result.text

    for event, text in zip(subs, translated, strict=True):
        event.text = text

    out_path = os.path.join(os.path.dirname(src_path), f"subs.{target_lang}.vtt")
    subs.save(out_path)
    return out_path


def save_channel_info(
    channel_folder: str,
    channel_id: str,
    channel_name: str,
    avatar_url: str | None = None,
    banner_url: str | None = None,
) -> str:
    """Save _info.json for a YouTube channel. Returns the hash_salt."""
    from nadeshiko_dev_tools.common.file_utils import download_and_save_image

    info_path = os.path.join(channel_folder, "_info.json")

    if os.path.exists(info_path):
        with open(info_path) as f:
            existing = json.load(f)
        return existing["hash_salt"]

    hash_salt = secrets.token_hex(16)
    info: dict = {
        "category": "YOUTUBE_CHANNEL",
        "version": "6",
        "media_source": "youtube",
        "channel_id": channel_id,
        "name": channel_name,
        "hash_salt": hash_salt,
    }

    if avatar_url:
        try:
            info["cover"] = download_and_save_image(avatar_url, channel_folder, "cover")
        except Exception:
            logger.warning("Failed to download channel avatar", exc_info=True)

    if banner_url:
        try:
            info["banner"] = download_and_save_image(banner_url, channel_folder, "banner")
        except Exception:
            logger.warning("Failed to download channel banner", exc_info=True)

    atomic_write_json(info_path, info)

    return hash_salt


def save_video_meta(video_folder: str, meta: VideoMeta, translated_langs: list[str]) -> None:
    """Save _meta.json with video-level metadata consumed by process-media."""
    data = {
        "video_id": meta.video_id,
        "channel_id": meta.channel_id,
        "title": meta.title,
        "published_at": meta.published_at,
        "duration_ms": meta.duration_ms,
        "translated_langs": translated_langs,
    }
    atomic_write_json(os.path.join(video_folder, "_meta.json"), data)
