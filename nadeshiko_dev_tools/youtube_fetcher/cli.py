"""YouTube subtitle fetcher — download subtitles from YouTube videos or channels.

Only videos with manual (non-auto-generated) Japanese subtitles are processed.
English and Spanish subtitles are downloaded if available manually; otherwise
they are translated from Japanese via DeepL (requires TOKEN env var).

The check phase (does a video have manual JA subs?) is cached per channel in
``_checked.json`` so re-runs skip already-decided videos. Transient failures
(rate limits, network, bot walls) are retried with backoff and never cached, so
they are re-attempted next run instead of being silently dropped.

Usage:
    # Single video
    uv run fetch-youtube https://www.youtube.com/watch?v=XXX --out /mnt/storage/yt

    # Whole channel, newest 100 only, from a cookies file, 3 workers
    uv run fetch-youtube https://www.youtube.com/@Handle --out /mnt/storage/yt \\
        --limit 100 --cookies-file ~/yt_cookies.txt --workers 3
"""

import argparse
import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from nadeshiko_dev_tools.common.deepl import get_translator
from nadeshiko_dev_tools.youtube_fetcher.fetcher import (
    TransientFetchError,
    download_media,
    export_cookies,
    get_channel_metadata,
    get_video_info,
    load_checked,
    meta_from_cache,
    meta_to_cache_entry,
    save_channel_info,
    save_checked,
    save_video_meta,
    translate_subtitle,
    with_retries,
)

load_dotenv()

console = Console()
logger = logging.getLogger("fetch-youtube")
handler = RichHandler(console=console, show_time=True, show_path=False, markup=True)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _process_video(
    meta,
    channel_folder: str,
    translator,
    cookies_file: str | None,
) -> bool:
    video_folder = os.path.join(channel_folder, meta.video_id)
    meta_path = os.path.join(video_folder, "_meta.json")

    if os.path.exists(meta_path):
        console.print("  [yellow]Already fetched, skipping[/yellow]")
        return True

    os.makedirs(video_folder, exist_ok=True)

    langs_to_download = [lang for lang in ["ja", "en", "es"] if lang in meta.available_manual_langs]
    missing_langs = [lang for lang in ["en", "es"] if lang not in meta.available_manual_langs]

    console.print(f"  Downloading video + subs {langs_to_download}...")
    try:
        video_path, downloaded = with_retries(
            lambda: download_media(meta.video_id, video_folder, langs_to_download, cookies_file),
            label=f"  {meta.video_id} download",
        )
    except TransientFetchError:
        logger.error("  [red]Download failed after retries — will retry on next run[/red]")
        return False

    if "ja" not in downloaded:
        logger.error("  [red]JA subtitle download failed[/red]")
        return False
    if video_path is None:
        logger.error("  [red]Video download failed[/red]")
        return False

    translated_langs = []
    for lang in missing_langs:
        if not translator:
            logger.warning(f"  [yellow]No DeepL token — skipping {lang} translation[/yellow]")
            continue
        console.print(f"  Translating {lang} via DeepL...")
        try:
            translate_subtitle(downloaded["ja"], lang, translator)
            translated_langs.append(lang)
        except Exception:
            logger.error(f"  [red]DeepL translation to {lang} failed[/red]", exc_info=True)

    save_video_meta(video_folder, meta, translated_langs)
    return True


def _check_one(video_id: str, cookies_file: str | None, since: str | None):
    """Check one video for manual JA subs, retrying transient failures with backoff."""
    return with_retries(
        lambda: get_video_info(video_id, cookies_file, since),
        label=f"check {video_id}",
    )


def _normalize_since(value: str | None) -> str | None:
    """Accept YYYY-MM-DD or YYYYMMDD and return YYYYMMDD (yt-dlp's upload_date format)."""
    if not value:
        return None
    digits = value.replace("-", "")
    if len(digits) != 8 or not digits.isdigit():
        raise SystemExit(f"--since must be YYYY-MM-DD or YYYYMMDD, got: {value!r}")
    return digits


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch subtitles from a YouTube video or channel")
    parser.add_argument("url", help="YouTube video or channel URL")
    parser.add_argument("--out", required=True, help="Output root directory")
    parser.add_argument(
        "--browser",
        default=None,
        metavar="BROWSER",
        help="Extract cookies from browser to bypass bot detection (e.g. chrome, firefox, brave)",
    )
    parser.add_argument(
        "--cookies-file",
        default=None,
        metavar="PATH",
        help="Path to a Netscape-format cookies file (alternative to --browser)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only consider the newest N videos of the channel",
    )
    parser.add_argument(
        "--playlist-items",
        default=None,
        metavar="SPEC",
        help="yt-dlp item spec (e.g. '1:50,100:120'); overrides --limit",
    )
    parser.add_argument(
        "--since",
        default=None,
        metavar="DATE",
        help="Skip videos uploaded before this date (YYYY-MM-DD or YYYYMMDD)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Parallel workers for the subtitle-check phase (default 3; keep low to avoid bans)",
    )
    parser.add_argument(
        "--recheck",
        action="store_true",
        help="Ignore the _checked.json cache and re-check every video",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    since = _normalize_since(args.since)

    if not shutil.which("deno"):
        console.print(
            "[red bold]deno is required but not found in PATH.[/red bold]\n"
            "Install it from https://deno.com or via your package manager, then retry."
        )
        return 1

    translator = get_translator()
    if not translator:
        logger.warning("No translator configured — missing EN/ES subtitles will be skipped")

    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)

    # A user-supplied cookies file must not be deleted; only an exported one is temporary.
    cookies_file = None
    exported_cookies = None
    if args.cookies_file:
        cookies_file = os.path.abspath(os.path.expanduser(args.cookies_file))
        if not os.path.exists(cookies_file):
            console.print(f"[red]Cookies file not found: {cookies_file}[/red]")
            return 1
        console.print(f"[cyan]Using cookies file {cookies_file}[/cyan]")
    elif args.browser:
        console.print(f"[cyan]Loading cookies from {args.browser}...[/cyan]")
        cookies_file = exported_cookies = export_cookies(args.browser)
        console.print("[green]Cookies loaded[/green]")

    try:
        console.print("[cyan]Resolving URL...[/cyan]")
        channel_id, channel_name, video_ids, avatar_url, banner_url = get_channel_metadata(
            args.url, cookies_file, limit=args.limit, playlist_items=args.playlist_items
        )

        if not channel_id:
            console.print("[red]Could not resolve channel ID from URL[/red]")
            return 1

        console.print(f"[green]Channel: {channel_name} ({channel_id})[/green]")
        console.print(f"[cyan]{len(video_ids)} video(s) to consider[/cyan]")

        channel_folder = os.path.join(out_root, channel_id)
        os.makedirs(channel_folder, exist_ok=True)
        hash_salt = save_channel_info(
            channel_folder, channel_id, channel_name, avatar_url, banner_url
        )
        console.print(f"[green]_info.json ready (salt: {hash_salt[:8]}...)[/green]")

        # Split into cached-decisions vs videos that still need a network check.
        checked = {} if args.recheck else load_checked(channel_folder)
        valid_by_id: dict[str, object] = {}
        skipped = 0
        to_check = []
        for vid in video_ids:
            entry = checked.get(vid)
            if entry and entry.get("status") == "valid":
                valid_by_id[vid] = meta_from_cache(entry)
            elif entry and entry.get("status") == "skip":
                skipped += 1
            else:
                to_check.append(vid)

        if len(to_check) < len(video_ids):
            console.print(
                f"[dim]Cache: {len(valid_by_id)} valid, {skipped} skipped, "
                f"{len(to_check)} to check[/dim]"
            )

        errored = 0
        if to_check:
            total = len(to_check)
            console.print(
                f"[cyan]Checking {total} video(s) with {args.workers} worker(s)...[/cyan]"
            )
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(_check_one, vid, cookies_file, since): vid for vid in to_check
                }
                for done, future in enumerate(as_completed(futures), 1):
                    video_id = futures[future]
                    try:
                        meta = future.result()
                    except TransientFetchError:
                        # Not cached — retried next run rather than silently dropped.
                        console.print(
                            f"  [{done}/{total}] [red]✗ {video_id} — transient error[/red]"
                        )
                        errored += 1
                        continue
                    if meta:
                        console.print(f"  [{done}/{total}] [green]✓ {meta.title}[/green]")
                        valid_by_id[video_id] = meta
                        checked[video_id] = meta_to_cache_entry(meta)
                    else:
                        console.print(f"  [{done}/{total}] [dim]{video_id} — skip[/dim]")
                        checked[video_id] = {"status": "skip", "meta": None}
                        skipped += 1
            save_checked(channel_folder, checked)

        valid = [valid_by_id[vid] for vid in video_ids if vid in valid_by_id]

        summary = f"\n[cyan bold]{len(valid)} valid / {skipped} skipped"
        if errored:
            summary += f" / {errored} transient errors (will retry next run)"
        console.print(summary + "[/cyan bold]")

        if not valid:
            console.print("[yellow]Nothing to fetch.[/yellow]")
            return 1 if errored else 0

        success = 0
        failed = 0
        for i, meta in enumerate(valid, 1):
            console.print(f"\n[cyan bold][{i}/{len(valid)}] {meta.title}[/cyan bold]")
            ok = _process_video(meta, channel_folder, translator, cookies_file)
            if ok:
                success += 1
            else:
                failed += 1

        console.print(f"\n[green bold]Done: {success} fetched, {failed} failed[/green bold]")
        return 0 if failed == 0 and errored == 0 else 1
    finally:
        if exported_cookies:
            Path(exported_cookies).unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
