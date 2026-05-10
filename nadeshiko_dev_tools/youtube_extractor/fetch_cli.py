"""YouTube subtitle fetcher — download subtitles from YouTube videos or channels.

Only videos with manual (non-auto-generated) Japanese subtitles are processed.
English and Spanish subtitles are downloaded if available manually; otherwise
they are translated from Japanese via DeepL (requires TOKEN env var).

Usage:
    # Single video
    uv run fetch-youtube https://www.youtube.com/watch?v=XXX --out /mnt/storage/yt

    # Entire channel
    uv run fetch-youtube https://www.youtube.com/@ChannelHandle --out /mnt/storage/yt
"""

import argparse
import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import deepl as deepl_lib
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from nadeshiko_dev_tools.youtube_extractor.fetch import (
    download_subtitles,
    export_cookies,
    get_channel_metadata,
    get_video_info,
    save_channel_info,
    save_video_meta,
    translate_subtitle,
)

load_dotenv()

console = Console()
logger = logging.getLogger("fetch-youtube")
handler = RichHandler(console=console, show_time=True, show_path=False, markup=True)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _process_video(meta, channel_folder: str, translator, cookies_file: str | None) -> bool:
    video_folder = os.path.join(channel_folder, meta.video_id)
    meta_path = os.path.join(video_folder, "_meta.json")

    if os.path.exists(meta_path):
        console.print("  [yellow]Already fetched, skipping[/yellow]")
        return True

    os.makedirs(video_folder, exist_ok=True)

    langs_to_download = [lang for lang in ["ja", "en", "es"] if lang in meta.available_manual_langs]
    missing_langs = [lang for lang in ["en", "es"] if lang not in meta.available_manual_langs]

    console.print(f"  Downloading: {langs_to_download}")
    downloaded = download_subtitles(meta.video_id, video_folder, langs_to_download, cookies_file)

    if "ja" not in downloaded:
        logger.error("  [red]JA subtitle download failed[/red]")
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch subtitles from a YouTube video or channel"
    )
    parser.add_argument("url", help="YouTube video or channel URL")
    parser.add_argument("--out", required=True, help="Output root directory")
    parser.add_argument(
        "--browser",
        default=None,
        metavar="BROWSER",
        help="Extract cookies from browser to bypass bot detection (e.g. chrome, firefox, brave)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not shutil.which("deno"):
        console.print(
            "[red bold]deno is required but not found in PATH.[/red bold]\n"
            "Install it from https://deno.com or via your package manager, then retry."
        )
        return 1

    deepl_token = os.getenv("TOKEN")
    translator = deepl_lib.Translator(deepl_token) if deepl_token else None
    if not translator:
        logger.warning("No DeepL token — missing EN/ES subtitles will be skipped")

    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)

    cookies_file = None
    if args.browser:
        console.print(f"[cyan]Loading cookies from {args.browser}...[/cyan]")
        cookies_file = export_cookies(args.browser)
        console.print("[green]Cookies loaded[/green]")

    try:
        console.print("[cyan]Resolving URL...[/cyan]")
        channel_id, channel_name, video_ids, avatar_url, banner_url = get_channel_metadata(
            args.url, cookies_file
        )

        if not channel_id:
            console.print("[red]Could not resolve channel ID from URL[/red]")
            return 1

        console.print(f"[green]Channel: {channel_name} ({channel_id})[/green]")
        console.print(f"[cyan]{len(video_ids)} video(s) to check[/cyan]")

        channel_folder = os.path.join(out_root, channel_id)
        os.makedirs(channel_folder, exist_ok=True)
        hash_salt = save_channel_info(
            channel_folder, channel_id, channel_name, avatar_url, banner_url
        )
        console.print(f"[green]_info.json ready (salt: {hash_salt[:8]}...)[/green]")

        valid_by_id: dict[str, object] = {}
        skipped = 0
        total = len(video_ids)

        console.print(f"[cyan]Checking {total} video(s) in parallel...[/cyan]")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(get_video_info, vid, cookies_file): vid for vid in video_ids
            }
            for done, future in enumerate(as_completed(futures), 1):
                video_id = futures[future]
                meta = future.result()
                if meta:
                    console.print(f"  [{done}/{total}] [green]✓ {meta.title}[/green]")
                    valid_by_id[video_id] = meta
                else:
                    console.print(f"  [{done}/{total}] [dim]{video_id} — no manual JA subs[/dim]")
                    skipped += 1

        valid = [valid_by_id[vid] for vid in video_ids if vid in valid_by_id]

        console.print(
            f"\n[cyan bold]{len(valid)} valid / {skipped} skipped (no manual JA subs)[/cyan bold]"
        )

        if not valid:
            console.print("[yellow]Nothing to fetch.[/yellow]")
            return 0

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
        return 0 if failed == 0 else 1
    finally:
        if cookies_file:
            Path(cookies_file).unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
