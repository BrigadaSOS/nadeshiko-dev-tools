"""process-audiobook — turn a Honomiya alignment + m4b into _data.json segments.

Usage:
    uv run process-audiobook ./books/oregairu-1 --output ./output
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from nadeshiko_dev_tools.audiobook_segment_extractor.extractor import (
    DEFAULT_MAX_DURATION_MS,
    DEFAULT_MIN_DURATION_MS,
    TRANSLATIONS_FILE,
    BookInputError,
    Chapter,
    PreparedBook,
    TranslationCache,
    prepare_book,
    process_chapter,
)
from nadeshiko_dev_tools.common.quality_check import run_qc
from nadeshiko_dev_tools.common.translator import get_translator

load_dotenv()

console = Console()
logger = logging.getLogger("process-audiobook")
handler = RichHandler(console=console, show_time=True, show_path=False, markup=True)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def parse_chapter_selection(value: str) -> set[int]:
    """'2-4,7' → {2, 3, 4, 7}."""
    wanted: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            wanted.update(range(lo, hi + 1))
        else:
            wanted.add(int(part))
    return wanted


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Turn a Honomiya read-listen alignment and its m4b into segment data"
    )
    parser.add_argument(
        "book_folder",
        help="Folder with audio.m4b, alignment.json, cover.* and book.json",
    )
    parser.add_argument("--output", default="./output", help="Output root (default: ./output)")
    parser.add_argument(
        "--chapters",
        help="Chapters to process (default: all): numbers and ranges, e.g. 2-10 or 2,3,5",
    )
    parser.add_argument("--min-duration-ms", type=int, default=DEFAULT_MIN_DURATION_MS)
    parser.add_argument("--max-duration-ms", type=int, default=DEFAULT_MAX_DURATION_MS)
    parser.add_argument(
        "--skip-audio-check",
        action="store_true",
        help="Don't verify that audio.m4b's sha256 matches the alignment",
    )
    parser.add_argument(
        "--reuse-translations",
        metavar="DIR",
        action="append",
        default=[],
        help="Earlier process-audiobook output of the same book whose EN/ES are reused for "
        "identical sentences (repeatable), so re-cutting doesn't pay for translation again",
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="Make no translation calls: use only cached/reused translations; sentences "
        "without one go to ignored_segments",
    )
    return parser.parse_args(argv)


def _run_chapters(
    prepared: PreparedBook, chapters: list[Chapter], translator, args, cache: TranslationCache
) -> dict[int, int | None]:
    stats: dict[int, int | None] = {}
    for chapter in chapters:
        n = chapter.number
        cues = prepared.cues_by_chapter.get(n, [])
        console.print(f"\n[cyan bold]Chapter {n}: {chapter.title} ({len(cues)} cues)[/cyan bold]")
        try:
            stats[n] = process_chapter(
                chapter,
                cues,
                audio_path=prepared.audio_path,
                frame_path=prepared.frame_path,
                out_folder=prepared.out_folder,
                asin=prepared.book["asin"],
                hash_salt=prepared.hash_salt,
                translator=translator,
                min_ms=args.min_duration_ms,
                max_ms=args.max_duration_ms,
                content_rating=prepared.book["content_rating"],
                cache=cache,
            )
            console.print(f"[green]Chapter {n}: {stats[n]} segments[/green]")
        except Exception:
            logger.error(f"[red]Chapter {n} failed[/red]", exc_info=True)
            stats[n] = None
    return stats


def main(argv=None):
    args = parse_args(argv)

    if args.no_translate:
        translator = None
        console.print("[yellow]--no-translate: only cached translations are used[/yellow]")
    else:
        translator = get_translator()
        if translator is None:
            console.print(
                "[red]OPENAI_API_KEY is not set — audiobooks need EN/ES translation "
                "(or pass --no-translate with --reuse-translations)[/red]"
            )
            return 1

    try:
        console.print("[cyan]Checking input (hashing the m4b can take a few seconds)...[/cyan]")
        prepared = prepare_book(
            os.path.abspath(args.book_folder),
            os.path.abspath(args.output),
            verify_audio=not args.skip_audio_check,
        )
    except BookInputError as e:
        console.print(f"[red]{e}[/red]")
        return 1

    wanted = None
    if args.chapters:
        wanted = parse_chapter_selection(args.chapters)
        unknown = wanted - {ch.number for ch in prepared.chapters}
        if unknown:
            console.print(
                f"[red]Unknown chapter(s): {sorted(unknown)}; "
                f"book has 1-{len(prepared.chapters)}[/red]"
            )
            return 1
    cache = TranslationCache(os.path.join(prepared.out_folder, TRANSLATIONS_FILE))
    for folder in args.reuse_translations:
        added = cache.seed_from_output(os.path.abspath(folder))
        console.print(f"[cyan]Reusing {added} translations from {folder}[/cyan]")
    cache.save()

    # Chapter numbers always follow the m4b, whatever is selected.
    chapters = [ch for ch in prepared.chapters if wanted is None or ch.number in wanted]

    console.print(
        f"[green]{prepared.book['japanese_name']}[/green] — {len(prepared.chapters)} chapters, "
        f"{sum(len(c) for c in prepared.cues_by_chapter.values())} cues → {prepared.out_folder}"
    )

    # Validate on the first chapter that has cues before committing to the rest.
    first = next((ch for ch in chapters if prepared.cues_by_chapter.get(ch.number)), None)
    if first is None:
        console.print("[red]No cues in the selected chapters[/red]")
        return 1
    stats = _run_chapters(prepared, [first], translator, args, cache)
    if not stats[first.number]:
        console.print(
            f"[red bold]Chapter {first.number} produced no segments — stopping.[/red bold]"
        )
        return 1
    rest = [ch for ch in chapters if ch.number != first.number]
    stats.update(_run_chapters(prepared, rest, translator, args, cache))

    console.print("\n[green bold]Summary[/green bold]")
    for n, count in stats.items():
        console.print(f"  {n}: {count if count is not None else '[red]FAILED[/red]'}")

    usage = getattr(translator, "usage", None)
    if usage:
        console.print(
            f"[cyan]Translation: {usage['requests']} requests, "
            f"{usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion tokens"
            "[/cyan]"
        )

    console.print("\n[magenta bold]QC all segments[/magenta bold]")
    passed = run_qc(prepared.out_folder, checks={"segments"}).summary()
    failed = any(c is None for c in stats.values())
    return 0 if passed and not failed else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
