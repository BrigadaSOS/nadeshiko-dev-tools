"""CLI entry point for Japanese tokenization."""

import argparse
import json
import os
import sys
from pathlib import Path

from rich.console import Console

console = Console()

ENGINES = ("sudachi", "unidic", "all")

# Latest UniDic-CWJ download info
UNIDIC_LATEST_VERSION = "2025-12"
UNIDIC_DOWNLOAD_URL = "https://clrd.ninjal.ac.jp/unidic/download.html"


def _make_tokenizers(engine: str) -> dict:
    """Instantiate requested tokenizer(s). Returns {name: instance}."""
    tokenizers = {}

    if engine in ("sudachi", "all"):
        try:
            from .tokenizer import JapaneseTokenizer
            tokenizers["sudachi"] = JapaneseTokenizer()
        except ImportError:
            console.print(
                "[yellow]Sudachi not available"
                " (install tokenizer extra)[/yellow]"
            )
        except Exception as e:
            console.print(f"[red]Failed to load Sudachi: {e}[/red]")

    if engine in ("unidic", "all"):
        try:
            from .tokenizer import UnidicTokenizer
            tokenizers["unidic"] = UnidicTokenizer()
        except ImportError as e:
            console.print(
                "[yellow]UniDic not available"
                " (install tokenizer extra)[/yellow]"
            )
        except RuntimeError as e:
            console.print(f"[yellow]UniDic: {e}[/yellow]")
        except Exception as e:
            console.print(f"[red]Failed to load UniDic: {e}[/red]")

    return tokenizers


def _check_tokenizers() -> None:
    """Check and display available tokenizer versions."""
    from .tokenizer import get_available_tokenizers

    console.print("[bold]Japanese Tokenizer Status[/bold]\n")

    info = get_available_tokenizers()

    for engine_name in ("sudachi", "unidic"):
        engine_info = info.get(engine_name, {})
        if "error" in engine_info:
            console.print(f"[red]{engine_name.upper()}:[/red] {engine_info['error']}")
            continue

        # Format is now a string like "lib0.6.10.dic2026-01-16" or "dic2025-12"
        version_str = engine_info.get("version", "unknown")
        console.print(f"[cyan]{engine_name.upper()}:[/cyan] {version_str}")

        # Check if UniDic is outdated
        if engine_name == "unidic" and version_str != "unknown":
            # Extract version from "dic2025-12" format
            if version_str.startswith("dic"):
                version = version_str[3:]  # Remove "dic" prefix
                if version < UNIDIC_LATEST_VERSION:
                    console.print(
                        f"  [yellow]⚠ Outdated ({version} < {UNIDIC_LATEST_VERSION})[/yellow]"
                    )

    console.print()
    console.print("[bold]Environment:[/bold]")
    unidic_dir = os.environ.get("UNIDIC_DIR")
    if unidic_dir:
        console.print(f"  UNIDIC_DIR: {unidic_dir}")
    else:
        console.print(f"  UNIDIC_DIR: [dim](not set, using bundled)[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Japanese text tokenizer (sudachipy + fugashi/UniDic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check available tokenizers and versions
  %(prog)s check

  # Tokenize a single string (both engines)
  %(prog)s test "彼女は学校に行く"

  # Tokenize with only UniDic
  %(prog)s test "彼女は学校に行く" --engine unidic

  # Tokenize all segments in a media folder
  %(prog)s tokenize /mnt/storage/nade-processed/132126

  # Batch process all media folders
  %(prog)s batch /mnt/storage/nade-processed

Setting up latest UniDic:
  1. Download UniDic-CWJ 2025.12 from: https://clrd.ninjal.ac.jp/unidic/download.html
  2. Extract the zip file
  3. Set environment variable: export UNIDIC_DIR="/path/to/unidic-cwj-2025.12"

Or the tokenizer will auto-download and setup the latest UniDic on first run.
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- check subcommand --
    subparsers.add_parser(
        "check",
        help="Check available tokenizers and their versions",
    )

    # -- test subcommand --
    test_parser = subparsers.add_parser(
        "test",
        help="Tokenize a single string and print results",
    )
    test_parser.add_argument("text", help="Japanese text to tokenize")
    test_parser.add_argument(
        "--engine", choices=ENGINES, default="all",
        help="Tokenizer engine to use (default: all)",
    )

    # -- tokenize subcommand --
    tokenize_parser = subparsers.add_parser(
        "tokenize",
        help="Tokenize all segments in a media folder and write pos_analysis to _data.json",
    )
    tokenize_parser.add_argument(
        "media_folder",
        type=Path,
        help="Path to media folder (must contain _info.json)",
    )
    tokenize_parser.add_argument(
        "--engine", choices=ENGINES, default="all",
        help="Tokenizer engine to use (default: all)",
    )

    # -- batch subcommand --
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch tokenize all media folders in a directory",
    )
    batch_parser.add_argument(
        "root_dir",
        type=Path,
        help="Path to directory containing media folders",
    )
    batch_parser.add_argument(
        "--engine", choices=ENGINES, default="all",
        help="Tokenizer engine to use (default: all)",
    )
    batch_parser.add_argument(
        "--no-resume", action="store_true",
        help="Reprocess segments that already have pos_analysis",
    )

    args = parser.parse_args()

    if args.command == "check":
        _check_tokenizers()
    elif args.command == "test":
        _test(args.text, args.engine)
    elif args.command == "tokenize":
        _tokenize_media(args.media_folder.resolve(), args.engine)
    elif args.command == "batch":
        _run_batch(args.root_dir.resolve(), args.engine, not args.no_resume)


def _test(text: str, engine: str) -> None:
    """Tokenize a single string and print results."""
    tokenizers = _make_tokenizers(engine)
    if not tokenizers:
        console.print("[red]No tokenizer available.[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Input:[/bold] {text}\n")

    result = {}
    for name, tok in tokenizers.items():
        tokens = tok.tokenize(text)
        result[name] = tokens
        console.print(f"[bold cyan]{name}:[/bold cyan]")
        console.print(json.dumps(tokens, ensure_ascii=False, indent=2))
        console.print()


def _tokenize_media(media_folder: Path, engine: str) -> None:
    """Tokenize all segments in a media folder, writing pos_analysis to _data.json."""
    if not (media_folder / "_info.json").exists():
        console.print(
            f"[red]Not a valid media folder (missing _info.json):"
            f" {media_folder}[/red]"
        )
        sys.exit(1)

    tokenizers = _make_tokenizers(engine)
    if not tokenizers:
        console.print("[red]No tokenizer available.[/red]")
        sys.exit(1)

    # Show tokenizer versions
    console.print("[bold]Tokenizers:[/bold]")
    for name, tok in tokenizers.items():
        console.print(f"  [cyan]{name}:[/cyan] {tok.info}")
    console.print()

    episode_folders = sorted(
        d for d in media_folder.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )

    total_segments = 0
    total_tokenized = 0

    for episode_folder in episode_folders:
        try:
            int(episode_folder.name)
        except ValueError:
            continue

        data_path = episode_folder / "_data.json"
        if not data_path.exists():
            continue

        with open(data_path) as f:
            data = json.load(f)

        modified = False
        for segment in data.get("segments", []):
            total_segments += 1
            content_ja = segment.get("content_ja", "")
            if not content_ja:
                continue

            pos = segment.get("pos_analysis") or {}

            # Store tokenizer version info (simplified format)
            for name, tok in tokenizers.items():
                pos[name] = tok.tokenize(content_ja)
                pos[f"_tokenizer_{name}"] = tok.info

            segment["pos_analysis"] = pos
            total_tokenized += 1
            modified = True

        if modified:
            with open(data_path, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            count = sum(
                1 for s in data.get("segments", [])
                if s.get("pos_analysis")
            )
            console.print(
                f"  [green]E{episode_folder.name}:[/green]"
                f" tokenized {count} segments"
            )

    console.print(
        f"\n[bold green]Done:[/bold green]"
        f" {total_tokenized}/{total_segments} segments tokenized"
    )


def _run_batch(root_dir: Path, engine: str, resume: bool) -> None:
    """Run batch tokenization on all media folders in root_dir."""
    from .batch import run_batch
    run_batch(root_dir, engine=engine, resume=resume)


if __name__ == "__main__":
    main()
