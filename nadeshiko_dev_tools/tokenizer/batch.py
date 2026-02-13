"""Batch tokenization of all media in a processed directory.

Walks a folder of media folders (e.g., /mnt/storage/nade-processed/),
tokenizes all Japanese content in segments, and writes pos_analysis to _data.json.
"""

import json
from pathlib import Path

from rich.console import Console

from nadeshiko_dev_tools.common.archive import discover_episodes
from nadeshiko_dev_tools.common.progress import create_progress

console = Console()


def _tokenizers_available(engine: str) -> dict:
    """Instantiate requested tokenizer(s). Returns {name: instance}."""
    from .tokenizer import JapaneseTokenizer, UnidicTokenizer

    tokenizers = {}

    if engine in ("sudachi", "all"):
        try:
            tokenizers["sudachi"] = JapaneseTokenizer()
        except ImportError:
            console.print("[yellow]Sudachi not available (install tokenizer extra)[/yellow]")
        except Exception as e:
            console.print(f"[red]Failed to load Sudachi: {e}[/red]")

    if engine in ("unidic", "all"):
        try:
            tokenizers["unidic"] = UnidicTokenizer()
        except ImportError:
            console.print("[yellow]UniDic not available (install tokenizer extra)[/yellow]")
        except Exception as e:
            console.print(f"[red]Failed to load UniDic: {e}[/red]")

    return tokenizers


def run_batch(
    root_dir: Path,
    engine: str = "all",
    resume: bool = True,
):
    """Run batch tokenization on all media folders in root_dir.

    Args:
        root_dir: Path to directory containing media folders (each must have _info.json)
        engine: Tokenizer engine to use ("sudachi", "unidic", or "all")
        resume: Skip segments that already have pos_analysis for the requested engine
    """
    root_dir = root_dir.resolve()
    console.print("[bold]Batch POS Analysis Tokenizer[/bold]")
    console.print(f"Root: {root_dir}")
    console.print(f"Engine: {engine}")
    console.print()

    # Find all media folders (must contain _info.json)
    media_dirs = sorted(
        d for d in root_dir.iterdir()
        if d.is_dir() and (d / "_info.json").exists()
    )

    if not media_dirs:
        console.print("[yellow]No media folders found (missing _info.json).[/yellow]")
        return

    console.print(f"Found {len(media_dirs)} media folder(s)")
    console.print()

    # Load tokenizers
    tokenizers = _tokenizers_available(engine)
    if not tokenizers:
        console.print("[red]No tokenizer available.[/red]")
        return

    # Show tokenizer versions
    console.print(f"[green]Loaded engines:[/green]")
    for name, tok in tokenizers.items():
        console.print(f"  [cyan]{name}:[/cyan] {tok.info}")
    console.print()

    # Count total segments to process
    total_segments = 0
    for media_dir in media_dirs:
        for ep_dir in discover_episodes(media_dir):
            data_path = ep_dir / "_data.json"
            if data_path.exists():
                try:
                    with open(data_path) as f:
                        data = json.load(f)
                    segments = data.get("segments", [])
                    for seg in segments:
                        if seg.get("content_ja"):
                            total_segments += 1
                except Exception:
                    pass

    if total_segments == 0:
        console.print("[yellow]No segments with Japanese content found.[/yellow]")
        return

    console.print(f"Total segments with Japanese content: {total_segments:,}")
    console.print()

    # Process each media folder
    total_processed = 0
    total_skipped = 0

    with create_progress(console) as progress:
        task = progress.add_task("Tokenizing", total=total_segments)

        for media_dir in media_dirs:
            media_name = media_dir.name
            media_processed = 0

            for ep_dir in discover_episodes(media_dir):
                data_path = ep_dir / "_data.json"
                if not data_path.exists():
                    continue

                try:
                    with open(data_path) as f:
                        data = json.load(f)
                except Exception:
                    continue

                modified = False
                for segment in data.get("segments", []):
                    content_ja = segment.get("content_ja", "")
                    if not content_ja:
                        progress.advance(task)
                        continue

                    # Check if already tokenized
                    pos = segment.get("pos_analysis") or {}
                    already_done = all(name in pos for name in tokenizers.keys())

                    if resume and already_done:
                        total_skipped += 1
                        progress.advance(task)
                        continue

                    # Tokenize
                    for name, tok in tokenizers.items():
                        pos[name] = tok.tokenize(content_ja)
                        # Store version info (simplified format)
                        pos[f"_tokenizer_{name}"] = tok.info

                    segment["pos_analysis"] = pos
                    total_processed += 1
                    media_processed += 1
                    modified = True
                    progress.advance(task)

                if modified:
                    with open(data_path, "w") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

            if media_processed > 0:
                console.print(f"  [cyan]{media_name}:[/cyan] {media_processed} segments")

    console.print()
    console.print("[bold green]Tokenization complete![/bold green]")
    console.print(f"  Processed: {total_processed:,}")
    console.print(f"  Skipped: {total_skipped:,}")
