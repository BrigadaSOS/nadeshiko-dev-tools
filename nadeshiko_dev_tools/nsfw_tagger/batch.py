"""Batch content rating classification of processed segment images.

Walks a media folder, classifies all WebP screenshots,
and writes results as JSON manifests.

Results are stored alongside the media:
  {media_folder}/_nsfw_results/
    results.json    — per-segment classification results
    summary.json    — aggregate statistics
"""

import json
import os
import time
from pathlib import Path

from rich.console import Console

from nadeshiko_dev_tools.common.archive import discover_episodes, discover_files
from nadeshiko_dev_tools.common.progress import create_progress

from .classifier import ClassificationResult, WDTagger

console = Console()

RESULTS_DIR_NAME = "_nsfw_results"
RESULTS_FILE = "results.json"
DEFAULT_BATCH_SIZE = 128
OOM_PATTERNS = (
    "failed to allocate memory",
    "bfcarena::allocaterawinternal",
    "cuda out of memory",
    "cublas_status_alloc_failed",
    "std::bad_alloc",
)


def _is_oom_error(exc: Exception) -> bool:
    """Return True if an exception looks like an ONNX/CUDA OOM."""
    error_text = str(exc).lower()
    return any(pattern in error_text for pattern in OOM_PATTERNS)


def _resolve_batch_size(batch_size: int | None) -> int:
    """Resolve effective batch size from CLI arg/env/default."""
    if batch_size is not None:
        if batch_size < 1:
            raise ValueError("Batch size must be >= 1")
        return batch_size

    env_value = os.getenv("NSFW_TAGGER_BATCH_SIZE", "").strip()
    if not env_value:
        return DEFAULT_BATCH_SIZE

    try:
        parsed = int(env_value)
    except ValueError:
        console.print(
            f"[yellow]Invalid NSFW_TAGGER_BATCH_SIZE={env_value!r}; "
            f"using default {DEFAULT_BATCH_SIZE}.[/yellow]"
        )
        return DEFAULT_BATCH_SIZE

    if parsed < 1:
        console.print(
            f"[yellow]NSFW_TAGGER_BATCH_SIZE must be >= 1 "
            f"(got {parsed}); using default {DEFAULT_BATCH_SIZE}.[/yellow]"
        )
        return DEFAULT_BATCH_SIZE

    return parsed


def run_batch_all(
    root_dir: Path,
    resume: bool = True,
    batch_size: int | None = None,
):
    """Run batch classification on all media folders in root_dir.

    Args:
        root_dir: Path to directory containing media folders (each must have _info.json)
        resume: Skip if results file already exists.
    """
    root_dir = root_dir.resolve()
    effective_batch_size = _resolve_batch_size(batch_size)
    console.print("[bold]Batch Content Rating Classifier[/bold]")
    console.print(f"Root: {root_dir}")
    console.print(f"Initial batch size: {effective_batch_size}")
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

    # Count total images to process
    total_images = 0
    for media_dir in media_dirs:
        for ep_dir in discover_episodes(media_dir):
            screenshots = discover_files(ep_dir, "*.webp")
            total_images += len(screenshots)

    if total_images == 0:
        console.print("[yellow]No images found.[/yellow]")
        return

    console.print(f"Total images to process: {total_images:,}")
    console.print()

    # Load model once
    console.print("[bold]Loading WD Tagger v3 model...[/bold]")
    tagger = WDTagger()
    console.print("[green]Model loaded.[/green]")
    console.print()

    start_time = time.time()

    # Process each media folder
    processed_count = 0
    skipped_count = 0

    with create_progress(console) as progress:
        overall_task = progress.add_task("Classifying", total=total_images)

        for media_dir in media_dirs:
            results_dir = media_dir / RESULTS_DIR_NAME
            result_file = results_dir / RESULTS_FILE

            # Check if already processed
            if resume and result_file.exists():
                skipped_count += 1
                # Still need to advance progress for images in this folder
                for ep_dir in discover_episodes(media_dir):
                    screenshots = discover_files(ep_dir, "*.webp")
                    progress.advance(overall_task, len(screenshots))
                continue

            # Create results dir
            results_dir.mkdir(exist_ok=True)

            # Classify this media
            media_result = classify_media(
                tagger,
                media_dir,
                progress,
                overall_task,
                batch_size=effective_batch_size,
            )

            # Save results
            with open(result_file, "w") as f:
                json.dump(media_result, f, separators=(",", ":"))

            # Compute and save summary
            by_rating: dict[str, int] = {
                "SAFE": 0,
                "SUGGESTIVE": 0,
                "QUESTIONABLE": 0,
                "EXPLICIT": 0,
            }
            for ep_data in media_result.values():
                for seg_data in ep_data.values():
                    cr = seg_data["content_rating"]
                    by_rating[cr] = by_rating.get(cr, 0) + 1

            summary = {
                "total_images": sum(by_rating.values()),
                "by_content_rating": by_rating,
            }
            with open(results_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            processed_count += 1
            console.print(f"  [cyan]{media_dir.name}:[/cyan] {summary['total_images']} images")

    elapsed = round(time.time() - start_time, 1)

    console.print()
    console.print("[bold green]Classification complete![/bold green]")
    console.print(f"  Processed: {processed_count}")
    console.print(f"  Skipped: {skipped_count}")
    console.print(f"  Time: {elapsed:.0f}s")


def classify_media(
    tagger: WDTagger,
    media_dir: Path,
    progress,
    task_id,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """Classify all screenshots for a single media.

    Returns a dict mapping episode -> { hashed_id -> classification }.
    """
    episodes = discover_episodes(media_dir)
    media_result = {}

    effective_batch_size = batch_size

    for ep_dir in episodes:
        screenshots = discover_files(ep_dir, "*.webp")
        if not screenshots:
            continue

        ep_num = ep_dir.name
        ep_results = {}
        cursor = 0

        while cursor < len(screenshots):
            batch_end = min(cursor + effective_batch_size, len(screenshots))
            batch_paths = screenshots[cursor:batch_end]

            try:
                results = tagger.classify_batch(batch_paths)
            except Exception as e:
                if _is_oom_error(e):
                    if effective_batch_size > 1:
                        new_batch_size = max(1, effective_batch_size // 2)
                        console.print(
                            f"[yellow]OOM in {ep_dir}; reducing batch size "
                            f"{effective_batch_size} -> {new_batch_size}[/yellow]"
                        )
                        effective_batch_size = new_batch_size
                        continue

                    # Already at batch size 1; skip this image and continue.
                    failing_path = batch_paths[0]
                    console.print(
                        f"[red]OOM classifying {failing_path}; skipping image.[/red]"
                    )
                    progress.advance(task_id, 1)
                    cursor += 1
                    continue

                console.print(
                    f"[red]Error classifying batch in {ep_dir}: {e}[/red]"
                )
                progress.advance(task_id, len(batch_paths))
                cursor = batch_end
                continue

            for path, result in zip(batch_paths, results, strict=True):
                hashed_id = path.stem  # filename without .webp
                ep_results[hashed_id] = _result_to_dict(result)

            progress.advance(task_id, len(batch_paths))
            cursor = batch_end

        if ep_results:
            media_result[ep_num] = ep_results

    return media_result


def _result_to_dict(result: ClassificationResult) -> dict:
    """Convert classification result to a compact JSON-serializable dict."""
    return {
        "content_rating": result.content_rating,
        "scores": result.rating_scores,
        "tags": result.tags,
    }


def run_batch(
    media_folder: Path,
    resume: bool = True,
    batch_size: int | None = None,
):
    """Run batch classification on a single media folder.

    Args:
        media_folder: Path to media folder (must contain _info.json)
        resume: Skip if results file already exists.
    """
    results_dir = media_folder / RESULTS_DIR_NAME
    results_dir.mkdir(exist_ok=True)
    effective_batch_size = _resolve_batch_size(batch_size)

    console.print("[bold]Content Rating Batch Classifier[/bold]")
    console.print(f"Media: {media_folder}")
    console.print(f"Results: {results_dir}")
    console.print(f"Initial batch size: {effective_batch_size}")
    console.print()

    # Check if already processed
    result_file = results_dir / RESULTS_FILE
    if resume and result_file.exists():
        console.print(
            "[green]Already processed (use --no-resume to reprocess).[/green]"
        )
        return

    # Discover episodes and count images
    episodes = discover_episodes(media_folder)
    if not episodes:
        console.print("[yellow]No episode folders found.[/yellow]")
        return

    total_images = 0
    for ep_dir in episodes:
        ep_images = len(discover_files(ep_dir, "*.webp"))
        total_images += ep_images
        if ep_images:
            console.print(f"  ep {ep_dir.name}: {ep_images} images")

    if total_images == 0:
        console.print("[yellow]No images found.[/yellow]")
        return

    console.print()
    console.print(f"Episodes: {len(episodes)}")
    console.print(f"Total images: {total_images:,}")
    console.print()

    # Load model
    console.print("[bold]Loading WD Tagger v3 model...[/bold]")
    tagger = WDTagger()
    console.print("[green]Model loaded.[/green]")
    console.print()

    start_time = time.time()

    with create_progress(console) as progress:
        overall_task = progress.add_task("Classifying", total=total_images)
        media_result = classify_media(
            tagger,
            media_folder,
            progress,
            overall_task,
            batch_size=effective_batch_size,
        )

    # Save results
    with open(result_file, "w") as f:
        json.dump(media_result, f, separators=(",", ":"))

    # Compute stats
    by_rating: dict[str, int] = {"SAFE": 0, "SUGGESTIVE": 0, "QUESTIONABLE": 0, "EXPLICIT": 0}
    for ep_num, ep_data in sorted(
        media_result.items(), key=lambda x: int(x[0])
    ):
        ep_ratings: dict[str, int] = {}
        for seg_data in ep_data.values():
            cr = seg_data["content_rating"]
            by_rating[cr] = by_rating.get(cr, 0) + 1
            ep_ratings[cr] = ep_ratings.get(cr, 0) + 1
        non_safe = {k: v for k, v in ep_ratings.items() if k != "SAFE"}
        suffix = f"  [yellow]{non_safe}[/yellow]" if non_safe else ""
        console.print(f"  ep {ep_num}: {len(ep_data)} images{suffix}")

    elapsed = round(time.time() - start_time, 1)

    # Save summary
    summary = {
        "total_images": sum(by_rating.values()),
        "by_content_rating": by_rating,
        "elapsed_seconds": elapsed,
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    console.print()
    console.print("[bold green]Classification complete![/bold green]")
    console.print(f"  Images processed: {summary['total_images']:,}")
    console.print(f"  Content ratings: {by_rating}")
    console.print(f"  Time: {elapsed:.0f}s")
    console.print(f"  Results: {result_file}")
