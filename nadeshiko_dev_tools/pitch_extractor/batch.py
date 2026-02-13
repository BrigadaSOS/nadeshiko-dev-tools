"""Batch F0 pitch contour extraction from processed segment audio.

Walks a media folder, extracts F0 contours from MP3 segments,
and writes results as _pitch.json files per episode.
"""

import json
import time
from pathlib import Path

import numpy as np
from rich.console import Console

from nadeshiko_dev_tools.common.archive import discover_episodes, discover_files
from nadeshiko_dev_tools.common.progress import create_progress

from .extractor import decode_mp3, extract_f0

console = Console()

PITCH_FILE = "_pitch.json"


def process_episode(
    episode_dir: Path,
    progress,
    task_id,
    *,
    demucs_model=None,
    demucs_device: str | None = None,
    save_vocals: bool = False,
    time_step: float = 0.01,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0,
) -> dict | None:
    """Extract F0 for all segments in an episode.

    Returns the pitch data dict, or None if no segments found.
    """
    segments = discover_files(episode_dir, "*.mp3")
    if not segments:
        return None

    use_separation = demucs_model is not None
    segments_data = {}

    for seg_path in segments:
        seg_hash = seg_path.stem
        try:
            audio_wav, sample_rate = decode_mp3(seg_path)

            if use_separation:
                from .separator import separate_vocals

                vocals = separate_vocals(demucs_model, demucs_device, audio_wav, sample_rate)

                if save_vocals:
                    vocals_dir = episode_dir / "_vocals"
                    vocals_dir.mkdir(exist_ok=True)
                    _save_wav(vocals, sample_rate, vocals_dir / f"{seg_hash}.wav")

                audio_wav = vocals

            f0 = extract_f0(
                audio_wav, sample_rate,
                time_step=time_step,
                pitch_floor=pitch_floor,
                pitch_ceiling=pitch_ceiling,
            )
            sample_ms = int(time_step * 1000)
            segments_data[seg_hash] = {"f0": f0, "sample_ms": sample_ms}

        except Exception as e:
            console.print(f"[red]Error processing {seg_path.name}: {e}[/red]")

        progress.advance(task_id)

    return segments_data if segments_data else None


def _save_wav(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    """Save a numpy waveform as a WAV file."""
    import wave

    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def run_batch(
    media_folder: Path,
    resume: bool = True,
    separation: bool = True,
    save_vocals: bool = False,
    sample_ms: int = 10,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0,
):
    """Run batch F0 extraction on a single media folder."""
    console.print("[bold]F0 Pitch Contour Extractor[/bold]")
    console.print(f"Media: {media_folder}")
    console.print(f"Separation: {'ON' if separation else 'OFF'}")
    console.print()

    # Discover episodes
    all_episodes = discover_episodes(media_folder)
    if not all_episodes:
        console.print("[yellow]No episode folders found.[/yellow]")
        return

    # Filter by resume and collect pending episodes
    pending: list[Path] = []
    skipped = 0
    for ep_dir in all_episodes:
        if resume and (ep_dir / PITCH_FILE).exists():
            skipped += 1
            continue
        if discover_files(ep_dir, "*.mp3"):
            pending.append(ep_dir)

    console.print(
        f"Episodes: {len(all_episodes)} total,"
        f" {len(pending)} to process"
        + (f", {skipped} done" if skipped else "")
    )

    if not pending:
        console.print("[green]All episodes already processed![/green]")
        return

    # Count total segments
    total_segments = sum(
        len(discover_files(ep_dir, "*.mp3")) for ep_dir in pending
    )

    if resume and skipped:
        console.print(
            f"[dim]Skipped episodes with existing {PITCH_FILE}"
            f" (use --no-resume to reprocess)[/dim]"
        )

    console.print(f"Total segments: {total_segments:,}")
    console.print()

    # Load Demucs model if separation is enabled (default).
    # Separation is required unless --no-separation is explicitly passed,
    # so we abort if it fails to load.
    demucs_model = None
    demucs_device = None
    separation_enabled = False
    if separation:
        console.print("[bold]Loading Demucs model (htdemucs)...[/bold]")
        try:
            from .separator import load_demucs_model

            demucs_model, demucs_device = load_demucs_model()
            separation_enabled = True
            console.print(f"[green]Demucs loaded on {demucs_device}.[/green]")
            console.print()
        except ModuleNotFoundError as e:
            missing = e.name or "Demucs dependencies"
            console.print(
                f"[red]Demucs separation required but unavailable"
                f" (missing {missing}).[/red]"
            )
            console.print(
                "[dim]Install demucs and torch, or use"
                " --no-separation to skip.[/dim]"
            )
            return
        except Exception as e:
            console.print(
                f"[red]Demucs initialization failed: {e}[/red]"
            )
            console.print(
                "[dim]Use --no-separation to extract"
                " without vocal isolation.[/dim]"
            )
            return

    time_step = sample_ms / 1000.0
    stats = {
        "episodes_processed": 0,
        "segments_processed": 0,
        "elapsed_seconds": 0,
    }
    start_time = time.time()

    with create_progress(console) as progress:
        overall_task = progress.add_task("Overall", total=total_segments)

        for ep_dir in pending:
            ep_num = ep_dir.name
            seg_count = len(discover_files(ep_dir, "*.mp3"))
            progress.update(overall_task, description=f"ep {ep_num}")

            ep_data = process_episode(
                ep_dir,
                progress,
                overall_task,
                demucs_model=demucs_model,
                demucs_device=demucs_device,
                save_vocals=save_vocals,
                time_step=time_step,
                pitch_floor=pitch_floor,
                pitch_ceiling=pitch_ceiling,
            )

            if ep_data:
                pitch_result = {
                    "metadata": {
                        "version": "1",
                        "sample_ms": sample_ms,
                        "pitch_floor_hz": int(pitch_floor),
                        "pitch_ceiling_hz": int(pitch_ceiling),
                        "separation": separation_enabled,
                    },
                    "segments": ep_data,
                }

                pitch_file = ep_dir / PITCH_FILE
                with open(pitch_file, "w") as f:
                    json.dump(pitch_result, f, separators=(",", ":"))

                stats["segments_processed"] += len(ep_data)
                stats["episodes_processed"] += 1
                console.print(
                    f"  ep {ep_num}: {len(ep_data)}/{seg_count} segments"
                )
            else:
                console.print(
                    f"  ep {ep_num}: [yellow]no segments extracted[/yellow]"
                )

    stats["elapsed_seconds"] = round(time.time() - start_time, 1)

    console.print()
    console.print("[bold green]Extraction complete![/bold green]")
    console.print(f"  Episodes processed: {stats['episodes_processed']}")
    console.print(f"  Segments processed: {stats['segments_processed']:,}")
    console.print(f"  Time: {stats['elapsed_seconds']:.0f}s")
