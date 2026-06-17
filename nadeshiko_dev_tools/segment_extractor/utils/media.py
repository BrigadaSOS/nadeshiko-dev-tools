"""Per-segment media extraction via ffmpeg.

Shared by the anime/jdrama splitter and the YouTube pipeline so both produce
identical audio (mp3), screenshot (webp) and clip (mp4) outputs. The "clip" is a
still screenshot looped over the segment's audio, matching the original splitter
behaviour.
"""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def _run_ffmpeg(cmd: list[str], what: str) -> None:
    """Run an ffmpeg command, raising RuntimeError with captured output on failure."""
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"[red]ffmpeg {what} failed: {result.stdout}[/red]")
        raise RuntimeError(f"ffmpeg {what} failed with code {result.returncode}")


def extract_audio(
    video_file: str,
    output_path: str,
    start_seconds: float,
    end_seconds: float,
    audio_index: int | None = None,
) -> None:
    """Extract a loudness-normalised mp3 for [start, end] from video_file."""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_seconds),
        "-i",
        video_file,
        "-t",
        str(end_seconds - start_seconds),
    ]
    if audio_index is not None:
        cmd.extend(["-map", f"0:{audio_index}"])
    cmd.extend(
        [
            "-vn",
            "-af",
            "loudnorm=I=-16:LRA=11:TP=-2",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "5",
            output_path,
        ]
    )
    _run_ffmpeg(cmd, "audio")


def extract_screenshot(video_file: str, output_path: str, time_seconds: float) -> None:
    """Grab a single 960x540 webp frame at time_seconds from video_file."""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(time_seconds),
        "-i",
        video_file,
        "-vframes",
        "1",
        "-vf",
        "scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:black",
        "-c:v",
        "libwebp",
        "-quality",
        "85",
        "-method",
        "6",
        output_path,
    ]
    _run_ffmpeg(cmd, "screenshot")


def build_clip(screenshot_path: str, audio_path: str, output_path: str) -> None:
    """Build a web-optimised mp4 looping the screenshot over the audio track."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-framerate",
        "24",
        "-i",
        screenshot_path,
        "-i",
        audio_path,
        "-vf",
        "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,setsar=1",
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-preset",
        "faster",
        "-tune",
        "fastdecode",
        "-crf",
        "35",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-aac_coder",
        "twoloop",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        "-shortest",
        output_path,
    ]
    _run_ffmpeg(cmd, "video")


def generate_segment_media(
    video_file: str,
    output_path: str,
    segment_hash: str,
    start_seconds: float,
    end_seconds: float,
    audio_index: int | None = None,
) -> dict[str, str]:
    """Produce audio + screenshot + clip for one segment.

    Returns the filenames keyed by type ({"audio", "screenshot", "video"}).
    Raises RuntimeError if any ffmpeg step fails.
    """
    audio_filename = f"{segment_hash}.mp3"
    screenshot_filename = f"{segment_hash}.webp"
    video_filename = f"{segment_hash}.mp4"

    audio_path = os.path.join(output_path, audio_filename)
    screenshot_path = os.path.join(output_path, screenshot_filename)
    video_path = os.path.join(output_path, video_filename)

    extract_audio(video_file, audio_path, start_seconds, end_seconds, audio_index)
    screenshot_time = (start_seconds + end_seconds) / 2
    extract_screenshot(video_file, screenshot_path, screenshot_time)
    build_clip(screenshot_path, audio_path, video_path)

    return {
        "audio": audio_filename,
        "screenshot": screenshot_filename,
        "video": video_filename,
    }
