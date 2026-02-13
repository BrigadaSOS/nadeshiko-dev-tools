"""F0 pitch contour extraction using parselmouth (Praat)."""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import parselmouth


def decode_mp3(audio_path: Path) -> tuple[np.ndarray, int]:
    """Decode MP3 to mono float32 numpy array + sample rate via ffmpeg."""
    sample_rate = 16000
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(audio_path),
                "-ac", "1", "-ar", str(sample_rate),
                "-f", "wav", tmp.name,
            ],
            capture_output=True,
            check=True,
        )
        snd = parselmouth.Sound(tmp.name)
        return snd.values[0], int(snd.sampling_frequency)


def extract_f0(
    audio_wav: np.ndarray,
    sample_rate: int,
    time_step: float = 0.01,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0,
) -> list[int]:
    """Extract F0 contour from a waveform array.

    Returns list of rounded Hz values (0 = unvoiced frame).
    """
    snd = parselmouth.Sound(audio_wav, sampling_frequency=sample_rate)
    pitch = snd.to_pitch_ac(
        time_step=time_step,
        pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling,
    )
    f0_values = pitch.selected_array["frequency"]
    return [int(round(v)) if v > 0 else 0 for v in f0_values]
