"""Demucs vocal isolation for cleaner F0 extraction."""

import numpy as np
import torch
import torchaudio


def load_demucs_model(device: str | None = None) -> tuple:
    """Load htdemucs model, auto-detecting GPU/CPU.

    Returns (model, device_str) tuple.
    """
    from demucs.pretrained import get_model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model("htdemucs")
    model.to(device)
    model.eval()
    return model, device


def separate_vocals(
    model,
    device: str,
    audio_wav: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Run Demucs on audio and return vocals-only waveform.

    Args:
        model: Loaded Demucs model.
        device: Device string ("cuda" or "cpu").
        audio_wav: Mono float32 waveform.
        sample_rate: Sample rate of the input audio.

    Returns:
        Mono float32 vocals waveform at the original sample rate.
    """
    from demucs.apply import apply_model

    model_sr = model.samplerate

    # Convert mono to stereo (Demucs expects stereo)
    wav_tensor = torch.from_numpy(audio_wav).float()
    if wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0).repeat(2, 1)

    # Resample to model's expected sample rate if needed
    if sample_rate != model_sr:
        wav_tensor = torchaudio.functional.resample(wav_tensor, sample_rate, model_sr)

    # Pad short segments (Demucs needs minimum length)
    min_samples = model_sr  # 1 second minimum
    original_length = wav_tensor.shape[-1]
    if original_length < min_samples:
        pad_amount = min_samples - original_length
        wav_tensor = torch.nn.functional.pad(wav_tensor, (0, pad_amount))

    # Run separation: (channels, samples) -> (batch, channels, samples)
    wav_input = wav_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        sources = apply_model(model, wav_input)

    # Extract vocals track (index 3 in htdemucs: drums, bass, other, vocals)
    vocals = sources[0, 3]  # (2, samples)

    # Remove padding if we added any
    if original_length < min_samples:
        vocals = vocals[:, :original_length]

    # Resample back to original sample rate if needed
    if sample_rate != model_sr:
        vocals = torchaudio.functional.resample(vocals, model_sr, sample_rate)

    # Convert stereo to mono
    vocals_mono = vocals.mean(dim=0).cpu().numpy()
    return vocals_mono
