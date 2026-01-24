import io
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torchaudio
import torchaudio.functional as F
from pydub import AudioSegment
from pydub.silence import split_on_silence


def resample_audio(audio_path, target_sr=16_000, save_path=None):
    """
    Resample audio to 16kHz mono and optionally save to disc
    Returns:
        torch.Tensor: Resampled waveform as a tensor of shape (channels, samples).
    """
    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if save_path:
        torchaudio.save(save_path, waveform, target_sr)

    return waveform


def apply_highpass_filter(
    waveform: torch.Tensor, sr: int, cutoff_freq: float = 80.0
) -> torch.Tensor:
    """
    Apply high-pass filter to remove low-frequency noise (rumble, handling noise).

    Args:
        waveform: Audio tensor of shape (channels, samples)
        sr: Sample rate
        cutoff_freq: Cutoff frequency in Hz (default 80Hz)

    Returns:
        Filtered waveform
    """
    return F.highpass_biquad(waveform, sr, cutoff_freq)


def calculate_rms(waveform: torch.Tensor) -> float:
    """
    Calculate RMS (Root Mean Square) energy of audio signal.

    Args:
        waveform: Audio tensor of shape (channels, samples)

    Returns:
        RMS value in linear scale [0 - silence, 1 - maximum loud]
    """
    return torch.sqrt(torch.mean(waveform**2)).item()


def calculate_rms_db(waveform: torch.Tensor) -> float:
    """
    Calculate RMS in decibels.

    Args:
        waveform: Audio tensor of shape (channels, samples)

    Returns:
        RMS value in dB [-∞ - silence, 0 - maximum loud]
    """
    rms = calculate_rms(waveform)
    if rms > 0:
        return 20 * torch.log10(torch.tensor(rms)).item()
    return -float("inf")


def normalize_rms(waveform: torch.Tensor, target_rms_db: float = -23.0) -> torch.Tensor:
    """
    Normalize audio to target RMS level in dB.

    Args:
        waveform: Audio tensor of shape (channels, samples)
        target_rms_db: Target RMS level in dB (default -23dB)

    Returns:
        Normalized waveform
    """
    current_rms = calculate_rms(waveform)
    if current_rms == 0:
        return waveform

    # Convert target dB to linear scale
    target_rms_linear = 10 ** (target_rms_db / 20.0)

    # Calculate gain needed
    gain = target_rms_linear / current_rms

    return waveform * gain


def apply_soft_limiter(waveform: torch.Tensor, threshold: float = 0.95) -> torch.Tensor:
    """
    Apply soft limiting to prevent clipping while maintaining naturalness.
    Uses tanh-based soft clipping.

    Args:
        waveform: Audio tensor of shape (channels, samples)
        threshold: Threshold for limiting (default 0.95 to leave headroom)

    Returns:
        Limited waveform
    """
    return threshold * torch.tanh(waveform / threshold)


def detect_silence_segments(
    waveform: torch.Tensor,
    sr: int,
    silence_threshold_db: float = -45.0,
    min_silence_duration: float = 1.3,
    frame_length: float = 0.02,  # 20ms frames
) -> list:
    """
    Detect silence segments in audio.

    Args:
        waveform: Audio tensor of shape (channels, samples)
        sr: Sample rate
        silence_threshold_db: Threshold in dB below which is considered silence
        min_silence_duration: Minimum duration in seconds to be considered silence
        frame_length: Frame length in seconds for analysis

    Returns:
        List of tuples (start_sample, end_sample) for silence segments
    """
    # Convert to mono if needed
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    frame_samples = int(frame_length * sr)
    hop_samples = frame_samples // 2

    # Calculate energy per frame
    num_frames = (waveform.size(1) - frame_samples) // hop_samples + 1
    energies = []

    for i in range(num_frames):
        start = i * hop_samples
        end = start + frame_samples
        frame = waveform[:, start:end]
        rms_db = calculate_rms_db(frame)
        energies.append((start, rms_db))

    # Find silence segments
    silence_segments = []
    in_silence = False
    silence_start = 0

    for i, (sample_pos, energy_db) in enumerate(energies):
        if energy_db < silence_threshold_db:
            if not in_silence:
                silence_start = sample_pos
                in_silence = True
        else:
            if in_silence:
                silence_end = sample_pos
                duration = (silence_end - silence_start) / sr
                if duration >= min_silence_duration:
                    silence_segments.append((silence_start, silence_end))
                in_silence = False

    # Check if still in silence at the end
    if in_silence:
        silence_end = waveform.size(1)
        duration = (silence_end - silence_start) / sr
        if duration >= min_silence_duration:
            silence_segments.append((silence_start, silence_end))

    return silence_segments


def apply_fade(
    waveform: torch.Tensor, sr: int, fade_duration: float = 0.05
) -> torch.Tensor:
    """
    Apply fade in/out to avoid sudden stops.

    Args:
        waveform: Audio tensor of shape (channels, samples)
        sr: Sample rate
        fade_duration: Fade duration in seconds

    Returns:
        Waveform with fades applied
    """
    fade_samples = int(fade_duration * sr)

    if fade_samples * 2 >= waveform.size(1):
        # Audio too short for fading
        return waveform

    # Create fade curves
    fade_in = torch.linspace(0, 1, fade_samples)
    fade_out = torch.linspace(1, 0, fade_samples)

    # Apply fades
    result = waveform.clone()
    result[:, :fade_samples] *= fade_in
    result[:, -fade_samples:] *= fade_out

    return result


def remove_long_silences(
    waveform: torch.Tensor,
    sr: int,
    silence_threshold_db: float = -45.0,
    min_silence_duration: float = 1.3,
    keep_duration: float = 0.5,
    fade_duration: float = 0.05,
) -> torch.Tensor:
    """
    Remove long silence segments while keeping natural pauses and applying fades.

    Args:
        waveform: Audio tensor of shape (channels, samples)
        sr: Sample rate
        silence_threshold_db: Threshold in dB for silence detection
        min_silence_duration: Minimum silence duration to remove (seconds)
        keep_duration: Duration of silence to keep for natural pauses (seconds)
        fade_duration: Fade duration for smooth transitions (seconds)

    Returns:
        Audio with long silences removed
    """
    silence_segments = detect_silence_segments(
        waveform, sr, silence_threshold_db, min_silence_duration
    )

    if not silence_segments:
        return waveform

    # Build list of segments to keep
    keep_segments = []
    last_end = 0

    keep_samples = int(keep_duration * sr)  # keep 500 ms natural silence

    for start, end in silence_segments:
        # Keep audio before silence
        if start > last_end:
            keep_segments.append(waveform[:, last_end:start])

        # Keep a small portion of silence for natural pause
        silence_keep_start = start
        silence_keep_end = min(start + keep_samples, end)

        if silence_keep_end > silence_keep_start:
            silence_segment = waveform[:, silence_keep_start:silence_keep_end]
            # Apply fade out and fade in
            silence_segment = apply_fade(silence_segment, sr, fade_duration)
            keep_segments.append(silence_segment)

        last_end = end

    # Keep remaining audio after last silence
    if last_end < waveform.size(1):
        keep_segments.append(waveform[:, last_end:])

    # Concatenate all segments
    if keep_segments:
        return torch.cat(keep_segments, dim=1)
    return waveform


def normalize_audio(
    waveform: torch.Tensor,
    sr: int,
    target_rms_db: float = -23.0,
    apply_highpass: bool = True,
    highpass_cutoff: float = 80.0,
    remove_silences: bool = True,
    silence_threshold_db: float = -45.0,
    min_silence_duration: float = 1.3,
    apply_limiter: bool = True,
    limiter_threshold: float = 0.95,
) -> torch.Tensor:
    """
    Complete audio normalization pipeline.

    Args:
        waveform: Audio tensor of shape (channels, samples)
        sr: Sample rate
        target_rms_db: Target RMS level in dB
        apply_highpass: Whether to apply high-pass filter
        highpass_cutoff: High-pass filter cutoff frequency
        remove_silences: Whether to remove long silences
        silence_threshold_db: Threshold for silence detection
        min_silence_duration: Minimum silence duration to remove (seconds)
        apply_limiter: Whether to apply soft limiting
        limiter_threshold: Limiter threshold

    Returns:
        Normalized waveform
    """
    # Step 1: High-pass filter
    if apply_highpass:
        waveform = apply_highpass_filter(waveform, sr, highpass_cutoff)

    # Step 2: Remove long silences
    if remove_silences:
        waveform = remove_long_silences(
            waveform, sr, silence_threshold_db, min_silence_duration
        )

    # Step 3: RMS normalization
    waveform = normalize_rms(waveform, target_rms_db)

    # Step 4: Soft limiting to prevent clipping
    if apply_limiter:
        waveform = apply_soft_limiter(waveform, limiter_threshold)

    return waveform


def process_audio_file(
    audio_path: str,
    target_sr: int = 16_000,
    save_path: Optional[str] = None,
    normalize: bool = True,
    **normalize_kwargs,
) -> Tuple[torch.Tensor, dict]:
    """
    Complete audio processing pipeline: resample, convert to mono, and normalize.

    Args:
        audio_path: Path to input audio file
        target_sr: Target sample rate
        save_path: Optional path to save processed audio
        normalize: Whether to apply normalization
        **normalize_kwargs: Additional arguments for normalize_audio()

    Returns:
        Tuple of (processed_waveform, statistics_dict)
    """
    # Load and resample
    waveform, sr = torchaudio.load(audio_path)

    return _process_waveform(
        Path(audio_path).name,
        waveform,
        sr,
        target_sr,
        save_path,
        normalize,
        **normalize_kwargs,
    )


def _process_waveform(
    file_name: str,
    waveform: torch.Tensor,
    current_sr: int,
    target_sr: int = 16_000,
    save_path: Optional[str] = None,
    normalize: bool = True,
    **normalize_kwargs,
):
    """Internal helper to handle the transformation logic and stats."""
    # Store original stats
    original_rms_db = calculate_rms_db(waveform)
    original_duration = waveform.size(1) / current_sr

    if current_sr != target_sr:
        resampler = torchaudio.transforms.Resample(current_sr, target_sr)
        waveform = resampler(waveform)
        current_sr = target_sr

    # Convert to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Apply normalization if requested
    if normalize:
        waveform = normalize_audio(waveform, current_sr, **normalize_kwargs)

    # Calculate final stats
    final_rms_db = calculate_rms_db(waveform)
    final_duration = waveform.size(1) / current_sr
    peak_amplitude = torch.abs(waveform).max().item()

    stats = {
        "input_audio_filename": file_name,
        "original_rms_db": round(original_rms_db, ndigits=4),
        "final_rms_db": round(final_rms_db, ndigits=4),
        "original_duration": original_duration,
        "final_duration": final_duration,
        "reduced_duration": round(original_duration - final_duration, ndigits=4),
        "peak_amplitude": round(peak_amplitude, ndigits=4),
        "is_clipped": peak_amplitude >= 0.99,
    }

    # Save if requested
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(save_path, waveform, current_sr)

    return waveform, stats


def process_audio_bytes(
    file_name: str,
    audio_bytes: bytes,
    target_sr: int = 16_000,
    save_path: Optional[str] = None,
    normalize: bool = True,
    **normalize_kwargs,
) -> Tuple[torch.Tensor, dict]:
    """Processes audio directly from bytes."""
    # Use BytesIO to make bytes seekable for torchaudio
    byte_stream = io.BytesIO(audio_bytes)
    waveform, sr = torchaudio.load(byte_stream)

    return _process_waveform(
        file_name, waveform, sr, target_sr, save_path, normalize, **normalize_kwargs
    )


def chunk_audio_on_silence(
    audio_file: str,
    input_dir: str,
    output_dir: str,
    min_silence_len: int = 400,
    silence_thresh: int = -45,
    keep_silence: int = 200,
    min_chunk_ms: int = 4000,
    max_chunk_ms: int = 15000,
) -> list[dict]:
    """
    Split an audio file on silence and merge chunks to target duration.

    Args:
        audio_file: Filename of the audio file
        input_dir: Directory containing the input audio file
        output_dir: Directory to save output chunks
        min_silence_len: Minimum silence length (ms) to split on
        silence_thresh: Silence threshold in dBFS
        keep_silence: Amount of silence (ms) to keep on each side
        min_chunk_ms: Minimum chunk duration (ms)
        max_chunk_ms: Maximum chunk duration (ms)

    Returns:
        List of metadata dicts for each chunk
    """
    audio = AudioSegment.from_file(os.path.join(input_dir, audio_file))

    raw_chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence,
    )

    if not raw_chunks:
        raw_chunks = [audio]

    # Merge small chunks
    merged_chunks = []
    current_chunk = AudioSegment.empty()

    for chunk in raw_chunks:
        if len(current_chunk) + len(chunk) <= max_chunk_ms:
            current_chunk += chunk
        else:
            if len(current_chunk) >= min_chunk_ms:
                merged_chunks.append(current_chunk)
            elif len(current_chunk) > 0:
                current_chunk += chunk
                merged_chunks.append(current_chunk)
                current_chunk = AudioSegment.empty()
                continue
            current_chunk = chunk

    if len(current_chunk) >= min_chunk_ms:
        merged_chunks.append(current_chunk)
    elif len(current_chunk) > 0 and merged_chunks:
        merged_chunks[-1] += current_chunk
    elif len(current_chunk) > 0:
        merged_chunks.append(current_chunk)

    # Export and collect metadata
    file_metadata = []
    for i, chunk in enumerate(merged_chunks):
        chunk_filename = f"{Path(audio_file).stem}_{i}.wav"
        chunk_path = os.path.join(output_dir, chunk_filename)
        chunk.export(chunk_path, format="wav")

        file_metadata.append(
            {
                "path": audio_file,
                "chunk_path": chunk_filename,
                "duration": round(len(chunk) / 1000, ndigits=2),
                "chunk_index": i,
            }
        )

    return file_metadata
