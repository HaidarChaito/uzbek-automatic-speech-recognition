from enum import Flag, auto
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from audiomentations import TimeStretch, PitchShift, AddGaussianNoise, TimeMask
from tqdm import tqdm


class AugmentationType(Flag):
    """Flag enum for audio augmentation types."""

    NONE = 0
    SPEED = auto()
    PITCH = auto()
    NOISE = auto()
    TIME_MASK = auto()
    ALL = SPEED | PITCH | NOISE | TIME_MASK


class AudiomentationsAugmentor:
    """Simple augmentor with tracking."""

    def __init__(self, target_sr=16_000, min_speed_rate=0.9, max_speed_rate=1.1):
        self.target_sr = target_sr
        self.min_speed_rate = min_speed_rate
        self.max_speed_rate = max_speed_rate

        # Augmentation configs
        self.speed_aug = TimeStretch(
            min_rate=min_speed_rate,
            max_rate=max_speed_rate,
            leave_length_unchanged=False,
            p=1.0,
        )
        self.pitch_aug = PitchShift(min_semitones=-2, max_semitones=2, p=1.0)
        self.noise_aug = AddGaussianNoise(
            min_amplitude=0.003, max_amplitude=0.013, p=1.0
        )
        self.time_mask_aug = TimeMask(
            min_band_part=0.0, max_band_part=0.1, fade_duration=0.005, p=1.0
        )

    def augment(self, audio_path, augmentation_type=AugmentationType.ALL):
        """Apply augmentations using audiomentations."""
        # Load with torchaudio
        waveform, sr = torchaudio.load(audio_path)

        # Convert to numpy (audiomentations expects numpy)
        audio_np = waveform.squeeze().numpy()

        # Resample if needed
        if sr != self.target_sr:
            resampler = T.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
            audio_np = waveform.squeeze().numpy()

        # Apply augmentations
        if AugmentationType.SPEED in augmentation_type:
            audio_np = self.speed_aug(audio_np, sample_rate=self.target_sr)

        if AugmentationType.PITCH in augmentation_type:
            audio_np = self.pitch_aug(audio_np, sample_rate=self.target_sr)

        if AugmentationType.NOISE in augmentation_type:
            audio_np = self.noise_aug(audio_np, sample_rate=self.target_sr)

        if AugmentationType.TIME_MASK in augmentation_type:
            audio_np = self.time_mask_aug(audio_np, sample_rate=self.target_sr)

        # Convert back to torch
        augmented = torch.from_numpy(audio_np).unsqueeze(0)

        return augmented, self.target_sr

    def augment_and_save(
        self,
        audio_path: str,
        output_path: str,
        augmentation_type: AugmentationType,
        noise_prob: float = 0.5,
        time_mask_prob: float = 0.5,
    ) -> dict:
        """
        Augment audio and return tracking info.

        Returns:
            dict with augmentation details including durations and speed rate
        """
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        original_duration = waveform.shape[1] / sr
        audio_np = waveform.squeeze().numpy()

        # Resample if needed
        if sr != self.target_sr:
            resampler = T.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
            audio_np = waveform.squeeze().numpy()
            original_duration = waveform.shape[1] / self.target_sr

        # Track what was applied
        applied = {
            "original_path": Path(audio_path).name,
            "augmented_path": Path(output_path).name,
            "original_duration": round(original_duration, 4),
            "augmented_duration": None,
            "speed_applied": False,
            "speed_rate_applied": None,
            "pitch_applied": False,
            "noise_applied": False,
            "time_mask_applied": False,
            "augmentation_type": str(augmentation_type),
        }

        # Apply augmentations
        if AugmentationType.SPEED in augmentation_type:
            # Randomly select and track speed rate
            speed_rate = np.random.uniform(self.min_speed_rate, self.max_speed_rate)
            applied["speed_rate_applied"] = round(speed_rate, 4)

            # Apply with specific rate
            audio_np = self.speed_aug(audio_np, sample_rate=self.target_sr)
            applied["speed_applied"] = True

        if AugmentationType.PITCH in augmentation_type:
            audio_np = self.pitch_aug(audio_np, sample_rate=self.target_sr)
            applied["pitch_applied"] = True

        if AugmentationType.NOISE in augmentation_type:
            if np.random.random() < noise_prob:
                audio_np = self.noise_aug(audio_np, sample_rate=self.target_sr)
                applied["noise_applied"] = True

        if AugmentationType.TIME_MASK in augmentation_type:
            if np.random.random() < time_mask_prob:
                audio_np = self.time_mask_aug(audio_np, sample_rate=self.target_sr)
                applied["time_mask_applied"] = True

        # Save and calculate augmented duration
        augmented = torch.from_numpy(audio_np).unsqueeze(0)
        augmented_duration = augmented.shape[1] / self.target_sr
        applied["augmented_duration"] = round(augmented_duration, 4)
        applied["speed_rate_applied"] = round(original_duration / augmented_duration, 4)

        torchaudio.save(output_path, augmented, self.target_sr)

        return applied


def augment_dataset(
    audio_paths: List[str],
    output_dir: str,
    augmentation_types: List[AugmentationType],
    noise_prob: float = 0.5,
    time_mask_prob: float = 0.5,
    min_speed_rate: float = 0.9,
    max_speed_rate: float = 1.1,
) -> pd.DataFrame:
    """
    Augment dataset and return tracking DataFrame.

    Args:
        audio_paths: List of audio file paths
        output_dir: Directory to save augmented files
        augmentation_types: List of augmentation types to apply
        noise_prob: Probability of applying noise
        time_mask_prob: Probability of applying time mask
        min_speed_rate: Minimum speed rate (e.g., 0.9 = 10% slower)
        max_speed_rate: Maximum speed rate (e.g., 1.1 = 10% faster)

    Returns:
        DataFrame with augmentation tracking info
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    augmentor = AudiomentationsAugmentor(
        target_sr=16_000, min_speed_rate=min_speed_rate, max_speed_rate=max_speed_rate
    )
    records = []

    for audio_path in tqdm(audio_paths, desc="Processing files..."):
        base_name = Path(audio_path).stem + "_augmented"

        for aug_type in augmentation_types:
            # Create output filename
            suffix = []
            if AugmentationType.SPEED in aug_type:
                suffix.append("spd")
            if AugmentationType.PITCH in aug_type:
                suffix.append("pch")
            if AugmentationType.NOISE in aug_type:
                suffix.append("nse")
            if AugmentationType.TIME_MASK in aug_type:
                suffix.append("msk")

            suffix_str = "_".join(suffix) if suffix else "orig"
            output_filename = f"{base_name}_{suffix_str}.wav"
            output_path = str(Path(output_dir) / output_filename)

            # Augment and track
            try:
                record = augmentor.augment_and_save(
                    audio_path=audio_path,
                    output_path=output_path,
                    augmentation_type=aug_type,
                    noise_prob=noise_prob,
                    time_mask_prob=time_mask_prob,
                )
                records.append(record)

            except Exception as e:
                print(f"Error: {audio_path} with {aug_type}: {e}")

    # Create DataFrame
    df = pd.DataFrame(records)

    # Save to CSV
    csv_path = Path(output_dir) / "augmentation_log.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved log to {csv_path}")

    return df


# Usage
if __name__ == "__main__":
    # Your audio files
    audio_paths = ["file1.wav", "file2.wav", "file3.wav"]

    # Augmentation strategies
    aug_types = [
        AugmentationType.NONE,
        AugmentationType.SPEED,
        AugmentationType.PITCH,
        AugmentationType.NOISE,
        AugmentationType.TIME_MASK,
        AugmentationType.ALL,
    ]

    # Augment and get tracking DataFrame
    df = augment_dataset(
        audio_paths=audio_paths,
        output_dir="augmented_data",
        augmentation_types=aug_types,
        noise_prob=0.5,
        time_mask_prob=0.5,
        min_speed_rate=0.9,
        max_speed_rate=1.1,
    )

    # Analysis
    print("\nSummary:")
    print(f"Total files: {len(df)}")
    print(f"Speed applied: {df['speed_applied'].sum()}")
    print(f"Pitch applied: {df['pitch_applied'].sum()}")
    print(f"Noise applied: {df['noise_applied'].sum()}")
    print(f"Time mask applied: {df['time_mask_applied'].sum()}")

    # Speed rate statistics
    speed_df = df[df["speed_rate_applied"].notna()]
    if len(speed_df) > 0:
        print(f"\nSpeed Rate Stats:")
        print(f"  Mean: {speed_df['speed_rate_applied'].mean():.4f}")
        print(f"  Min: {speed_df['speed_rate_applied'].min():.4f}")
        print(f"  Max: {speed_df['speed_rate_applied'].max():.4f}")

    # Duration changes
    print(f"\nAvg original duration: {df['original_duration'].mean():.2f}s")
    print(f"Avg augmented duration: {df['augmented_duration'].mean():.2f}s")
