"""Core inference pipeline utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import math

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - helpful runtime error
    raise ImportError(
        "The `datasets` package is required. Install it with `pip install datasets`."
    ) from exc

try:
    from transformers import pipeline as hf_pipeline
except ImportError as exc:  # pragma: no cover - helpful runtime error
    raise ImportError(
        "The `transformers` package is required. Install it with `pip install transformers`."
    ) from exc

from pipeline.visualizer import Visualizer

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


@dataclass
class PipelineResult:
    """Return payload describing what the pipeline produced."""

    model_name: str
    dataset_name: str
    sample_size: int
    predictions: List[Dict[str, Any]]
    artifacts: List[Path]


def run_pipeline(
    model_name: str,
    dataset_name: str,
    *,
    split: str = "train",
    sample_ratio: float = 0.05,
    audio_column: str = "audio",
    transcript_column: Optional[str] = "text",
    task: str = "automatic-speech-recognition",
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
    **pipeline_kwargs: Any,
) -> PipelineResult:
    """
    Run a Hugging Face model on a fraction of a dataset and visualize the outputs.

    Args:
        model_name: Hugging Face model identifier.
        dataset_name: Hugging Face dataset identifier.
        split: Dataset split to materialize (defaults to `"train"`).
        sample_ratio: Fraction of the split to keep (defaults to 5%).
        audio_column: Column containing audio samples (defaults to `"audio"`).
        transcript_column: Optional reference transcription column used for comparison.
        task: Transformers pipeline task (defaults to ASR).
        output_dir: Directory for generated artifacts (defaults to `/outputs`).
        seed: RNG seed used for shuffling before sampling.
        **pipeline_kwargs: Extra args forwarded to `transformers.pipeline`.

    Returns:
        PipelineResult with per-sample predictions and visualization artifact paths.
    """

    dataset_split = load_dataset(dataset_name, split=split)
    sample_size = _determine_sample_size(len(dataset_split), sample_ratio)
    subset = dataset_split.shuffle(seed=seed).select(range(sample_size))

    inference_pipeline = hf_pipeline(task=task, model=model_name, **pipeline_kwargs)
    predictions = _run_inference(
        inference_pipeline,
        subset,
        audio_column=audio_column,
        transcript_column=transcript_column,
    )

    visualizer = Visualizer(output_dir)
    artifacts = visualizer.visualize_predictions(
        predictions,
        metadata={
            "model": model_name,
            "dataset": dataset_name,
            "split": split,
            "sample_ratio": sample_ratio,
            "sample_size": sample_size,
            "task": task,
        },
    )

    return PipelineResult(
        model_name=model_name,
        dataset_name=dataset_name,
        sample_size=sample_size,
        predictions=predictions,
        artifacts=artifacts,
    )


def _determine_sample_size(dataset_length: int, sample_ratio: float) -> int:
    """Compute the number of samples to keep while guaranteeing at least one."""
    if not 0 < sample_ratio <= 1:
        raise ValueError("sample_ratio must be in the interval (0, 1].")
    if dataset_length <= 0:
        raise ValueError("Cannot sample from an empty dataset.")
    return max(1, math.ceil(dataset_length * sample_ratio))


def _run_inference(
    inference_pipeline: Any,
    dataset_subset: Iterable[Dict[str, Any]],
    *,
    audio_column: str,
    transcript_column: Optional[str],
) -> List[Dict[str, Any]]:
    """Iterate over dataset rows and collect predictions."""
    predictions: List[Dict[str, Any]] = []

    for idx, sample in enumerate(dataset_subset):
        audio_input = _extract_audio(sample, audio_column)
        if audio_input is None:
            raise ValueError(
                f"Sample index {idx} is missing the expected '{audio_column}' column."
            )

        raw_prediction = inference_pipeline(audio_input)
        prediction_text = _extract_prediction_text(raw_prediction)

        predictions.append(
            {
                "index": idx,
                "sample_id": sample.get("id", idx),
                "prediction": prediction_text,
                "reference": sample.get(transcript_column) if transcript_column else None,
            }
        )

    return predictions


def _extract_audio(sample: Dict[str, Any], audio_column: str) -> Any:
    """Support common dataset audio formats (dict with `array` or local path)."""
    audio = sample.get(audio_column)
    if audio is None:
        return None
    if isinstance(audio, dict):
        if "array" in audio:
            return audio["array"]
        if "path" in audio:
            return audio["path"]
    return audio


def _extract_prediction_text(prediction: Any) -> str:
    """Turn the Transformers pipeline output into a printable transcription."""
    if isinstance(prediction, dict):
        return (
            prediction.get("text")
            or prediction.get("generated_text")
            or str(prediction)
        )
    if isinstance(prediction, list) and prediction:
        first = prediction[0]
        if isinstance(first, dict):
            return first.get("text") or first.get("generated_text") or str(first)
        return str(first)
    return str(prediction)


__all__ = ["PipelineResult", "run_pipeline"]

