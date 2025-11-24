import os
import sys
from typing import List, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.pipeline import PipelineResult, run_pipeline
from resources.datasets import Datasets
from resources.models import Models

MODEL_DATASET_PAIRS: Sequence[Tuple[str, str]] = (
    (Models.PARAKEET_TDT_06B_V3, Datasets.NEWS_YOUTUBE_UZBEK_SPEECH),
    (Models.WHISPER_SMALL, Datasets.NEWS_YOUTUBE_UZBEK_SPEECH),
    (Models.WAV2VEC2_XLS_R_300M, Datasets.NEWS_YOUTUBE_UZBEK_SPEECH),
)

DEFAULT_SAMPLE_RATIO = 0.0001
DEFAULT_SPLIT = "train"
PREDICTION_PREVIEW_LIMIT = 3


def run_all_models(
    *,
    model_dataset_pairs: Sequence[Tuple[str, str]] = MODEL_DATASET_PAIRS,
    sample_ratio: float = DEFAULT_SAMPLE_RATIO,
    split: str = DEFAULT_SPLIT,
    **pipeline_kwargs,
) -> List[PipelineResult]:
    """Run each configured model on its dataset and print summaries."""
    results: List[PipelineResult] = []

    for model_name, dataset_name in model_dataset_pairs:
        print(
            f"\n=== Running {model_name} on {dataset_name} "
            f"(split={split}, sample_ratio={sample_ratio}) ==="
        )
        try:
            result = run_pipeline(
                model_name,
                dataset_name,
                split=split,
                sample_ratio=sample_ratio,
                **pipeline_kwargs,
            )
        except Exception as exc:
            print(f"[ERROR] Pipeline failed for {model_name} on {dataset_name}: {exc}")
            continue

        results.append(result)
        _print_result_summary(result)

    return results


def _print_result_summary(
    result: PipelineResult, *, preview_limit: int = PREDICTION_PREVIEW_LIMIT
) -> None:
    print(f"Model: {result.model_name}")
    print(f"Dataset: {result.dataset_name}")
    print(f"Sample size: {result.sample_size}")
    print("Artifacts:")
    if result.artifacts:
        for path in result.artifacts:
            print(f"  - {path}")
    else:
        print("  - (no artifacts)")

    print("Predictions:")
    if not result.predictions:
        print("  - (no predictions)")
        return

    for prediction in result.predictions[:preview_limit]:
        reference = prediction.get("reference") or ""
        print(
            f"  - idx={prediction.get('index')} "
            f"id={prediction.get('sample_id')} "
            f"pred=\"{_shorten(prediction.get('prediction'))}\" "
            f"ref=\"{_shorten(reference)}\""
        )
    remaining = len(result.predictions) - preview_limit
    if remaining > 0:
        print(f"  ... {remaining} more predictions omitted for brevity.")


def _shorten(value: str | None, max_chars: int = 80) -> str:
    if not value:
        return ""
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[WARN] Invalid float for {name}={raw!r}; using default {default}.")
        return default


if __name__ == "__main__":
    sample_ratio = _env_float("ASR_E2E_SAMPLE_RATIO", DEFAULT_SAMPLE_RATIO)
    split = os.environ.get("ASR_E2E_SPLIT", DEFAULT_SPLIT)
    device = os.environ.get("ASR_E2E_DEVICE")
    pipeline_kwargs = {}
    if device is not None:
        try:
            pipeline_kwargs["device"] = int(device)
        except ValueError:
            print(f"[WARN] Invalid ASR_E2E_DEVICE={device!r}; ignoring.")

    results = run_all_models(sample_ratio=sample_ratio, split=split, **pipeline_kwargs)
    print(
        f"\nCompleted {len(results)} successful runs out of "
        f"{len(MODEL_DATASET_PAIRS)} configured pairs."
    )