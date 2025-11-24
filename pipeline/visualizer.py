"""Helpers that turn pipeline predictions into artifacts inside `/outputs`."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


class Visualizer:
    """Generate lightweight artifacts such as JSON reports and charts."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_predictions(
        self,
        predictions: Iterable[Dict[str, Any]],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Path]:
        """Persist predictions and build at least one visualization."""
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        predictions = list(predictions)

        report_path = self._write_predictions_jsonl(
            predictions, metadata or {}, timestamp=timestamp
        )
        length_plot_path = self._plot_transcription_lengths(
            predictions, timestamp=timestamp
        )

        artifacts = [report_path]
        if length_plot_path:
            artifacts.append(length_plot_path)
        return artifacts

    def _write_predictions_jsonl(
        self,
        predictions: Iterable[Dict[str, Any]],
        metadata: Dict[str, Any],
        *,
        timestamp: str,
    ) -> Path:
        """Dump predictions to a JSONL file for downstream inspection."""
        path = self.output_dir / f"predictions_{timestamp}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            header = {"type": "metadata", "payload": metadata}
            handle.write(json.dumps(header) + "\n")
            for record in predictions:
                payload = {
                    "index": record.get("index"),
                    "sample_id": record.get("sample_id"),
                    "prediction": record.get("prediction"),
                    "reference": record.get("reference"),
                }
                handle.write(json.dumps({"type": "prediction", "payload": payload}) + "\n")
        return path

    def _plot_transcription_lengths(
        self,
        predictions: Iterable[Dict[str, Any]],
        *,
        timestamp: str,
    ) -> Optional[Path]:
        """Create a histogram showing the distribution of prediction lengths."""
        if plt is None:
            return None

        texts = [record.get("prediction") or "" for record in predictions]
        lengths = [len(text.split()) for text in texts]

        if not lengths:
            return None

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(lengths, bins=min(20, len(lengths)), color="#2E86AB", alpha=0.85)
        ax.set_xlabel("Words per transcription")
        ax.set_ylabel("Frequency")
        ax.set_title("Prediction Length Distribution")
        fig.tight_layout()

        path = self.output_dir / f"prediction_lengths_{timestamp}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path


__all__ = ["Visualizer"]

