from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.processing_utils import SpecificProcessorType

from scripts import similarity_metrics


# ============================================================================
# 1. FINE TUNING UTILS
# ============================================================================


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred, processor: SpecificProcessorType):
    """
    Compute metrics using Uzbek text normalizer
    Reports both normalized and raw WER
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels in batch (more efficient)
    preds_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Return corpus-level (macro-averaging) metrics
    return similarity_metrics.calculate_batch(labels_str, preds_str)


# ============================================================================
# 2. LOAD MODEL
# ============================================================================


def load_model(
    model_path: str,
    dataset: dict,
    data_collator,
    base_model_name="openai/whisper-small",
    eval_batch_size=128,
):
    """
    Load a saved model from disk.

    Args:
        model_path: Path to the saved model directory
        dataset: Dataset dictionary with 'validation' and 'test' splits
        data_collator: Whisper data collator (prepares data and converts to PyTorch Tensor)
        base_model_name: Original model name to load processor from
        eval_batch_size: Batch size used in evaluation

    Returns:
        Tuple of (model, trainer, processor)
        - model: Loaded WhisperForConditionalGeneration model
        - trainer: Seq2SeqTrainer instance for evaluation
        - processor: WhisperProcessor (needed for decoding predictions)
    """
    # Load model and processor
    model = WhisperForConditionalGeneration.from_pretrained(model_path)

    # Load processor from base model (checkpoints don't save processor files)
    processor = WhisperProcessor.from_pretrained(
        base_model_name, language="uz", task="transcribe"
    )

    # Ensure pad token is set properly
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    # Create a closure that captures the processor
    def compute_metrics_fn(pred):
        """Wrapper that provides processor to compute_metrics"""
        return compute_metrics(pred, processor)

    # Create trainer for evaluation (minimal config needed)
    eval_args = Seq2SeqTrainingArguments(
        output_dir="./eval_temp",  # Temporary directory
        per_device_eval_batch_size=eval_batch_size,  # Adjust based on your GPU
        predict_with_generate=True,
        fp16=True,  # Use if you have GPU
        fp16_full_eval=False,  # Use FP32 for eval (more stable)
        dataloader_num_workers=8,  # Parallel data loading
        dataloader_pin_memory=True,  # Faster GPU transfer
        generation_max_length=225,
        generation_num_beams=1,
        remove_unused_columns=False,  # Keep metadata columns
    )

    trainer = Seq2SeqTrainer(
        args=eval_args,
        model=model,
        eval_dataset=dataset["validation"],
        data_collator=data_collator,  # Your existing data collator
        compute_metrics=compute_metrics_fn,  # Use the closure
        processing_class=processor.feature_extractor,
    )

    return model, trainer, processor


def load_checkpoint(
    checkpoint_path: str,
    dataset: dict,
    data_collator,
    base_model_name="openai/whisper-small",
    eval_batch_size=128,
):
    """Load a specific checkpoint."""
    return load_model(
        checkpoint_path,
        dataset,
        data_collator,
        base_model_name=base_model_name,
        eval_batch_size=eval_batch_size,
    )


# ============================================================================
# 3. EVALUATION
# ============================================================================


def _print_evaluation_result(metrics: dict):
    print(
        f"{'WER (normalized)':<25} {metrics.get('eval_wer', metrics['wer']) * 100:>9.2f}%"
    )
    print(
        f"{'CER (normalized)':<25} {metrics.get('eval_cer', metrics['cer']) * 100:>9.2f}%"
    )
    print(
        f"{'Sequence Similarity':<25} {metrics.get('eval_sequence_similarity', metrics['sequence_similarity']) * 100:>9.2f}%"
    )

    print(
        f"{'WER (raw)':<25} {metrics.get('eval_wer_raw', metrics['wer_raw']) * 100:>9.2f}%"
    )
    print(
        f"{'CER (raw)':<25} {metrics.get('eval_cer_raw', metrics['cer_raw']) * 100:>9.2f}%"
    )
    print(
        f"{'Seq Similarity (raw)':<25} {metrics.get('eval_sequence_similarity_raw', metrics['sequence_similarity_raw']) * 100:>9.2f}%"
    )


def print_evaluation_results(
    metrics: dict, dataset_name: str = "VALIDATION", group_by_dataset: bool = False
):
    """
    Print formatted evaluation metrics with optional dataset grouping.

    Args:
        metrics: Dictionary of metrics from trainer.evaluate() or evaluate_by_dataset()
        dataset_name: Name of the dataset being evaluated
        group_by_dataset: If True, expects metrics from evaluate_by_dataset()
    """
    print(f"\n{'=' * 80}")
    print(f"DETAILED EVALUATION: {dataset_name.upper()}")
    print(f"{'=' * 80}")

    if group_by_dataset:
        # Overall metrics
        print("\nOVERALL METRICS")
        print("-" * 80)
        overall = metrics["overall"]
        _print_evaluation_result(overall)

        # Per-dataset metrics
        print(f"\n{'=' * 80}")
        print("METRICS BY DATASET")
        print(f"{'=' * 80}")

        # Sort by dataset name for consistent output
        for ds_name in sorted(metrics["by_dataset"].keys()):
            print(f"\n{ds_name}")
            print("-" * 80)
            ds_metrics = metrics["by_dataset"][ds_name]
            _print_evaluation_result(ds_metrics)
    else:
        print(f"{'Metric':<25} {'Value':>10}")
        print("-" * 80)
        _print_evaluation_result(metrics)

    print("=" * 80 + "\n")


def evaluate_by_dataset(trainer, processor, dataset_split, dataset_name="validation"):
    """
    Evaluate model and group results by dataset.

    Args:
        trainer: Seq2SeqTrainer instance
        processor: Model processor
        dataset_split: Dataset split to evaluate
        dataset_name: Dataset name (VALIDATION or TEST)

    Returns:
        Dictionary with overall and per-dataset metrics
    """
    from collections import defaultdict

    # Get predictions
    predictions = trainer.predict(dataset_split)

    # Decode predictions and labels
    pred_ids = predictions.predictions
    label_ids = predictions.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    labels_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    preds_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

    # Get dataset labels
    dataset_labels = dataset_split["dataset"]

    # Group by dataset
    dataset_groups = defaultdict(lambda: {"predictions": [], "references": []})

    for pred, label, ds_name in zip(preds_str, labels_str, dataset_labels):
        dataset_groups[ds_name]["predictions"].append(pred)
        dataset_groups[ds_name]["references"].append(label)

    # Compute metrics for each dataset
    results = {
        "overall": similarity_metrics.calculate_batch(labels_str, preds_str),
        "by_dataset": {},
    }

    # Per-dataset metrics
    for ds_name, data in dataset_groups.items():
        refs = data["references"]
        preds = data["predictions"]

        results["by_dataset"][ds_name] = similarity_metrics.calculate_batch(refs, preds)

    print_evaluation_results(results, dataset_name.upper(), group_by_dataset=True)

    return results


def evaluate(trainer, dataset):
    # Evaluate
    val_metrics = trainer.evaluate(dataset["validation"])
    print_evaluation_results(val_metrics, "validation")

    test_metrics = trainer.evaluate(dataset["test"])
    print_evaluation_results(test_metrics, "test")

    return {"validation": val_metrics, "test": test_metrics}
