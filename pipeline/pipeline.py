import argparse
import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Audio, Dataset, concatenate_datasets
from evaluate import load as load_metric
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEXT_COLUMN_CANDIDATES = (
    "ref_normalized",
    "sentence_checked",
    "text_spt",
    "predicted_sentence",
    "text",
    "transcript",
    "transcription",
)

PATH_COLUMN_CANDIDATES = ("path", "audio", "file", "file_path")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    csv_path: Path
    audio_dir: Optional[Path] = None


def build_dataset_configs(data_root: Path) -> List[DatasetConfig]:
    return [
        DatasetConfig(
            name="feruza_speech",
            csv_path=data_root / "feruza_speech_modified.csv",
            audio_dir=data_root / "feruza-speech" / "sampled_audio",
        ),
        DatasetConfig(
            name="uzbek_speech_corpus",
            csv_path=data_root / "uzbek-speech-corpus__case_insensitive.csv",
            audio_dir=data_root / "uzbek-speech-corpus" / "sampled_audio",
        ),
        DatasetConfig(
            name="it_dataset",
            csv_path=data_root / "it_dataset_modified.csv",
            audio_dir=data_root / "it_youtube_uzbek_speech_dataset" / "sampled_audio",
        ),
        DatasetConfig(
            name="news_dataset",
            csv_path=data_root / "news_dataset_modified.csv",
            audio_dir=data_root / "news_youtube_uzbek_speech_dataset" / "sampled_audio",
        ),
        DatasetConfig(
            name="uzbek_voice",
            csv_path=data_root / "uzbek_voice_modified.csv",
            audio_dir=data_root / "uzbekvoice_dataset" / "sampled_audio",
        ),
        DatasetConfig(
            name="common_voice",
            csv_path=data_root / "common_voice_modified.csv",
            audio_dir=data_root / "common_voice" / "sampled_audio",
        ),
    ]


def _select_column(candidates: Iterable[str], columns: Iterable[str]) -> Optional[str]:
    column_set = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in column_set:
            return column_set[candidate]
    return None


def _resolve_audio_dir(
    config: DatasetConfig,
    data_root: Path,
    audio_root: Optional[Path],
    audio_map: Optional[dict],
) -> Path:
    if audio_map and config.name in audio_map:
        mapped = Path(audio_map[config.name])
        if mapped.exists():
            return mapped

    if audio_root:
        candidates = []
        try:
            relative = config.csv_path.parent.relative_to(data_root)
            candidates.append(audio_root / relative)
        except ValueError:
            pass
        candidates.extend(
            [
                audio_root / config.name,
                audio_root / config.csv_path.parent.name,
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate

    if config.audio_dir and config.audio_dir.exists():
        return config.audio_dir

    candidates = [
        config.csv_path.parent,
        config.csv_path.parent / "audio",
        config.csv_path.parent / "wavs",
        config.csv_path.parent / "clips",
        config.csv_path.parent.parent / "audio",
        config.csv_path.parent.parent / "wavs",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return config.csv_path.parent


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def _load_csv_dataset(
    config: DatasetConfig,
    sample_fraction: float,
    seed: int,
    data_root: Path,
    audio_root: Optional[Path],
    audio_map: Optional[dict],
) -> Tuple[str, Dataset]:
    if not config.csv_path.exists():
        raise FileNotFoundError(f"Missing CSV for {config.name}: {config.csv_path}")

    df = pd.read_csv(config.csv_path)
    text_column = _select_column(TEXT_COLUMN_CANDIDATES, df.columns)
    path_column = _select_column(PATH_COLUMN_CANDIDATES, df.columns)
    if not text_column or not path_column:
        raise ValueError(
            f"CSV {config.csv_path} is missing required columns for {config.name}."
        )

    df = df[[path_column, text_column]].rename(
        columns={path_column: "audio_path", text_column: "text"}
    )
    df = df.dropna(subset=["audio_path", "text"])
    if 0 < sample_fraction < 1:
        df = df.sample(frac=sample_fraction, random_state=seed)

    audio_dir = _resolve_audio_dir(config, data_root, audio_root, audio_map)
    resolved_paths = []
    for raw_path in df["audio_path"].tolist():
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = audio_dir / candidate
        resolved_paths.append(candidate)
    df["audio"] = [str(path) for path in resolved_paths]
    df["text"] = df["text"].map(_normalize_text)

    exists_mask = [Path(path).exists() for path in df["audio"].tolist()]
    missing = len(exists_mask) - sum(exists_mask)
    if missing:
        print(f"[{config.name}] skipping {missing} missing audio files.")
    df = df[exists_mask]

    if df.empty:
        raise ValueError(
            f"No audio files found for {config.name}. Check audio paths for {config.csv_path}."
        )

    dataset = Dataset.from_pandas(df[["audio", "text"]])
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    return config.name, dataset


def _build_vocab(texts: List[str], output_dir: Path) -> Path:
    vocab_chars = sorted(set("".join(texts)))
    vocab_dict = {}
    for char in vocab_chars:
        if char == " ":
            vocab_dict["|"] = len(vocab_dict)
        else:
            vocab_dict[char] = len(vocab_dict)
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.json"
    with vocab_path.open("w", encoding="utf-8") as handle:
        json.dump(vocab_dict, handle, ensure_ascii=False)
    return vocab_path


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: str = "longest"

    def __call__(self, features: List[dict]) -> dict:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


def _prepare_batch(
    batch: dict, processor: Wav2Vec2Processor
) -> dict:
    audio = batch["audio"]
    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    batch["input_length"] = len(audio["array"])
    batch["labels"] = processor(text=batch["text"]).input_ids
    return batch


def _filter_by_duration(
    dataset: Dataset,
    max_audio_seconds: Optional[float],
    min_audio_seconds: Optional[float],
) -> Dataset:
    if max_audio_seconds is None and min_audio_seconds is None:
        return dataset

    def _keep(row: dict) -> bool:
        audio = row["audio"]
        duration = len(audio["array"]) / audio["sampling_rate"]
        if max_audio_seconds is not None and duration > max_audio_seconds:
            return False
        if min_audio_seconds is not None and duration < min_audio_seconds:
            return False
        return True

    return dataset.filter(_keep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune XLS-R on local Uzbek datasets.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets"),
        help="Root directory for local datasets.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.1,
        help="Fraction of each dataset to use for training/testing.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/wav2vec2-xls-r-300m",
        help="Hugging Face model name or path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wav2vec2-xls-r-300m"),
        help="Directory to save checkpoints and artifacts.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Optional root directory for audio files.",
    )
    parser.add_argument(
        "--audio-map",
        type=Path,
        default=None,
        help="JSON file mapping dataset names to audio directories.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Optional max training steps override.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "mps"),
        help="Force training device (auto uses CPU on MPS).",
    )
    parser.add_argument(
        "--mps-fallback",
        action="store_true",
        help="Enable CPU fallback for unsupported MPS ops.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Per-device train batch size override.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Per-device eval batch size override.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps override.",
    )
    parser.add_argument(
        "--group-by-length",
        action="store_true",
        help="Group samples of similar length to reduce padding.",
    )
    parser.add_argument(
        "--no-group-by-length",
        action="store_true",
        help="Disable length-based batching.",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=None,
        help="Drop audio samples longer than this duration (seconds).",
    )
    parser.add_argument(
        "--min-audio-seconds",
        type=float,
        default=None,
        help="Drop audio samples shorter than this duration (seconds).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce GPU memory usage.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing.",
    )
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=0,
        help="Print N decoded predictions vs labels during evaluation.",
    )
    parser.add_argument(
        "--debug-metrics",
        action="store_true",
        help="Print basic stats about empty/short predictions and labels.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup steps for the learning rate scheduler.",
    )
    parser.add_argument(
        "--freeze-feature-extractor",
        action="store_true",
        help="Freeze the wav2vec2 feature extractor.",
    )
    parser.add_argument(
        "--no-freeze-feature-extractor",
        action="store_true",
        help="Do not freeze the wav2vec2 feature extractor.",
    )
    parser.add_argument(
        "--ctc-zero-infinity",
        action="store_true",
        help="Enable CTC zero infinity to avoid NaNs.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_mps = torch.backends.mps.is_available()
    if args.mps_fallback:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    dataset_configs = build_dataset_configs(args.data_root)
    audio_map = None
    if args.audio_map:
        if not args.audio_map.exists():
            raise FileNotFoundError(f"Audio map file not found: {args.audio_map}")
        with args.audio_map.open("r", encoding="utf-8") as handle:
            audio_map = json.load(handle)
    datasets = []
    for config in dataset_configs:
        try:
            name, dataset = _load_csv_dataset(
                config,
                sample_fraction=args.sample_fraction,
                seed=args.seed,
                data_root=args.data_root,
                audio_root=args.audio_root,
                audio_map=audio_map,
            )
            print(f"[{name}] loaded {len(dataset)} samples")
            before_filter = len(dataset)
            dataset = _filter_by_duration(
                dataset,
                max_audio_seconds=args.max_audio_seconds,
                min_audio_seconds=args.min_audio_seconds,
            )
            if len(dataset) != before_filter:
                print(
                    f"[{name}] filtered {before_filter - len(dataset)} samples "
                    f"outside duration constraints."
                )
            datasets.append(dataset)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[{config.name}] skipped: {exc}")

    if not datasets:
        raise SystemExit("No datasets were loaded. Check CSV paths and audio files.")

    combined = concatenate_datasets(datasets).shuffle(seed=args.seed)
    vocab_path = _build_vocab(combined["text"], args.output_dir)

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path.as_posix(),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16_000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    split = combined.train_test_split(test_size=0.2, seed=args.seed)
    train_valid = split["train"].train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = train_valid["train"]
    eval_dataset = train_valid["test"]
    test_dataset = split["test"]

    prepare_fn = lambda batch: _prepare_batch(batch, processor)
    train_dataset = train_dataset.map(prepare_fn)
    eval_dataset = eval_dataset.map(prepare_fn)
    test_dataset = test_dataset.map(prepare_fn)

    data_collator = DataCollatorCTCWithPadding(processor=processor)
    wer_metric = load_metric("wer")

    printed_debug = False

    def compute_metrics(pred) -> dict:
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred_str = processor.batch_decode(pred_ids)
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        nonlocal printed_debug
        if args.debug_samples > 0 and not printed_debug:
            printed_debug = True
            sample_count = min(args.debug_samples, len(pred_str))
            print("\n[debug] sample predictions vs labels")
            for idx in range(sample_count):
                print(f"[pred {idx}] {pred_str[idx]}")
                print(f"[label {idx}] {label_str[idx]}")
                print("---")
        if args.debug_metrics:
            pred_lengths = [len(text.strip()) for text in pred_str]
            label_lengths = [len(text.strip()) for text in label_str]
            empty_preds = sum(length == 0 for length in pred_lengths)
            empty_labels = sum(length == 0 for length in label_lengths)
            avg_pred_len = float(np.mean(pred_lengths)) if pred_lengths else 0.0
            avg_label_len = float(np.mean(label_lengths)) if label_lengths else 0.0
            print(
                "[debug] empty preds/labels: "
                f"{empty_preds}/{len(pred_str)} | {empty_labels}/{len(label_str)}"
            )
            print(
                "[debug] avg pred/label length: "
                f"{avg_pred_len:.1f} | {avg_label_len:.1f}"
            )
            if pred_ids.size:
                flat_ids = pred_ids.reshape(-1)
                counts = np.bincount(flat_ids, minlength=len(processor.tokenizer))
                top_id = int(np.argmax(counts))
                top_token = processor.tokenizer.convert_ids_to_tokens([top_id])[0]
                print(
                    f"[debug] most common pred token: id={top_id} token={top_token}"
                )
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "accuracy": 1.0 - wer}

    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name,
        vocab_size=len(processor.tokenizer),
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        ctc_zero_infinity=args.ctc_zero_infinity,
    )
    if args.no_freeze_feature_extractor:
        freeze_feature_extractor = False
    elif args.freeze_feature_extractor:
        freeze_feature_extractor = True
    else:
        freeze_feature_extractor = True
    if freeze_feature_extractor:
        model.freeze_feature_extractor()

    if args.device == "cpu":
        use_mps = False
    elif args.device == "mps":
        use_mps = True

    force_cpu = args.device == "cpu" or (args.device == "auto" and use_mps)
    use_cuda = torch.cuda.is_available() and not force_cpu

    if args.no_gradient_checkpointing:
        gradient_checkpointing = False
    elif args.gradient_checkpointing:
        gradient_checkpointing = True
    else:
        gradient_checkpointing = use_cuda or (use_mps and not force_cpu)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    per_device_train_batch_size = (
        args.train_batch_size
        if args.train_batch_size is not None
        else 1
        if (use_mps and not force_cpu) or use_cuda
        else 2
    )
    per_device_eval_batch_size = (
        args.eval_batch_size
        if args.eval_batch_size is not None
        else 1
        if (use_mps and not force_cpu) or use_cuda
        else 2
    )
    gradient_accumulation_steps = (
        args.grad_accum_steps
        if args.grad_accum_steps is not None
        else 4
        if (use_mps and not force_cpu) or use_cuda
        else 2
    )
    if args.no_group_by_length:
        group_by_length = False
    elif args.group_by_length:
        group_by_length = True
    else:
        group_by_length = use_mps

    training_kwargs = dict(
        output_dir=args.output_dir.as_posix(),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        max_steps=args.max_train_steps or -1,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=gradient_checkpointing,
        logging_steps=50,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_total_limit=2,
        report_to=[],
        group_by_length=group_by_length,
        length_column_name="input_length",
        dataloader_pin_memory=not use_mps,
    )
    init_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    if "evaluation_strategy" not in init_params and "eval_strategy" in init_params:
        training_kwargs["eval_strategy"] = training_kwargs.pop("evaluation_strategy")
    if force_cpu:
        if "use_cpu" in init_params:
            training_kwargs["use_cpu"] = True
        elif "no_cuda" in init_params:
            training_kwargs["no_cuda"] = True
    training_kwargs = {
        key: value for key, value in training_kwargs.items() if key in init_params
    }
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate(test_dataset)
    print(
        f"Final evaluation -> WER: {metrics.get('eval_wer'):.4f} | "
        f"Accuracy: {metrics.get('eval_accuracy'):.4f}"
    )


if __name__ == "__main__":
    main()
