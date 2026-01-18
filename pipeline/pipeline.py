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
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_dataset
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

from resources.datasets import Datasets

TEXT_COLUMN_CANDIDATES = (
    "sentence_checked",
    "text_spt",
    "predicted_sentence",
    "text",
    "transcript",
    "transcription",
    "ref_normalized",
)

PATH_COLUMN_CANDIDATES = ("path", "audio", "file", "file_path")
HF_AUDIO_COLUMN_CANDIDATES = ("audio", "file", "path")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    csv_path: Path
    audio_dir: Optional[Path] = None
    hf_repo_id: Optional[str] = None


def build_dataset_configs(data_root: Path) -> List[DatasetConfig]:
    return [
        DatasetConfig(
            name="feruza_speech",
            csv_path=data_root / "feruza-speech" / "google_spt_transcriptions.csv",
            audio_dir=data_root / "feruza-speech",
            hf_repo_id=Datasets.FERUZA_SPEECH,
        ),
        DatasetConfig(
            name="uzbek_speech_corpus",
            csv_path=data_root
            / "uzbek-speech-corpus"
            / "data"
            / "google_spt_transcriptions.csv",
            audio_dir=data_root / "uzbek-speech-corpus" / "data",
            hf_repo_id=Datasets.UZBEK_SPEECH_CORPUS,
        ),
        DatasetConfig(
            name="it_youtube_uzbek_speech",
            csv_path=data_root
            / "it_youtube_uzbek_speech_dataset"
            / "data"
            / "it_dataset_checked.csv",
            audio_dir=data_root / "it_youtube_uzbek_speech_dataset" / "data",
            hf_repo_id=Datasets.IT_YOUTUBE_UZBEK_SPEECH,
        ),
        DatasetConfig(
            name="news_youtube_uzbek_speech",
            csv_path=data_root
            / "news_youtube_uzbek_speech_dataset"
            / "data"
            / "news_dataset_checked.csv",
            audio_dir=data_root / "news_youtube_uzbek_speech_dataset" / "data",
            hf_repo_id=Datasets.NEWS_YOUTUBE_UZBEK_SPEECH,
        ),
        DatasetConfig(
            name="uzbekvoice_dataset",
            csv_path=data_root / "uzbekvoice_dataset" / "data" / "google_spt_transcriptions.csv",
            audio_dir=data_root / "uzbekvoice_dataset" / "data",
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
    return " ".join(str(text).lower().split())


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


def _load_hf_dataset(
    config: DatasetConfig,
    sample_fraction: float,
    seed: int,
    hf_config: Optional[str],
    hf_split: Optional[str],
) -> Tuple[str, Dataset]:
    if not config.hf_repo_id:
        raise ValueError(f"No Hugging Face repo configured for {config.name}.")

    if hf_split:
        dataset = load_dataset(config.hf_repo_id, hf_config, split=hf_split)
    else:
        dataset = load_dataset(config.hf_repo_id, hf_config)

    if isinstance(dataset, DatasetDict):
        dataset = concatenate_datasets([dataset[split] for split in dataset.keys()])

    text_column = _select_column(TEXT_COLUMN_CANDIDATES, dataset.column_names)
    audio_column = _select_column(HF_AUDIO_COLUMN_CANDIDATES, dataset.column_names)
    if not text_column or not audio_column:
        raise ValueError(
            f"Hugging Face dataset {config.hf_repo_id} is missing audio/text columns."
        )

    if text_column != "text":
        dataset = dataset.rename_column(text_column, "text")
    if audio_column != "audio":
        dataset = dataset.rename_column(audio_column, "audio")
    dataset = dataset.filter(
        lambda row: row["text"] is not None and str(row["text"]).strip() != ""
    )
    if 0 < sample_fraction < 1:
        dataset = dataset.shuffle(seed=seed)
        subset_size = max(1, int(len(dataset) * sample_fraction))
        dataset = dataset.select(range(subset_size))

    dataset = dataset.map(lambda row: {"text": _normalize_text(row["text"])})
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
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
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
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch


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
        "--data-source",
        type=str,
        default="hf",
        choices=("hf", "local"),
        help="Load audio from Hugging Face or local CSV/audio folders.",
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
    parser.add_argument(
        "--hf-config",
        type=str,
        default=None,
        help="Optional Hugging Face dataset config name.",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default=None,
        help="Optional Hugging Face dataset split (e.g. train).",
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
            if args.data_source == "hf":
                name, dataset = _load_hf_dataset(
                    config,
                    sample_fraction=args.sample_fraction,
                    seed=args.seed,
                    hf_config=args.hf_config,
                    hf_split=args.hf_split,
                )
            else:
                name, dataset = _load_csv_dataset(
                    config,
                    sample_fraction=args.sample_fraction,
                    seed=args.seed,
                    data_root=args.data_root,
                    audio_root=args.audio_root,
                    audio_map=audio_map,
                )
            print(f"[{name}] loaded {len(dataset)} samples")
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

    def compute_metrics(pred) -> dict:
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred_str = processor.batch_decode(pred_ids)
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "accuracy": 1.0 - wer}

    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name,
        vocab_size=len(processor.tokenizer),
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    model.freeze_feature_extractor()

    if args.device == "cpu":
        use_mps = False
    elif args.device == "mps":
        use_mps = True

    force_cpu = args.device == "cpu" or (args.device == "auto" and use_mps)

    per_device_train_batch_size = (
        args.train_batch_size
        if args.train_batch_size is not None
        else 1
        if use_mps and not force_cpu
        else 2
    )
    per_device_eval_batch_size = (
        args.eval_batch_size
        if args.eval_batch_size is not None
        else 1
        if use_mps and not force_cpu
        else 2
    )
    gradient_accumulation_steps = (
        args.grad_accum_steps
        if args.grad_accum_steps is not None
        else 4
        if use_mps and not force_cpu
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
        logging_steps=50,
        learning_rate=3e-4,
        warmup_steps=100,
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
