from __future__ import annotations

import inspect
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from training.candidate_gate import promotion_gate


MAX_CORRECTION_REPEAT = 64


class TokenizedTextDataset(TorchDataset):
    def __init__(self, frame, tokenizer, max_length):
        texts = frame["text"].astype(str).tolist()
        labels = frame["label"].astype(int).tolist()

        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: torch.tensor(values[index])
            for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long,
        )
        return item


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp.replace(path)


def _load_binary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {path}"
        )

    frame = pd.read_csv(path)

    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} missing columns: {sorted(missing)}"
        )

    frame = frame[["text", "label"]].dropna().copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["text"].str.len() > 0]
    frame["label"] = frame["label"].astype(int)

    invalid = sorted(set(frame["label"]) - {0, 1})
    if invalid:
        raise ValueError(
            f"{path} contains non-binary labels: {invalid}"
        )

    return frame.reset_index(drop=True)


def _load_included_corrections(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Included correction audit file not found: {path}"
        )

    frame = pd.read_csv(path)

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "correction_id",
                "text",
                "training_label",
            ]
        )

    required = {
        "correction_id",
        "text",
        "training_label",
    }

    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} missing columns: {sorted(missing)}"
        )

    frame = frame.copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["training_label"] = frame["training_label"].astype(int)

    invalid = sorted(set(frame["training_label"]) - {0, 1})
    if invalid:
        raise ValueError(
            f"{path} contains invalid training labels: {invalid}"
        )

    return frame.reset_index(drop=True)


def latest_ready_dataset_run(root: Path) -> Path:
    retraining_root = root / "data" / "retraining"

    if not retraining_root.exists():
        raise RuntimeError(
            "No data/retraining directory found. "
            "Run Milestone 2.2A first."
        )

    candidates = []

    for manifest_path in retraining_root.glob(
        "dataset_*/dataset_manifest.json"
    ):
        try:
            manifest = _read_json(manifest_path)
        except Exception:
            continue

        if manifest.get("ready_for_candidate_training"):
            candidates.append(
                (
                    str(manifest.get("created_at_utc", "")),
                    manifest_path.parent,
                )
            )

    if not candidates:
        raise RuntimeError(
            "No READY Milestone 2.2A snapshot found."
        )

    candidates.sort(
        key=lambda item: (item[0], str(item[1]))
    )

    return candidates[-1][1]


def _thresholds(active_model: Path):
    path = active_model / "thresholds.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Frozen threshold file not found: {path}"
        )

    data = _read_json(path)

    low = float(data["low_threshold"])
    high = float(data["high_threshold"])

    if not 0 <= low < high <= 1:
        raise ValueError(
            f"Invalid thresholds: low={low}, high={high}"
        )

    return low, high


def build_effective_training_frame(
    *,
    candidate_train: pd.DataFrame,
    included_corrections: pd.DataFrame,
    correction_repeat: int,
    seed: int,
):
    correction_repeat = int(correction_repeat)

    if correction_repeat < 1:
        raise ValueError(
            "correction_repeat must be at least 1."
        )

    if correction_repeat > MAX_CORRECTION_REPEAT:
        raise ValueError(
            f"correction_repeat cannot exceed {MAX_CORRECTION_REPEAT}."
        )

    unique_count = int(len(included_corrections))
    replay_rows_added = 0

    replay_frame = pd.DataFrame(
        columns=["text", "label"]
    )

    if unique_count > 0 and correction_repeat > 1:
        extra_copies = correction_repeat - 1

        replay_frame = pd.concat(
            [
                included_corrections[
                    ["text", "training_label"]
                ].rename(
                    columns={"training_label": "label"}
                )
            ]
            * extra_copies,
            ignore_index=True,
        )

        replay_rows_added = int(len(replay_frame))

    effective = pd.concat(
        [
            candidate_train[["text", "label"]].copy(),
            replay_frame,
        ],
        ignore_index=True,
    )

    effective = effective.sample(
        frac=1.0,
        random_state=int(seed),
    ).reset_index(drop=True)

    replay_metadata = {
        "strategy": (
            "bounded duplicate replay during candidate training only"
        ),
        "correction_repeat_total_appearances": correction_repeat,
        "unique_eligible_corrections": unique_count,
        "replay_rows_added": replay_rows_added,
        "candidate_snapshot_rows_before_replay": int(
            len(candidate_train)
        ),
        "effective_training_rows_after_replay": int(
            len(effective)
        ),
        "max_allowed_repeat": MAX_CORRECTION_REPEAT,
        "interpretation": (
            "Replay copies increase optimization weight. "
            "They are not new independent annotations."
        ),
    }

    return effective, replay_metadata


def _tokenized(frame, tokenizer, max_length):
    return TokenizedTextDataset(
        frame,
        tokenizer,
        max_length,
    )


def _hf_metrics(eval_pred):
    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels, predictions, zero_division=0
        ),
        "recall": recall_score(
            labels, predictions, zero_division=0
        ),
        "f1": f1_score(
            labels, predictions, zero_division=0
        ),
    }


def _training_arguments(
    output_dir,
    epochs,
    learning_rate,
    train_batch_size,
    eval_batch_size,
    warmup_steps,
    seed,
):
    parameters = inspect.signature(
        TrainingArguments.__init__
    ).parameters

    kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=float(epochs),
        learning_rate=float(learning_rate),
        per_device_train_batch_size=int(train_batch_size),
        per_device_eval_batch_size=int(eval_batch_size),
        weight_decay=0.01,
        warmup_steps=int(warmup_steps),
        logging_steps=25,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=int(seed),
        data_seed=int(seed),
        report_to=[],
        save_total_limit=2,
    )

    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        raise RuntimeError(
            "Unsupported transformers version: "
            "no evaluation strategy argument."
        )

    return TrainingArguments(**kwargs)


def _trainer(
    model,
    tokenizer,
    args,
    train_dataset,
    eval_dataset,
):
    parameters = inspect.signature(
        Trainer.__init__
    ).parameters

    kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_hf_metrics,
    )

    if "processing_class" in parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in parameters:
        kwargs["tokenizer"] = tokenizer

    return Trainer(**kwargs)


def _probabilities(
    model,
    tokenizer,
    texts,
    batch_size,
    max_length,
    device,
):
    result = []

    model.to(device)
    model.eval()

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            logits = model(**encoded).logits

        probs = torch.softmax(logits, dim=-1)[:, 1]

        result.extend(
            probs.detach().cpu().numpy().tolist()
        )

    return np.asarray(result, dtype=float)


def _evaluate(
    model,
    tokenizer,
    frame,
    low,
    high,
    batch_size,
    max_length,
    device,
):
    probs = _probabilities(
        model,
        tokenizer,
        frame["text"].astype(str).tolist(),
        batch_size,
        max_length,
        device,
    )

    labels = frame["label"].astype(int).to_numpy()
    binary = (probs >= 0.5).astype(int)

    keep = (probs < low) | (probs >= high)
    selective = (probs[keep] >= high).astype(int)

    return {
        "size": int(len(frame)),
        "binary_at_0_5": {
            "accuracy": float(
                accuracy_score(labels, binary)
            ),
            "precision": float(
                precision_score(
                    labels, binary, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    labels, binary, zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    labels, binary, zero_division=0
                )
            ),
            "roc_auc": (
                float(roc_auc_score(labels, probs))
                if len(set(labels.tolist())) == 2
                else None
            ),
            "confusion_matrix": confusion_matrix(
                labels,
                binary,
                labels=[0, 1],
            ).tolist(),
        },
        "three_state_policy": {
            "low_threshold": low,
            "high_threshold": high,
            "coverage": float(keep.mean()),
            "abstention_rate": float(1.0 - keep.mean()),
            "classified_examples": int(keep.sum()),
            "ambiguous_examples": int((~keep).sum()),
            "accuracy_on_classified_examples": (
                float(
                    accuracy_score(
                        labels[keep],
                        selective,
                    )
                )
                if keep.any()
                else None
            ),
        },
    }


def _retention(
    candidate_model,
    tokenizer,
    included_csv,
    batch_size,
    max_length,
    device,
):
    frame = _load_included_corrections(included_csv)

    if frame.empty:
        return {
            "size": 0,
            "correct": 0,
            "accuracy": None,
            "items": [],
        }

    probs = _probabilities(
        candidate_model,
        tokenizer,
        frame["text"].astype(str).tolist(),
        batch_size,
        max_length,
        device,
    )

    labels = frame["training_label"].astype(int).to_numpy()
    predictions = (probs >= 0.5).astype(int)
    correct = int((predictions == labels).sum())

    items = []

    for row, prob, prediction in zip(
        frame.to_dict(orient="records"),
        probs.tolist(),
        predictions.tolist(),
    ):
        items.append(
            {
                "correction_id": int(row["correction_id"]),
                "text": row["text"],
                "training_label": int(row["training_label"]),
                "humor_score": float(prob),
                "predicted_label": int(prediction),
                "correct": bool(
                    prediction == int(row["training_label"])
                ),
            }
        )

    return {
        "size": int(len(frame)),
        "correct": correct,
        "accuracy": float(correct / len(frame)),
        "items": items,
        "note": (
            "Retention is measured on training corrections. "
            "It is a sanity check, not an independent metric."
        ),
    }


def _delta(candidate, active):
    result = {}

    for key in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ):
        candidate_value = candidate["binary_at_0_5"].get(key)
        active_value = active["binary_at_0_5"].get(key)

        result[key] = (
            None
            if candidate_value is None or active_value is None
            else float(candidate_value) - float(active_value)
        )

    return result


def train_candidate(
    *,
    root: Path,
    dataset_dir: Path | None = None,
    correction_repeat: int = 32,
    epochs: float = 1.0,
    learning_rate: float = 3e-6,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    max_length: int = 128,
    seed: int = 42,
    gate_tolerance: float = 0.005,
    include_test: bool = True,
):
    root = Path(root).resolve()

    dataset_dir = (
        Path(dataset_dir).resolve()
        if dataset_dir
        else latest_ready_dataset_run(root)
    )

    manifest = _read_json(
        dataset_dir / "dataset_manifest.json"
    )

    if not manifest.get("ready_for_candidate_training"):
        raise RuntimeError(
            "Selected 2.2A snapshot is not READY."
        )

    candidate_train = _load_binary(
        dataset_dir / "candidate_train.csv"
    )

    included = _load_included_corrections(
        dataset_dir / "corrections_included.csv"
    )

    if included.empty:
        raise RuntimeError(
            "No eligible binary corrections exist "
            "in the selected dataset snapshot."
        )

    effective_train, replay = build_effective_training_frame(
        candidate_train=candidate_train,
        included_corrections=included,
        correction_repeat=correction_repeat,
        seed=seed,
    )

    validation = _load_binary(
        root / "data" / "processed" / "validation.csv"
    )

    test = (
        _load_binary(
            root / "data" / "processed" / "test.csv"
        )
        if include_test
        else None
    )

    active_dir = root / "models" / "humor_transformer"
    low, high = _thresholds(active_dir)

    candidate_id = (
        datetime.now(timezone.utc)
        .strftime("candidate_%Y%m%dT%H%M%SZ_")
        + uuid4().hex[:8]
    )

    candidate_dir = (
        root / "models" / "candidates" / candidate_id
    )

    model_dir = candidate_dir / "model"
    trainer_dir = candidate_dir / "trainer"

    candidate_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    metadata = {
        "milestone": "2.2B-revision2-exe",
        "candidate_id": candidate_id,
        "status": "training",
        "created_at_utc": _now(),
        "dataset_run_id": manifest["run_id"],
        "dataset_dir": str(dataset_dir),
        "active_model": str(active_dir),
        "candidate_model": str(model_dir),
        "frozen_thresholds": {
            "low": low,
            "high": high,
        },
        "training": {
            "initialization": "current active trained model",
            "epochs": float(epochs),
            "learning_rate": float(learning_rate),
            "train_batch_size": int(train_batch_size),
            "eval_batch_size": int(eval_batch_size),
            "max_length": int(max_length),
            "seed": int(seed),
        },
        "correction_replay": replay,
        "test_policy": (
            "Held-out test is report-only and not used by the promotion gate."
            if include_test
            else "Held-out test skipped."
        ),
    }

    _write_json(
        metadata,
        candidate_dir / "candidate_metadata.json",
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(active_dir)

    active_model = (
        AutoModelForSequenceClassification
        .from_pretrained(active_dir)
    )

    candidate_model = (
        AutoModelForSequenceClassification
        .from_pretrained(active_dir)
    )

    train_dataset = _tokenized(
        effective_train,
        tokenizer,
        max_length,
    )

    validation_dataset = _tokenized(
        validation,
        tokenizer,
        max_length,
    )

    steps_per_epoch = math.ceil(
        len(effective_train) / max(1, train_batch_size)
    )

    warmup_steps = max(
        1,
        int(
            steps_per_epoch
            * epochs
            * 0.05
        ),
    )

    args = _training_arguments(
        trainer_dir,
        epochs,
        learning_rate,
        train_batch_size,
        eval_batch_size,
        warmup_steps,
        seed,
    )

    trainer = _trainer(
        candidate_model,
        tokenizer,
        args,
        train_dataset,
        validation_dataset,
    )

    try:
        train_result = trainer.train()

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)

        shutil.copy2(
            active_dir / "thresholds.json",
            model_dir / "thresholds.json",
        )

        candidate_model = trainer.model

        active_validation = _evaluate(
            active_model,
            tokenizer,
            validation,
            low,
            high,
            eval_batch_size,
            max_length,
            device,
        )

        candidate_validation = _evaluate(
            candidate_model,
            tokenizer,
            validation,
            low,
            high,
            eval_batch_size,
            max_length,
            device,
        )

        retention = _retention(
            candidate_model,
            tokenizer,
            dataset_dir / "corrections_included.csv",
            eval_batch_size,
            max_length,
            device,
        )

        active_test = None
        candidate_test = None

        if test is not None:
            active_test = _evaluate(
                active_model,
                tokenizer,
                test,
                low,
                high,
                eval_batch_size,
                max_length,
                device,
            )

            candidate_test = _evaluate(
                candidate_model,
                tokenizer,
                test,
                low,
                high,
                eval_batch_size,
                max_length,
                device,
            )

        gate = promotion_gate(
            active_validation=active_validation,
            candidate_validation=candidate_validation,
            correction_retention=retention,
            tolerance=gate_tolerance,
        )

        comparison = {
            "candidate_id": candidate_id,
            "dataset_run_id": manifest["run_id"],
            "evaluated_at_utc": _now(),
            "correction_replay": replay,
            "validation": {
                "active": active_validation,
                "candidate": candidate_validation,
                "delta": _delta(
                    candidate_validation,
                    active_validation,
                ),
            },
            "correction_retention": retention,
            "test": (
                {
                    "active": active_test,
                    "candidate": candidate_test,
                    "delta": _delta(
                        candidate_test,
                        active_test,
                    ),
                    "warning": (
                        "Report-only. Test metrics are not "
                        "used by the promotion gate."
                    ),
                }
                if active_test is not None
                else None
            ),
            "promotion_gate": gate,
            "active_model_unchanged": True,
        }

        _write_json(
            comparison,
            candidate_dir / "comparison.json",
        )

        metadata.update(
            {
                "status": (
                    "passed"
                    if gate["passed"]
                    else "not_passed"
                ),
                "completed_at_utc": _now(),
                "train_metrics": train_result.metrics,
                "comparison": str(
                    candidate_dir / "comparison.json"
                ),
                "promotion_gate_passed": gate["passed"],
                "active_model_unchanged": True,
            }
        )

        _write_json(
            metadata,
            candidate_dir / "candidate_metadata.json",
        )

        _write_json(
            {
                "candidate_id": candidate_id,
                "candidate_dir": str(candidate_dir),
                "model_dir": str(model_dir),
                "status": metadata["status"],
                "promotion_gate_passed": gate["passed"],
                "correction_repeat": correction_repeat,
                "learning_rate": learning_rate,
                "updated_at_utc": _now(),
            },
            root / "models" / "candidate_current.json",
        )

        return {
            "metadata": metadata,
            "comparison": comparison,
        }

    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "completed_at_utc": _now(),
                "error": str(exc),
            }
        )

        _write_json(
            metadata,
            candidate_dir / "candidate_metadata.json",
        )

        raise
