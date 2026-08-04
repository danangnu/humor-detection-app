from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "humor_transformer"
BASE_MODEL = "distilroberta-base"


def load_split(name: str) -> Dataset:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run prepare_socket_dataset.py first.")
    frame = pd.read_csv(path)[["text", "label"]].dropna()
    frame["label"] = frame["label"].astype(int)
    return Dataset.from_pandas(frame, preserve_index=False)


def metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, pred),
        "precision": precision_score(labels, pred, zero_division=0),
        "recall": recall_score(labels, pred, zero_division=0),
        "f1": f1_score(labels, pred, zero_division=0),
    }


def main() -> None:
    train_ds, val_ds = load_split("train"), load_split("validation")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={0: "not_humor", 1: "humor"},
        label2id={"not_humor": 0, "humor": 1},
    )
    args = TrainingArguments(
        output_dir=str(ROOT / "training" / "runs" / "humor_m1"),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        report_to=[],
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=metrics,
    )
    trainer.train()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    result = trainer.evaluate(val_ds)
    (MODEL_DIR / "validation_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Model saved to {MODEL_DIR}")
    print("Next: python training/calibrate_thresholds.py")


if __name__ == "__main__":
    main()
