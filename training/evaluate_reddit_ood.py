from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "humor_transformer"
OOD_DIR = ROOT / "data" / "ood"


def normalize_annotation(value: object):
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "humor": 1,
        "humorous": 1,
        "1": 1,
        "not_humor": 0,
        "not_humorous": 0,
        "non_humor": 0,
        "0": 0,
        "ambiguous": None,
        "uncertain": None,
        "": None,
        "nan": None,
    }
    if raw not in mapping:
        raise ValueError(f"Unknown annotation label: {value!r}")
    return mapping[raw]


def predict_probabilities(texts, tokenizer, model, device, batch_size=64):
    probs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
        probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist())
    return np.asarray(probs, dtype=float)


def evaluate(labels, probs, low, high):
    labels = np.asarray(labels, dtype=int)
    binary_pred = (probs >= 0.5).astype(int)

    keep = (probs < low) | (probs >= high)
    selective_pred = (probs[keep] >= high).astype(int)

    result = {
        "size": int(len(labels)),
        "binary_at_0_5": {
            "accuracy": float(accuracy_score(labels, binary_pred)),
            "precision": float(precision_score(labels, binary_pred, zero_division=0)),
            "recall": float(recall_score(labels, binary_pred, zero_division=0)),
            "f1": float(f1_score(labels, binary_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, probs)) if len(set(labels)) == 2 else None,
            "confusion_matrix": confusion_matrix(labels, binary_pred, labels=[0, 1]).tolist(),
        },
        "three_state_policy": {
            "low_threshold": low,
            "high_threshold": high,
            "coverage": float(keep.mean()),
            "abstention_rate": float(1.0 - keep.mean()),
            "accuracy_on_classified_examples": (
                float(accuracy_score(labels[keep], selective_pred))
                if keep.any() else None
            ),
            "classified_examples": int(keep.sum()),
            "ambiguous_model_outputs": int((~keep).sum()),
        },
    }
    return result, binary_pred, keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        choices=["source", "adjudicated"],
        default="source",
    )
    args = parser.parse_args()

    thresholds = json.loads(
        (MODEL_DIR / "thresholds.json").read_text(encoding="utf-8")
    )
    low = float(thresholds["low_threshold"])
    high = float(thresholds["high_threshold"])

    annotation_stats = None

    if args.labels == "source":
        path = OOD_DIR / "reddit_ood_provisional.csv"
        df = pd.read_csv(path)
        labels = df["source_label"].astype(int).to_numpy()
        label_kind = "provisional_source_niche"

    else:
        path = OOD_DIR / "reddit_ood_annotation.csv"
        df = pd.read_csv(path)

        a1 = df["annotator_1"].map(normalize_annotation)
        a2 = df["annotator_2"].map(normalize_annotation)
        comparable = a1.notna() & a2.notna()

        kappa = None
        raw_agreement = None
        if comparable.any():
            a1c = a1[comparable].astype(int)
            a2c = a2[comparable].astype(int)
            kappa = float(cohen_kappa_score(a1c, a2c))
            raw_agreement = float((a1c.to_numpy() == a2c.to_numpy()).mean())

        final = df["adjudicated_label"].map(normalize_annotation)
        usable = final.notna()

        annotation_stats = {
            "annotator_comparable_examples": int(comparable.sum()),
            "raw_agreement": raw_agreement,
            "cohen_kappa": kappa,
            "final_binary_examples": int(usable.sum()),
            "unresolved_or_ambiguous_examples": int((~usable).sum()),
            "note": (
                "Agreement is reported as observed. Unresolved examples may "
                "remain ambiguous rather than being forced into a binary label."
            ),
        }

        df = df.loc[usable].copy()
        labels = final.loc[usable].astype(int).to_numpy()

        if len(labels) == 0:
            raise RuntimeError(
                "No adjudicated binary labels found in reddit_ood_annotation.csv."
            )

        label_kind = "human_adjudicated"

    texts = df["text"].astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    probs = predict_probabilities(texts, tokenizer, model, device)
    result, binary_pred, keep = evaluate(labels, probs, low, high)

    payload = {
        "evaluation": "Reddit out-of-distribution",
        "label_kind": label_kind,
        "thresholds_frozen_before_ood": {"low": low, "high": high},
        "metrics": result,
        "annotation": annotation_stats,
        "warning": (
            "Do not retune the model or thresholds on this frozen OOD set "
            "and continue to describe the same set as an independent final test."
        ),
    }

    output_name = (
        "reddit_ood_metrics_source.json"
        if args.labels == "source"
        else "reddit_ood_metrics_adjudicated.json"
    )
    output_path = MODEL_DIR / output_name
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    prediction_path = OOD_DIR / (
        "reddit_ood_predictions_source.csv"
        if args.labels == "source"
        else "reddit_ood_predictions_adjudicated.csv"
    )

    out = df[["ood_id", "text"]].copy()
    out["gold_label"] = labels
    out["humor_score"] = probs
    out["binary_prediction"] = binary_pred
    out["model_state"] = np.where(
        probs < low,
        "not_humorous",
        np.where(probs >= high, "humorous", "ambiguous"),
    )
    out["classified_by_three_state_policy"] = keep
    out.to_csv(prediction_path, index=False, encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print()
    print(f"Saved metrics:     {output_path}")
    print(f"Saved predictions: {prediction_path}")


if __name__ == "__main__":
    main()
