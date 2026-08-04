from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "humor_transformer"
TEST = ROOT / "data" / "processed" / "test.csv"


def main() -> None:
    thresholds = json.loads((MODEL_DIR / "thresholds.json").read_text(encoding="utf-8"))
    low, high = float(thresholds["low_threshold"]), float(thresholds["high_threshold"])
    frame = pd.read_csv(TEST).dropna(subset=["text", "label"])
    texts = frame["text"].astype(str).tolist()
    y = frame["label"].astype(int).to_numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
    probs = []
    for start in range(0, len(texts), 64):
        batch = texts[start:start+64]
        enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = model(**enc).logits
        probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist())
    probs = np.asarray(probs)
    pred05 = (probs >= 0.5).astype(int)
    keep = (probs < low) | (probs >= high)
    selected_pred = (probs[keep] >= high).astype(int)
    result = {
        "test_size": int(len(y)),
        "binary_at_0_5": {
            "accuracy": float(accuracy_score(y, pred05)),
            "precision": float(precision_score(y, pred05, zero_division=0)),
            "recall": float(recall_score(y, pred05, zero_division=0)),
            "f1": float(f1_score(y, pred05, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probs)),
            "confusion_matrix": confusion_matrix(y, pred05).tolist(),
        },
        "three_state_policy": {
            "low_threshold": low,
            "high_threshold": high,
            "coverage": float(keep.mean()),
            "abstention_rate": float(1.0 - keep.mean()),
            "accuracy_on_classified_examples": float(accuracy_score(y[keep], selected_pred)) if keep.any() else None,
            "classified_examples": int(keep.sum()),
            "ambiguous_examples": int((~keep).sum()),
        },
        "warning": "In-domain test only. OOD evaluation must be reported separately.",
    }
    path = MODEL_DIR / "test_metrics.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
