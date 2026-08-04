from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "humor_transformer"
VAL = ROOT / "data" / "processed" / "validation.csv"


def get_probs(texts, tokenizer, model, device, batch_size=64):
    out = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = model(**enc).logits
        out.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist())
    return np.asarray(out)


def main() -> None:
    frame = pd.read_csv(VAL).dropna(subset=["text", "label"])
    texts = frame["text"].astype(str).tolist()
    y = frame["label"].astype(int).to_numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
    probs = get_probs(texts, tok, model, device)

    best = None
    for low in np.arange(0.20, 0.50, 0.02):
        for high in np.arange(0.52, 0.82, 0.02):
            keep = (probs < low) | (probs >= high)
            coverage = float(keep.mean())
            if coverage < 0.80 or keep.sum() == 0:
                continue
            pred = (probs[keep] >= high).astype(int)
            acc = float(accuracy_score(y[keep], pred))
            candidate = (acc, coverage, -(high-low), float(low), float(high))
            if best is None or candidate > best:
                best = candidate
    if best is None:
        raise RuntimeError("No threshold pair met the >=80% coverage rule.")
    acc, coverage, _, low, high = best
    payload = {
        "low_threshold": round(low, 4),
        "high_threshold": round(high, 4),
        "validation_selective_accuracy": round(acc, 6),
        "validation_coverage": round(coverage, 6),
        "selection_rule": "maximize validation accuracy subject to >=80% coverage",
        "note": "Selected on validation only; do not retune on test or OOD data.",
    }
    (MODEL_DIR / "thresholds.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
