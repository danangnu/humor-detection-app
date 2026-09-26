from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class HumorThresholds:
    low: float = 0.40
    high: float = 0.60

    def validate(self) -> "HumorThresholds":
        if not (0.0 <= self.low < self.high <= 1.0):
            raise ValueError("Expected 0 <= low threshold < high threshold <= 1.")
        return self


class HumorClassifier:
    """Binary P(humor) model plus a three-state abstention policy."""

    def __init__(self) -> None:
        configured = os.getenv("HUMOR_MODEL_PATH", "models/humor_transformer")
        self.local_model_path = (BASE_DIR / configured).resolve()
        self.allow_bootstrap = os.getenv("ALLOW_BOOTSTRAP_MODEL", "false").lower() in {
            "1", "true", "yes", "on"
        }
        self.bootstrap_model_id = os.getenv(
            "BOOTSTRAP_MODEL_ID", "VitalContribution/JokeDetectBERT"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.model_source = ""
        self.thresholds = self._load_thresholds()

    def _load_thresholds(self) -> HumorThresholds:
        path = self.local_model_path / "thresholds.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return HumorThresholds(
                float(data["low_threshold"]), float(data["high_threshold"])
            ).validate()
        return HumorThresholds(
            float(os.getenv("HUMOR_LOW_THRESHOLD", "0.40")),
            float(os.getenv("HUMOR_HIGH_THRESHOLD", "0.60")),
        ).validate()

    def load(self) -> None:
        if self.model is not None:
            return
        if self.local_model_path.exists() and (self.local_model_path / "config.json").exists():
            ref = str(self.local_model_path)
            self.model_source = "local-trained-model"
        elif self.allow_bootstrap:
            ref = self.bootstrap_model_id
            self.model_source = f"bootstrap:{self.bootstrap_model_id}"
        else:
            raise RuntimeError(
                "No local Humor Bot model found. Run training/train_transformer.py first."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(ref, local_files_only=bool(os.getenv("DEPLOY_MODEL_DIR")), trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(ref, local_files_only=bool(os.getenv("DEPLOY_MODEL_DIR")), trust_remote_code=False, use_safetensors=True)
        self.model.to(self.device).eval()

    def _humor_probability(self, logits: torch.Tensor) -> float:
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu()
        id2label = getattr(self.model.config, "id2label", {}) or {}
        positive = None
        for idx, raw in id2label.items():
            label = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
            if label in {"humor", "humorous", "joke", "funny", "label_1"}:
                positive = int(idx)
                break
        if positive is None:
            if len(probs) != 2:
                raise RuntimeError("Configured model is not a recognizable binary humor model.")
            positive = 1
        return float(probs[positive].item())

    def _decision(self, score: float) -> tuple[str, str]:
        low = self.thresholds.low
        high = self.thresholds.high

        if score < low:
            label = "Not Humorous"

            # Measure how far the score has moved from the
            # low decision threshold toward 0.
            strength = (low - score) / low if low > 0 else 1.0

        elif score >= high:
            label = "Humorous"

            # Measure how far the score has moved from the
            # high decision threshold toward 1.
            strength = (
                (score - high) / (1.0 - high)
                if high < 1.0
                else 1.0
            )

        else:
            return "Ambiguous / Uncertain", "Low"

        if strength >= 0.75:
            confidence = "High"
        elif strength >= 0.35:
            confidence = "Moderate"
        else:
            confidence = "Low"

        return label, confidence

    def analyze(self, text: str) -> dict[str, Any]:
        cleaned = " ".join(text.split()).strip()
        if not cleaned:
            raise ValueError("Text cannot be empty.")
        self.load()
        batch = self.tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=256)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode():
            logits = self.model(**batch).logits
        score = self._humor_probability(logits)
        label, confidence = self._decision(score)
        if label == "Humorous":
            explanation = "The model score is above the current humor threshold."
        elif label == "Not Humorous":
            explanation = "The model score is below the current humor threshold."
        else:
            explanation = "The score falls inside the uncertain band, so the app does not force a yes-or-no label."
        return {
            "score": round(score, 6),
            "label": label,
            "confidence": confidence,
            "is_humorous": True if label == "Humorous" else False if label == "Not Humorous" else None,
            "low_threshold": self.thresholds.low,
            "high_threshold": self.thresholds.high,
            "model_source": self.model_source,
            "explanation": explanation,
        }


classifier = HumorClassifier()
