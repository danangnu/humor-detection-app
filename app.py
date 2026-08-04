from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from core import classifier
from generator import generate_reply

STATIC_DIR = BASE_DIR / "static"
app = FastAPI(title="Humor Detection Application", version="0.1.0-milestone1")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    local_ready = classifier.local_model_path.exists() and (classifier.local_model_path / "config.json").exists()
    return {
        "status": "ok",
        "version": app.version,
        "local_model_ready": local_ready,
        "model_loaded": classifier.model is not None,
        "model_source": classifier.model_source or None,
        "bootstrap_allowed": classifier.allow_bootstrap,
        "thresholds": {"low": classifier.thresholds.low, "high": classifier.thresholds.high},
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
    }


@app.post("/analyze")
def analyze(payload: AnalyzeRequest):
    try:
        return classifier.analyze(payload.text)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/chat")
def chat(payload: ChatRequest):
    try:
        analysis = classifier.analyze(payload.message)
        reply, source = generate_reply(payload.message, analysis)
        return {
            "reply": reply,
            "response_source": source,
            "user_humor_score": analysis["score"],
            "user_label": analysis["label"],
            "user_confidence": analysis["confidence"],
            "user_is_humorous": analysis["is_humorous"],
            "user_explanation": analysis["explanation"],
            "low_threshold": analysis["low_threshold"],
            "high_threshold": analysis["high_threshold"],
            "model_source": analysis["model_source"],
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
