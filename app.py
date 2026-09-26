from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core import classifier
from feedback_store import (
    VALID_LABELS,
    VALID_STATUSES,
    add_correction,
    correction_counts,
    delete_correction,
    feedback_db_path,
    initialize_database,
    list_corrections,
    update_correction,
)
from generator import generate_reply
from model_admin import ModelAdminError, ModelAdminService


SOURCE_ROOT = Path(__file__).resolve().parent

if getattr(sys, "frozen", False):
    RESOURCE_ROOT = Path(
        getattr(sys, "_MEIPASS", SOURCE_ROOT)
    ).resolve()

    RUNTIME_ROOT = Path(
        os.environ.get(
            "HUMOR_RUNTIME_ROOT",
            Path(
                os.environ.get(
                    "LOCALAPPDATA",
                    Path.home() / "AppData" / "Local",
                )
            )
            / "HumorBot",
        )
    ).resolve()
else:
    RESOURCE_ROOT = SOURCE_ROOT
    RUNTIME_ROOT = Path(os.getenv("HUMOR_RUNTIME_ROOT", str(SOURCE_ROOT))).resolve()

STATIC_DIR = RESOURCE_ROOT / "static"

# In a packaged build the launcher loads .env first, but this keeps direct
# Python development and alternate launchers working as well.
load_dotenv(RUNTIME_ROOT / ".env")
load_dotenv(SOURCE_ROOT / ".env")

initialize_database()

model_admin = ModelAdminService(RUNTIME_ROOT)

app = FastAPI(
    title="Humor Detection Application",
    version="0.2.3-milestone2-final",
    description=(
        "Humor classification, persistent correction review, "
        "controlled candidate retraining, promotion and rollback."
    ),
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


class FeedbackRequest(BaseModel):
    input_text: str = Field(min_length=1, max_length=8000)
    original_label: str
    original_score: float = Field(ge=0.0, le=1.0)
    corrected_label: str
    source: str = "analyze"
    model_source: str = ""


class CorrectionUpdateRequest(BaseModel):
    corrected_label: str | None = None
    status: str | None = None
    notes: str | None = Field(default=None, max_length=2000)


class PromotionRequest(BaseModel):
    confirmation: str


class RollbackRequest(BaseModel):
    confirmation: str


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin")
def admin_page():
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/health")
def health():
    local_ready = (
        classifier.local_model_path.exists()
        and (classifier.local_model_path / "config.json").exists()
    )

    counts = correction_counts()

    return {
        "status": "ok",
        "version": app.version,
        "local_model_ready": local_ready,
        "model_loaded": classifier.model is not None,
        "model_source": classifier.model_source or None,
        "bootstrap_allowed": classifier.allow_bootstrap,
        "thresholds": {
            "low": classifier.thresholds.low,
            "high": classifier.thresholds.high,
        },
        "gemini_configured": bool(
            os.getenv("GEMINI_API_KEY", "").strip()
        ),
        "feedback_count": counts["total"],
        "feedback_pending": counts["pending"],
        "active_model": model_admin.active_model_info(),
        "runtime_root": str(RUNTIME_ROOT),
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
        reply, response_source = generate_reply(payload.message, analysis)

        return {
            "reply": reply,
            "response_source": response_source,
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


@app.post("/feedback")
def save_feedback(payload: FeedbackRequest):
    if (
        payload.original_label not in VALID_LABELS
        or payload.corrected_label not in VALID_LABELS
    ):
        raise HTTPException(
            status_code=400,
            detail="Unknown humor label.",
        )

    try:
        saved = add_correction(
            input_text=payload.input_text,
            original_label=payload.original_label,
            original_score=payload.original_score,
            corrected_label=payload.corrected_label,
            source=payload.source,
            model_source=payload.model_source,
        )

        return {
            "ok": True,
            "message": "Correction saved for review.",
            "correction": saved,
            "counts": correction_counts(),
        }
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/feedback/count")
def feedback_count():
    return correction_counts()


@app.get("/admin/corrections")
def admin_corrections(
    status: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=2000),
):
    if status is not None and status not in VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail="Unknown correction status.",
        )

    return {
        "items": list_corrections(status=status, limit=limit),
        "counts": correction_counts(),
        "database": str(feedback_db_path()),
    }


@app.put("/admin/corrections/{correction_id}")
def admin_update_correction(
    correction_id: int,
    payload: CorrectionUpdateRequest,
):
    try:
        updated = update_correction(
            correction_id,
            corrected_label=payload.corrected_label,
            status=payload.status,
            notes=payload.notes,
        )

        return {
            "ok": True,
            "correction": updated,
            "counts": correction_counts(),
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/admin/corrections/{correction_id}")
def admin_delete_correction(correction_id: int):
    try:
        delete_correction(correction_id)
        return {
            "ok": True,
            "counts": correction_counts(),
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/admin/status")
def admin_status():
    try:
        return model_admin.status_payload()
    except ModelAdminError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/admin/retrain")
def admin_retrain():
    try:
        status = model_admin.start_retraining()
        return {
            "ok": True,
            "message": "Candidate retraining started.",
            "status": status,
        }
    except ModelAdminError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/admin/promote")
def admin_promote(payload: PromotionRequest):
    try:
        return model_admin.promote_candidate(payload.confirmation)
    except ModelAdminError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/admin/rollback")
def admin_rollback(payload: RollbackRequest):
    try:
        return model_admin.rollback_latest(payload.confirmation)
    except ModelAdminError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/admin/history")
def admin_history():
    return {
        "items": model_admin.promotion_history()
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
