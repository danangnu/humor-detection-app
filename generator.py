from __future__ import annotations

import os
from typing import Any


def _fallback(analysis: dict[str, Any]) -> str:
    if analysis["label"] == "Humorous":
        return "I caught the joke. The model reads that as humorous."
    if analysis["label"] == "Ambiguous / Uncertain":
        return "That one is hard to call from the text alone. A little more context could change the reading."
    return "The model reads that as mostly straightforward rather than humorous."


def generate_reply(text: str, analysis: dict[str, Any]) -> tuple[str, str]:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        return _fallback(analysis), "Local fallback"
    try:
        from google import genai
        model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        client = genai.Client(api_key=key, http_options={"timeout": 20000})
        prompt = f"""You are the response component of a humor-detection demo.
The classifier has already returned:
- score: {analysis['score']:.3f}
- label: {analysis['label']}
- confidence: {analysis['confidence']}

User text: {text}

Reply naturally in one or two sentences. Do not claim the score measures objective funniness. If the classifier is uncertain, do not force a joke interpretation."""
        response = client.models.generate_content(model=model, contents=prompt)
        reply = (response.text or "").strip()
        if reply:
            return reply, f"Gemini ({model})"
    except Exception:
        pass
    return _fallback(analysis), "Local fallback"
