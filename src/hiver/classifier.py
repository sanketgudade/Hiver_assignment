"""Intent classifier: LLM-based + TF-IDF baseline."""
import json
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .intents import INTENTS, INTENT_NAMES, format_taxonomy_for_prompt
from .llm import chat

SYSTEM = f"""You are an intent classifier for Apple's customer support on Twitter.
Classify the customer's message into exactly one of these intents:

{format_taxonomy_for_prompt()}

Return ONLY a JSON object with keys:
  "intent": one of {INTENT_NAMES}
  "confidence": float 0-1
  "reason": one-sentence justification (max 15 words)

Do not add commentary outside the JSON."""


def _parse_json(s: str) -> dict:
    """Parse JSON from LLM output, tolerating truncation."""
    m = re.search(r"\{.*", s, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON in response: {s!r}")
    candidate = m.group(0).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # try to repair truncated JSON: strip trailing comma, close string, close object
        fixed = candidate.rstrip().rstrip(",")
        # close an unterminated string if needed
        if fixed.count('"') % 2 == 1:
            fixed += '"'
        if not fixed.endswith("}"):
            fixed += "}"
        try:
            return json.loads(fixed)
        except Exception as e:
            raise ValueError(f"Unparseable JSON: {candidate!r} ({e})")


def classify_llm(text: str) -> dict:
    raw = chat(SYSTEM, text, max_tokens=300)
    out = _parse_json(raw)
    if out.get("intent") not in INTENT_NAMES:
        out["intent"] = "general_inquiry"
        out["confidence"] = 0.0
    # ensure confidence is float
    try:
        out["confidence"] = float(out.get("confidence", 0.0))
    except (TypeError, ValueError):
        out["confidence"] = 0.0
    out.setdefault("reason", "")
    return out


def train_tfidf(df: pd.DataFrame) -> Pipeline:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(df["text"], df["intent"])
    return pipe


def classify_tfidf(pipe: Pipeline, text: str) -> dict:
    proba = pipe.predict_proba([text])[0]
    idx = proba.argmax()
    return {
        "intent": pipe.classes_[idx],
        "confidence": float(proba[idx]),
        "reason": "tfidf baseline",
    }


def classify_keyword(text: str) -> dict:
    t = text.lower()
    for name, meta in INTENTS.items():
        for kw in meta["keywords"]:
            if kw in t:
                return {"intent": name, "confidence": 0.5, "reason": f"matched '{kw}'"}
    return {"intent": "general_inquiry", "confidence": 0.3, "reason": "no keyword match"}