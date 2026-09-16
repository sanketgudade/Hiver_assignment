"""LLM-as-judge for reply quality. 5 criteria, 0-5 each."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.hiver.llm import chat_fast as chat

RUBRIC = """Score the reply on FIVE criteria, each 0-5:
1. helpfulness: does it move the customer toward resolution?
2. tone: warm, professional, Apple-appropriate?
3. accuracy: no invented facts, order numbers, policies, or promises?
4. groundedness: consistent with how AppleSupport actually replies?
5. brevity: under 280 chars, no filler?

Return ONLY JSON:
{"helpfulness": int, "tone": int, "accuracy": int, "groundedness": int, "brevity": int, "comment": "one sentence"}"""


def judge_reply(customer: str, reply: str, intent: str) -> dict:
    user = f"Intent: {intent}\nCustomer: {customer}\nReply: {reply}\n\nScore it."
    raw = chat(RUBRIC, user, max_tokens=300)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    empty = {k: 0 for k in ["helpfulness", "tone", "accuracy", "groundedness", "brevity", "total"]}
    empty["comment"] = "parse_error"
    if not m:
        return empty
    try:
        out = json.loads(m.group(0))
    except Exception:
        return empty
    for k in ["helpfulness", "tone", "accuracy", "groundedness", "brevity"]:
        out[k] = int(out.get(k, 0))
    out["total"] = sum(out[k] for k in ["helpfulness", "tone", "accuracy", "groundedness", "brevity"])
    return out