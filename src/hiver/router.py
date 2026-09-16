"""Decide auto-handle vs escalate, with a stated reason."""
import re
from . import config

# Reason text per non-auto-handle intent — explains WHY public handling is unsafe
ESCALATION_REASONS = {
    "account_access": "Account/identity issue — public resolution would require credentials. Route to DM/human.",
    "refund_billing": "Money/refund issue — needs private billing info and a human commitment.",
    "complaint_escalation": "Escalation signal (anger, repeat contact, legal) — human intervention required.",
    "product_issue": "Device issue — may need diagnostics that exceed public reply scope. Route to DM.",
    "unclassifiable": "Intent could not be determined confidently — route to human for triage.",
}

# Target team / entity handling the escalation
ESCALATE_TARGETS = {
    "account_access": "human_dm",
    "refund_billing": "human_dm",
    "product_issue": "human_dm",
    "complaint_escalation": "human_agent",
    "unclassifiable": "human_triage",
}


def route(intent: str, confidence: float, text: str) -> dict:
    t = text.lower()

    # Rule 1: keyword escalation (word-boundary so 'sue' doesn't match 'issue')
    for kw in config.ESCALATION_KEYWORDS:
        if re.search(r"\b" + re.escape(kw.lower()) + r"\b", t):
            return {
                "decision": "escalate",
                "escalate_to": "human_agent",
                "reason": f"Contains sensitive keyword '{kw}' — human intervention required.",
            }

    # Rule 2: low confidence
    if confidence < config.CONFIDENCE_THRESHOLD:
        return {
            "decision": "escalate",
            "escalate_to": "human_triage",
            "reason": f"Classifier confidence {confidence:.2f} below {config.CONFIDENCE_THRESHOLD} — uncertain, route to human.",
        }

    # Rule 3: intent allowlist with a stated *why*
    if intent not in config.AUTO_HANDLE_INTENTS:
        reason = ESCALATION_REASONS.get(
            intent,
            f"Intent '{intent}' requires private context or human commitment — not safe to auto-handle publicly.",
        )
        target = ESCALATE_TARGETS.get(intent, "human_dm")
        return {
            "decision": "escalate",
            "escalate_to": target,
            "reason": reason,
        }

    return {
        "decision": "auto_handle",
        "escalate_to": "ai",
        "reason": f"Confident ({confidence:.2f}) '{intent}' resolvable publicly without PII or commitments.",
    }