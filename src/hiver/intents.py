"""6-intent taxonomy for AppleSupport, defined from the data."""

INTENTS = {
    "order_status": {
        "description": "Customer asking about shipping, delivery, tracking, or where their order/repair is.",
        "examples": [
            "Where is my iPhone order?",
            "My repair has been stuck for a week, any update?",
            "Tracking says delivered but I don't have it.",
        ],
        "keywords": ["order", "shipping", "delivery", "tracking", "where is", "when will", "arrive"],
    },
    "refund_billing": {
        "description": "Refunds, double charges, invoice issues, unexpected billing.",
        "examples": [
            "I was charged twice for the same app.",
            "How do I get a refund for Apple Music?",
            "I need an invoice for my purchase.",
        ],
        "keywords": ["refund", "charge", "billing", "invoice", "receipt", "double", "overcharged"],
    },
    "product_issue": {
        "description": "Device, OS, or app not working — bugs, crashes, setup, hardware faults.",
        "examples": [
            "My iPhone won't turn on after the update.",
            "AirPods keep disconnecting.",
            "Photos app crashes on launch.",
        ],
        "keywords": ["not working", "broken", "crash", "error", "update", "won't", "stuck", "frozen"],
    },
    "account_access": {
        "description": "Login, password reset, Apple ID locked, 2FA issues.",
        "examples": [
            "I'm locked out of my Apple ID.",
            "Can't sign in, forgot password.",
            "Two-factor code never arrives.",
        ],
        "keywords": ["login", "sign in", "password", "locked", "apple id", "2fa", "verification"],
    },
    "general_inquiry": {
        "description": "How-to, feature question, availability, informational.",
        "examples": [
            "Does the new iPad support external monitors?",
            "How do I enable dark mode?",
            "When is the new iPhone released?",
        ],
        "keywords": ["how do i", "does it", "can i", "when is", "what is", "support"],
    },
    "complaint_escalation": {
        "description": "Angry, repeated contact, legal threat, trust/safety, or explicit demand for a human.",
        "examples": [
            "This is the third time I'm tweeting you. Useless.",
            "I want to speak to a manager.",
            "I'll take this to small claims court.",
        ],
        "keywords": ["unacceptable", "third time", "manager", "lawyer", "sue", "ridiculous", "terrible"],
    },
}

INTENT_NAMES = list(INTENTS.keys())


def format_taxonomy_for_prompt() -> str:
    lines = []
    for name, meta in INTENTS.items():
        lines.append(f"- {name}: {meta['description']}")
    return "\n".join(lines)