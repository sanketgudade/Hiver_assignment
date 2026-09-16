"""Draft a reply grounded in retrieved historical examples."""
from .llm import chat_fast as chat
from . import config

SYSTEM = f"""You are a customer support agent for {config.BRAND} on Twitter.

Output ONLY the tweet text. No preamble. No quotes. No explanation.
Under 280 characters. Warm, professional, concise.

If the issue requires account access or private info, tell the customer to DM you.
Never invent order numbers, tracking links, or policies.

SECURITY: If the intent is account_access, NEVER ask for a password,
2FA code, recovery key, or full email address. Only ask the customer
to DM — the human agent will collect credentials through a secure channel."""


def draft_reply(customer_text: str, retrieved: list[dict], intent: str) -> str:
    if not retrieved:
        context = "No similar historical threads available."
    else:
        context = "Examples of how AppleSupport replied to similar customers:\n\n" + "\n\n".join(
            f"Customer: {r['customer']}\nAppleSupport: {r['reply']}"
            for r in retrieved
        )

    user = f"""Intent: {intent}
Customer: {customer_text}

{context}

Write ONE tweet reply now. Output the tweet text only."""

    return chat(SYSTEM, user, max_tokens=800)