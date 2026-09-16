"""Unified LLM client: Groq primary, OpenAI fallback. Both use OpenAI SDK."""
import os
import time
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from . import config

load_dotenv()

_groq_client = None
_openai_client = None


def _client():
    global _groq_client, _openai_client
    if os.getenv("GROQ_API_KEY"):
        if _groq_client is None:
            _groq_client = OpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
            )
        return _groq_client, config.GROQ_MODEL
    if os.getenv("OPENAI_API_KEY"):
        if _openai_client is None:
            _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return _openai_client, config.OPENAI_MODEL
    raise RuntimeError("No API key found. Set GROQ_API_KEY or OPENAI_API_KEY in .env")


def chat(system: str, user: str, temperature: float = None, max_tokens: int = None) -> str:
    client, model = _client()
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature if temperature is not None else config.TEMPERATURE,
            max_tokens=max_tokens or config.MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()
    except RateLimitError as e:
        if "openai/gpt-oss-120b" in model:
            time.sleep(1)
            resp = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                temperature=temperature if temperature is not None else config.TEMPERATURE,
                max_tokens=max_tokens or config.MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content.strip()
        raise e


def chat_fast(system: str, user: str, max_tokens: int = 300) -> str:
    """Fast non-reasoning LLM call with rate limit resilience."""
    if os.getenv("GROQ_API_KEY"):
        client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        try:
            resp = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                temperature=config.TEMPERATURE,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except RateLimitError:
            time.sleep(1)
            resp = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                temperature=config.TEMPERATURE,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
    return chat(system, user, max_tokens=max_tokens)