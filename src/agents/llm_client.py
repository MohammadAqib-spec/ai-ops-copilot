"""
Thin LLM client that supports free-tier providers via OpenAI-compatible
APIs, and falls back to a deterministic template if no API key is set
-- so the whole pipeline still runs end-to-end with zero cost and zero
network dependency while you're developing.

Supported providers (set LLM_PROVIDER in .env):
  - "groq"   : https://console.groq.com  (generous free tier, very fast)
  - "gemini" : https://ai.google.dev     (Google's free tier)
  - "none"   : offline template fallback (default if no key found)

Both providers expose an OpenAI-compatible /chat/completions endpoint,
so swapping providers is a one-line base_url + model change.
"""

import os

import requests

PROVIDER = os.getenv("LLM_PROVIDER", "none").lower()
API_KEY = os.getenv("LLM_API_KEY", "")

PROVIDER_CONFIG = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.1-8b-instant",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model": "gemini-1.5-flash",
    },
}


def call_llm(prompt: str, system: str = "", max_tokens: int = 300) -> str:
    """Returns the LLM's text response, or a template fallback offline."""
    if PROVIDER not in PROVIDER_CONFIG or not API_KEY:
        return _offline_fallback(prompt)

    config = PROVIDER_CONFIG[PROVIDER]
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    try:
        resp = requests.post(config["base_url"], headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:  # network / quota / schema issues -> degrade gracefully
        return _offline_fallback(prompt, error=str(e))


def _offline_fallback(prompt: str, error: str = "") -> str:
    note = f" (LLM call failed: {error})" if error else " (offline mode: no LLM_API_KEY set)"
    return (
        "Template explanation" + note + ". "
        "This transaction was flagged based on statistical deviation from "
        "typical account behavior (amount, timing, and frequency features). "
        "Set LLM_PROVIDER and LLM_API_KEY in .env for a natural-language "
        "explanation from a real model."
    )
