"""
Drex client (the "apex predator").

Calls POST /v1/systemone with one "choice" question (format from
console.nace.ai/docs/drex/guides/questions). Signature, return shape and
timing match agent/llm_client.py, so the two are interchangeable.

Needs in .env:
    DREX_API_KEY=...
    DREX_MODEL=drex-max-v1.1   (optional)
"""

import os
import time

from dotenv import load_dotenv

load_dotenv()

import requests

DREX_URL = "https://console.nace.ai/v1/systemone"
DREX_MODEL = os.getenv("DREX_MODEL", "drex-max-v1.1")

# From console.nace.ai/docs/drex/reference/pricing (Nov 2026):
# $0.03 per 1M INPUT tokens. Output tokens are free. No per-request fee.
DREX_PRICE_PER_INPUT_TOKEN_USD = 0.03 / 1_000_000

# One shared connection so we don't pay the TLS handshake on every call.
_session = requests.Session()


def is_configured() -> bool:
    """True when a Drex key is set in .env."""
    return bool(os.getenv("DREX_API_KEY"))


def _call_drex(facts, question: str, options: list) -> dict:
    """
    Send one "choice" question to Drex. Request shape is from
    console.nace.ai/docs/drex/guides/questions:
        state     = the case facts (same text the LLM sees)
        questions = {"decision": {"type": "choice", "instructions": ..., "criteria": {option: null}}}
    Returns the parsed JSON response.
    """
    # Lazy import so both clients format facts in exactly the same way.
    from agent.llm_client import format_facts

    body = {
        "model": DREX_MODEL,
        "state": format_facts(facts),
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": question,
                # Option labels only, no descriptions - the LLM also gets labels only.
                "criteria": {str(option): None for option in options},
            }
        },
    }
    headers = {
        "Authorization": f"Bearer {os.getenv('DREX_API_KEY')}",
        "X-Nace-Client": "drexathon/1",
        "Content-Type": "application/json",
    }
    response = _session.post(DREX_URL, json=body, headers=headers, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


def decide(facts, question: str, options: list) -> dict:
    """
    Same signature and return keys as llm_client.decide().
    Drex returns a probability per option; the answer is the most likely one.
    """
    result = {
        "model": f"drex:{DREX_MODEL}",
        "answer": None,
        "probabilities": None,
        "latency_s": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "invalid": True,
        "raw": "",
        "error": None,
    }

    start = time.perf_counter()
    try:
        data = _call_drex(facts, question, options)
    except Exception as error:
        result["latency_s"] = time.perf_counter() - start
        result["error"] = f"{type(error).__name__}: {error}"
        return result
    result["latency_s"] = time.perf_counter() - start

    answer = data.get("answers", {}).get("decision", {})
    usage = data.get("usage", {})
    result["raw"] = str(data)
    result["model"] = f"drex:{data.get('model', DREX_MODEL)}"
    result["probabilities"] = answer.get("probabilities")
    result["drex_eval_ms"] = data.get("evaluation_time_ms")  # Drex's own compute time
    result["input_tokens"] = usage.get("input_tokens", 0)
    result["output_tokens"] = usage.get("output_tokens", 0)
    result["cost_usd"] = result["input_tokens"] * DREX_PRICE_PER_INPUT_TOKEN_USD

    # Map Drex's label back to the exact option object we were given.
    for option in options:
        if str(option) == answer.get("choice"):
            result["answer"] = option
    result["invalid"] = result["answer"] is None
    return result
