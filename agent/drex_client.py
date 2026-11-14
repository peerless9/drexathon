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

# Two places we can run Drex. Same request body and same response shape at both.
BACKENDS = {
    # Nace's own API: Drex Max v1.1. $0.03 per 1M input tokens, output free.
    "nace": {
        "url": DREX_URL,
        "model": DREX_MODEL,
        "key_env": "DREX_API_KEY",
        "price_in_per_token": 0.03 / 1_000_000,
        "price_out_per_token": 0.0,
        "extra_headers": {"X-Nace-Client": "drexathon/1"},
    },
    # DeepInfra: the open-weights Drex 2.0 (deepinfra.com/nace-ai/drex-v2.0/api).
    # Listed at $0.06 per 1M tokens; we charge it on input AND output to be safe.
    "deepinfra": {
        "url": "https://api.deepinfra.com/v1/decisions",
        "model": os.getenv("DEEPINFRA_DREX_MODEL", "nace-ai/drex-v2.0"),
        "key_env": "DEEPINFRA_API_KEY",
        "price_in_per_token": 0.06 / 1_000_000,
        "price_out_per_token": 0.06 / 1_000_000,
        "extra_headers": {},
    },
}


MAX_RETRIES = int(os.getenv("DREX_MAX_RETRIES", "4"))

# One shared connection so we don't pay the TLS handshake on every call.
_session = requests.Session()


def is_configured(backend: str = "nace") -> bool:
    """True when the key for this backend is set in .env."""
    return bool(os.getenv(BACKENDS[backend]["key_env"]))


def _call_drex(facts, question: str, options: list, option_descriptions: dict = None,
               backend: str = "nace") -> dict:
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
        "model": BACKENDS[backend]["model"],
        "state": format_facts(facts),
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": question,
                # Label -> description (or None). The LLM sees the same descriptions.
                "criteria": {str(option): (option_descriptions or {}).get(option) for option in options},
            }
        },
    }
    headers = {
        "Authorization": f"Bearer {os.getenv(BACKENDS[backend]['key_env'])}",
        **BACKENDS[backend]["extra_headers"],
        "Content-Type": "application/json",
    }
    response = _session.post(BACKENDS[backend]["url"], json=body, headers=headers, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


def decide(facts, question: str, options: list, option_descriptions: dict = None,
           backend: str = "nace") -> dict:
    """
    Same signature and return keys as llm_client.decide().
    Drex returns a probability per option; the answer is the most likely one.
    """
    result = {
        "model": f"drex:{BACKENDS[backend]['model']}",
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

    # Shared servers sometimes answer 429 "Model busy". Wait and retry, like llm_client.
    # Only the successful attempt's time counts as latency.
    for attempt in range(MAX_RETRIES + 1):
        start = time.perf_counter()
        try:
            data = _call_drex(facts, question, options, option_descriptions, backend)
            result["error"] = None
            break
        except Exception as error:
            result["latency_s"] = time.perf_counter() - start
            result["error"] = f"{type(error).__name__}: {error}"
            retryable = any(code in result["error"] for code in ("HTTP 429", "HTTP 5", "Timeout"))
            if not retryable or attempt == MAX_RETRIES:
                return result
            time.sleep(1 + 2 * attempt)
    result["retries"] = attempt
    result["latency_s"] = time.perf_counter() - start

    answer = data.get("answers", {}).get("decision", {})
    usage = data.get("usage", {})
    result["raw"] = str(data)
    result["model"] = f"drex:{data.get('model', BACKENDS[backend]['model'])}"
    result["probabilities"] = answer.get("probabilities")
    result["drex_eval_ms"] = data.get("evaluation_time_ms")  # Drex's own compute time
    result["input_tokens"] = usage.get("input_tokens", 0)
    result["output_tokens"] = usage.get("output_tokens", 0)
    price = BACKENDS[backend]
    result["cost_usd"] = (result["input_tokens"] * price["price_in_per_token"]
                          + result["output_tokens"] * price["price_out_per_token"])

    # Map Drex's label back to the exact option object we were given.
    for option in options:
        if str(option) == answer.get("choice"):
            result["answer"] = option
    result["invalid"] = result["answer"] is None
    return result
