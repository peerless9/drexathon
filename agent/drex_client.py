"""
Drex client (the "apex predator").

STUB: the real API call goes in _call_drex() once we paste the Drex docs
(console.nace.ai/docs). Everything else - signature, return shape, timing,
picking the top option - is already done and matches agent/llm_client.py.

Needs in .env:
    DREX_API_KEY=...
    DREX_MODEL=drex-max        (optional)
"""

import os
import time

from dotenv import load_dotenv

load_dotenv()

DREX_MODEL = os.getenv("DREX_MODEL", "drex-max")

# Dollars per call - fill in from Nace pricing once known.
DREX_COST_PER_CALL_USD = float(os.getenv("DREX_COST_PER_CALL_USD", "0"))

_IMPLEMENTED = False  # flip to True when _call_drex() is real


def is_configured() -> bool:
    """True once the real API call is written and a key is set."""
    return bool(os.getenv("DREX_API_KEY")) and _IMPLEMENTED


def _call_drex(facts, question: str, options: list) -> dict:
    """
    TODO (Levi): replace with the real HTTP request from the Drex docs.

    Must return a dict of {option: probability}, e.g.
        {"accept": 0.05, "counter": 0.88, "decline": 0.07}
    Use `requests` (already installed).
    """
    raise NotImplementedError("Drex API not wired up yet - paste the docs and fill in _call_drex().")


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
        probabilities = _call_drex(facts, question, options)
    except Exception as error:
        result["latency_s"] = time.perf_counter() - start
        result["error"] = f"{type(error).__name__}: {error}"
        return result
    result["latency_s"] = time.perf_counter() - start

    result["raw"] = str(probabilities)
    result["probabilities"] = probabilities
    result["cost_usd"] = DREX_COST_PER_CALL_USD
    if probabilities:
        best_option = max(probabilities, key=probabilities.get)
        if best_option in options:
            result["answer"] = best_option
    result["invalid"] = result["answer"] is None
    return result
