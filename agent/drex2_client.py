"""
Drex 2.0 (open weights) hosted on DeepInfra.

A thin wrapper so the benchmark can treat it as its own model, exactly like
llm_client and drex_client: same decide() signature, same return keys.
All the real work is in drex_client.py (backend="deepinfra").

Needs in .env:  DEEPINFRA_API_KEY=...
"""

from agent import drex_client


def is_configured() -> bool:
    return drex_client.is_configured("deepinfra")


def decide(facts, question: str, options: list, option_descriptions: dict = None) -> dict:
    return drex_client.decide(facts, question, options, option_descriptions, backend="deepinfra")
