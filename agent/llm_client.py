"""
Frontier LLM client (the "giant").

ONE function matters: decide(facts, question, options).
It has the exact same signature and return shape as agent/drex_client.py,
so the benchmark and the app can swap the two models freely.

Provider is picked from .env:
    LLM_PROVIDER=anthropic   (default)  -> needs ANTHROPIC_API_KEY
    LLM_PROVIDER=openai                 -> needs OPENAI_API_KEY
    LLM_MODEL=...            optional override of the model name
    LLM_EFFORT=low           (Anthropic only) thinking effort: low/medium/high
    LLM_PRICE_IN_PER_M / LLM_PRICE_OUT_PER_M   optional price override ($ per 1M tokens)
"""

import json
import os
import re
import time

from dotenv import load_dotenv

load_dotenv()

# Default model per provider. Change in .env with LLM_MODEL=...
DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-4o",
}

# Price in US dollars per 1 MILLION tokens: (input, output).
# Anthropic prices are from Anthropic's model table (Nov 2026).
# OpenAI prices are ESTIMATES - check the pricing page and override in .env if needed.
PRICES_PER_MILLION = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "gpt-4o": (2.50, 10.00),
}

# Same system prompt for every decision. Drex gets the same facts/question/options.
SYSTEM_PROMPT = (
    "You are a decision engine for a used-clothing resale business. "
    "You will be given case facts, one question, and a fixed list of options. "
    "Pick exactly one option. Reply with ONLY a JSON object, no other text: "
    '{"answer": "<one option, copied exactly>"}'
)


def get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "anthropic").strip().lower()


def get_model() -> str:
    return os.getenv("LLM_MODEL") or DEFAULT_MODELS[get_provider()]


def format_facts(facts) -> str:
    """Turn a dict of facts into readable lines. Strings pass through unchanged."""
    if isinstance(facts, dict):
        lines = []
        for key, value in facts.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)
    return str(facts)


def build_user_prompt(facts, question: str, options: list) -> str:
    """The exact text both models see (Drex gets the same pieces as structured fields)."""
    option_lines = "\n".join(f"- {option}" for option in options)
    return (
        f"CASE FACTS:\n{format_facts(facts)}\n\n"
        f"QUESTION: {question}\n\n"
        f"OPTIONS (pick exactly one):\n{option_lines}"
    )


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of one call."""
    price_in, price_out = PRICES_PER_MILLION.get(model, (0.0, 0.0))
    if os.getenv("LLM_PRICE_IN_PER_M"):
        price_in = float(os.getenv("LLM_PRICE_IN_PER_M"))
    if os.getenv("LLM_PRICE_OUT_PER_M"):
        price_out = float(os.getenv("LLM_PRICE_OUT_PER_M"))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def parse_answer(text: str, options: list):
    """
    Pull the chosen option out of the model's reply.
    Returns the option string, or None if the reply is not one of the options.
    """
    if not text:
        return None
    # Find the first {...} block in case the model added extra words.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        answer = str(json.loads(match.group(0)).get("answer", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        return None
    # Must be exactly one of the options (ignore upper/lower case).
    for option in options:
        if answer.lower() == str(option).strip().lower():
            return option
    return None


def _call_anthropic(model: str, user_prompt: str):
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4000,  # room for thinking + the short JSON answer
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        # Opus 5.5 always thinks; effort controls how much. "low" = fast decisions.
        output_config={"effort": os.getenv("LLM_EFFORT", "low")},
    )
    if response.stop_reason == "refusal":
        return "", response.usage.input_tokens, response.usage.output_tokens
    text = "".join(block.text for block in response.content if block.type == "text")
    return text, response.usage.input_tokens, response.usage.output_tokens


def _call_openai(model: str, user_prompt: str):
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},  # forces valid JSON
    )
    text = response.choices[0].message.content or ""
    return text, response.usage.prompt_tokens, response.usage.completion_tokens


def decide(facts, question: str, options: list) -> dict:
    """
    Ask the frontier LLM to pick one option.

    Returns a dict (same keys as drex_client.decide):
        answer         - one of options, or None if invalid
        probabilities  - always None here (LLMs don't give calibrated probabilities)
        latency_s      - wall-clock seconds for the API call
        input_tokens, output_tokens
        cost_usd       - estimated dollars for this call
        invalid        - True if the answer was missing or not an option
        raw            - the raw reply text
        error          - error message if the call failed, else None
    """
    model = get_model()
    user_prompt = build_user_prompt(facts, question, options)
    result = {
        "model": f"{get_provider()}:{model}",
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
        if get_provider() == "openai":
            text, tokens_in, tokens_out = _call_openai(model, user_prompt)
        else:
            text, tokens_in, tokens_out = _call_anthropic(model, user_prompt)
    except Exception as error:  # keep the benchmark running even if one call fails
        result["latency_s"] = time.perf_counter() - start
        result["error"] = f"{type(error).__name__}: {error}"
        return result

    result["latency_s"] = time.perf_counter() - start
    result["raw"] = text
    result["input_tokens"] = tokens_in
    result["output_tokens"] = tokens_out
    result["cost_usd"] = estimate_cost(model, tokens_in, tokens_out)
    result["answer"] = parse_answer(text, options)
    result["invalid"] = result["answer"] is None
    return result


if __name__ == "__main__":
    # Quick manual test:  python agent/llm_client.py
    print(decide(
        {"item": "Levi's 501 jeans", "list_price": 60, "offer": 30, "buyer_message": "can you do 30?"},
        "What should the seller do with this offer?",
        ["accept", "counter", "decline"],
    ))
