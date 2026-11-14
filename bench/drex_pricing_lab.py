"""
Pricing lab: can a better QUESTION DESIGN make Drex better at pricing?

We can't retrain Drex (it's Nace's hosted model), but we can choose how we ask.
Every design below uses the exact same round-3 cases from run_bench.py (same facts,
same 5 price points, same "correct" answer). Only the question shape changes:

  choice      - (round 3 as-is) pick one of 5 labels. Read from bench/results/v3/.
  score       - the 5 price levels as an ORDERED scale (Drex's "score" type, which
                its docs recommend for ordered labels). Answer = level nearest the
                expected score.
  thresholds  - 4 yes/no questions in ONE call: "Will it sell for more than $X?",
                X = midpoint between neighbouring price options. Answer = level
                given by the expected number of yeses (sum of probabilities).

Honesty rule: designs are compared on the DEV half (even rows) only; the chosen
design is then reported on the TEST half (odd rows), which never influenced it.

    python bench/drex_pricing_lab.py            # Drex Max (Nace)
    python bench/drex_pricing_lab.py deepinfra  # Drex 2.0 on DeepInfra

Writes bench/results/pricing_lab_<backend>.csv and .json.
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "bench"))

import run_bench  # noqa: E402
from agent import drex_client  # noqa: E402
from agent.llm_client import format_facts  # noqa: E402


def ask_drex(state: str, questions: dict, backend: str) -> tuple:
    """One raw Drex call with several questions. Returns (answers, seconds, input_tokens)."""
    settings = drex_client.BACKENDS[backend]
    body = {"model": settings["model"], "state": state, "questions": questions}
    headers = {"Authorization": f"Bearer {os.getenv(settings['key_env'])}",
               "Content-Type": "application/json", **settings["extra_headers"]}
    for attempt in range(5):
        start = time.perf_counter()
        response = drex_client._session.post(settings["url"], json=body, headers=headers, timeout=60)
        seconds = time.perf_counter() - start
        if response.status_code == 200:
            data = response.json()
            return data["answers"], seconds, data.get("usage", {}).get("input_tokens", 0)
        if response.status_code not in (429, 500, 502, 503, 529):
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
        time.sleep(1 + 2 * attempt)
    raise RuntimeError("Drex kept answering 'busy'")


def nearest_level(value: float, count: int) -> int:
    return max(0, min(count - 1, int(round(value))))


def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "nace"
    dataframe, _ = run_bench.load_csv("pricing")
    cases = run_bench.build_pricing_cases(dataframe, "v3")

    # Round-3 answers we already have (no new calls needed for these).
    v3_dir = run_bench.RESULTS_DIR / "v3"
    choice_file = v3_dir / ("pricing_drex.csv" if backend == "nace" else "pricing_drex2.csv")
    choice_answers = dict(zip(*pd.read_csv(choice_file)[["id", "answer"]].T.values))
    opus_answers = dict(zip(*pd.read_csv(v3_dir / "pricing_llm.csv")[["id", "answer"]].T.values))

    rows = []
    for number, case in enumerate(cases):
        labels = case["options"]                      # ordered: much lower ... much higher
        prices = [case["option_prices"][label] for label in labels]
        correct_level = labels.index(case["correct"])
        state = format_facts(case["facts"])

        # --- score: ordered scale, one call
        score_q = {"price_level": {
            "type": "score",
            "instructions": "What will this item most likely sell for?",
            "criteria": [f"{label}: sells for about ${price}" for label, price in zip(labels, prices)],
        }}
        answers, score_seconds, score_tokens = ask_drex(state, score_q, backend)
        score_value = answers["price_level"]["score"]
        score_level = nearest_level(score_value, len(labels))

        # --- thresholds: 4 yes/no questions, one call
        cut_points = [(low + high) / 2 for low, high in zip(prices, prices[1:])]
        threshold_q = {f"above_{i}": {"type": "noul",
                                      "instructions": f"Will this item sell for more than ${cut:.0f}?"}
                       for i, cut in enumerate(cut_points)}
        answers, threshold_seconds, threshold_tokens = ask_drex(state, threshold_q, backend)
        yes_probabilities = [answers[f"above_{i}"]["noul"] for i in range(len(cut_points))]
        threshold_level = nearest_level(sum(yes_probabilities), len(labels))

        choice_label = choice_answers.get(case["id"])
        opus_label = opus_answers.get(case["id"])
        rows.append({
            "id": case["id"], "half": "dev" if number % 2 == 0 else "test",
            "actual_price": case["actual_price"], "correct_level": correct_level,
            "choice_level": labels.index(choice_label) if choice_label in labels else None,
            "score_level": score_level, "score_value": round(score_value, 3),
            "threshold_level": threshold_level,
            "threshold_probs": json.dumps([round(p, 3) for p in yes_probabilities]),
            "opus_level": labels.index(opus_label) if opus_label in labels else None,
            "prices": json.dumps(prices),
            "score_seconds": score_seconds, "threshold_seconds": threshold_seconds,
            "score_tokens": score_tokens, "threshold_tokens": threshold_tokens,
        })
        print(f"  {number + 1}/{len(cases)} {case['id']}: correct={correct_level} "
              f"choice={rows[-1]['choice_level']} score={score_level} thresholds={threshold_level}")

    table = pd.DataFrame(rows)
    price_per_token = drex_client.BACKENDS[backend]["price_in_per_token"]
    report = {"backend": backend, "n": len(table), "designs": {}}
    for design in ("choice", "score", "threshold", "opus"):
        report["designs"][design] = {}
        for half in ("dev", "test", "all"):
            part = table if half == "all" else table[table["half"] == half]
            part = part[part[f"{design}_level"].notna()]
            levels = part[f"{design}_level"].astype(int)
            exact = (levels == part["correct_level"]).mean()
            within_one = ((levels - part["correct_level"]).abs() <= 1).mean()
            chosen_prices = [json.loads(p)[lvl] for p, lvl in zip(part["prices"], levels)]
            pct_errors = [abs(c - a) / a for c, a in zip(chosen_prices, part["actual_price"])]
            report["designs"][design][half] = {
                "n": len(part), "exact": round(exact, 4), "within_one_option": round(within_one, 4),
                "median_abs_pct_error": round(statistics.median(pct_errors), 4),
            }
        if design in ("score", "threshold"):
            report["designs"][design]["median_latency_s"] = round(table[f"{design}_seconds"].iloc[1:].median(), 4)
            report["designs"][design]["cost_per_1000_usd"] = round(
                1000 * table[f"{design}_tokens"].mean() * price_per_token, 4)

    table.to_csv(run_bench.RESULTS_DIR / f"pricing_lab_{backend}.csv", index=False)
    (run_bench.RESULTS_DIR / f"pricing_lab_{backend}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
