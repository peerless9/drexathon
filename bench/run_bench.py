"""
Benchmark: Drex vs a frontier LLM on two reselling decisions.

    python bench/run_bench.py                      # both benchmarks, both models
    python bench/run_bench.py --models llm         # just the LLM (e.g. before Drex is wired up)
    python bench/run_bench.py --limit 3            # first 3 rows of each CSV (cheap smoke test)
    python bench/run_bench.py --bench negotiation  # one benchmark only

Data:   bench/data/pricing.csv, bench/data/negotiation.csv
        (falls back to *_SAMPLE.csv if the real file is missing; skips if neither exists)
Output: bench/results/summary.json  +  one per-row CSV per (benchmark, model)
"""

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Let us import agent/ when running "python bench/run_bench.py" from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent import drex_client, llm_client  # noqa: E402

DATA_DIR = REPO_ROOT / "bench" / "data"
RESULTS_DIR = REPO_ROOT / "bench" / "results"

PRICING_QUESTION = "Which listing price is closest to what this item will actually sell for?"
NEGOTIATION_QUESTION = "What should the seller do with this buyer's offer?"
NEGOTIATION_OPTIONS = ["accept", "counter", "decline"]
SELLER_POLICY = (
    "Accept if the offer is at least 90% of the list price. "
    "Counter if the offer is at or above the floor price but below 90% of list. "
    "Decline if the offer is below the floor price, OR if the buyer shows scam signs "
    "(asks to pay off-platform e.g. Zelle/Venmo/CashApp, asks you to text/email them, "
    "or wants to send their own shipping label)."
)

MODELS = {
    "llm": llm_client,
    "drex": drex_client,
}


# ---------------------------------------------------------------- data loading

def load_csv(name: str):
    """Return (dataframe, path) for bench/data/<name>.csv, or the SAMPLE file, or (None, None)."""
    real_path = DATA_DIR / f"{name}.csv"
    sample_path = DATA_DIR / f"{name}_SAMPLE.csv"
    for path in (real_path, sample_path):
        if path.exists():
            # comment="#" lets a CSV start with a "# SAMPLE DATA" note line.
            dataframe = pd.read_csv(path, comment="#")
            if len(dataframe) > 0:
                return dataframe, path
    return None, None


# ---------------------------------------------------------------- pricing cases

def make_price_options(comp_prices: list) -> list:
    """5 options from the comps: 25th/40th/50th/60th/75th percentiles, rounded to whole dollars."""
    series = pd.Series(comp_prices, dtype=float)
    options = []
    for percentile in (0.25, 0.40, 0.50, 0.60, 0.75):
        price = int(round(series.quantile(percentile)))
        # Keep options distinct: if rounding made a duplicate, bump it up a dollar.
        while price in options:
            price += 1
        options.append(price)
    return options


def build_pricing_cases(dataframe: pd.DataFrame) -> list:
    """One case per item. Comps = the OTHER items in the same category (leave-one-out)."""
    cases = []
    for _, row in dataframe.iterrows():
        comps = dataframe[(dataframe["category"] == row["category"]) & (dataframe["id"] != row["id"])]
        if len(comps) < 2:
            print(f"  skip pricing id={row['id']}: fewer than 2 comps in category '{row['category']}'")
            continue

        option_prices = make_price_options(comps["sold_price"].tolist())
        options = [f"${price}" for price in option_prices]
        # Correct = the option closest to the real sold price.
        distances = [abs(price - float(row["sold_price"])) for price in option_prices]
        correct = options[distances.index(min(distances))]

        comp_lines = [
            f"{comp['brand']} {comp['title']} (size {comp['size']}, {comp['condition']}): sold ${comp['sold_price']}"
            for _, comp in comps.iterrows()
        ]
        facts = {
            "category": row["category"],
            "brand": row["brand"],
            "title": row["title"],
            "size": row["size"],
            "condition": row["condition"],
            "recently sold comparable items": "; ".join(comp_lines),
        }
        cases.append({"id": row["id"], "facts": facts, "question": PRICING_QUESTION,
                      "options": options, "correct": correct})
    return cases


# ---------------------------------------------------------------- negotiation cases

def build_negotiation_cases(dataframe: pd.DataFrame) -> list:
    cases = []
    for _, row in dataframe.iterrows():
        facts = {
            "seller policy": SELLER_POLICY,
            "item": row["item"],
            "list price": f"${row['list_price']}",
            "floor price (lowest the seller will take)": f"${row['floor_price']}",
            "days listed": row["days_listed"],
            "buyer message": row["buyer_message"],
            "offer": f"${row['offer']}",
        }
        cases.append({"id": row["id"], "facts": facts, "question": NEGOTIATION_QUESTION,
                      "options": NEGOTIATION_OPTIONS,
                      "correct": str(row["correct_action"]).strip().lower()})
    return cases


# ---------------------------------------------------------------- running + metrics

def run_cases(model_name: str, client, cases: list) -> list:
    rows = []
    for number, case in enumerate(cases, start=1):
        result = client.decide(case["facts"], case["question"], case["options"])
        is_correct = (not result["invalid"]) and result["answer"] == case["correct"]
        status = "ERROR" if result["error"] else ("ok " if is_correct else "MISS")
        print(f"  [{model_name}] {number}/{len(cases)} id={case['id']}: {status} "
              f"answer={result['answer']} correct={case['correct']} "
              f"{(result['latency_s'] or 0):.2f}s")
        if result["error"] and number == 1:
            print(f"    -> {result['error']}")
        rows.append({
            "id": case["id"],
            "correct_answer": case["correct"],
            "answer": result["answer"],
            "is_correct": is_correct,
            "invalid": result["invalid"],
            "latency_s": result["latency_s"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "cost_usd": result["cost_usd"],
            "probabilities": json.dumps(result["probabilities"]) if result["probabilities"] else "",
            "error": result["error"] or "",
            "model": result["model"],
        })
    return rows


def summarize(rows: list) -> dict:
    total = len(rows)
    correct = sum(1 for row in rows if row["is_correct"])
    invalid = sum(1 for row in rows if row["invalid"])
    errors = sum(1 for row in rows if row["error"])
    # Skip the first call: it pays connection/warm-up cost the others don't.
    latencies = [row["latency_s"] for row in rows[1:] if row["latency_s"] is not None and not row["error"]]
    if not latencies:
        latencies = [row["latency_s"] for row in rows if row["latency_s"] is not None and not row["error"]]
    costs = [row["cost_usd"] for row in rows if not row["error"]]
    return {
        "model": rows[0]["model"] if rows else None,
        "n": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else None,
        "invalid": invalid,
        "errors": errors,
        "median_latency_s": round(statistics.median(latencies), 4) if latencies else None,
        "cost_per_1000_usd": round(1000 * statistics.mean(costs), 4) if costs else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Drex vs frontier LLM benchmark")
    parser.add_argument("--models", default="llm,drex", help="comma list: llm,drex")
    parser.add_argument("--bench", default="pricing,negotiation", help="comma list: pricing,negotiation")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N rows of each CSV")
    args = parser.parse_args()

    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    bench_names = [name.strip() for name in args.bench.split(",") if name.strip()]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    builders = {"pricing": build_pricing_cases, "negotiation": build_negotiation_cases}
    summary = {"generated_at": datetime.now().isoformat(timespec="seconds"), "benchmarks": {}}

    for bench_name in bench_names:
        dataframe, path = load_csv(bench_name)
        if dataframe is None:
            print(f"\n== {bench_name}: no data (bench/data/{bench_name}.csv missing) - skipping")
            summary["benchmarks"][bench_name] = {"skipped": "no data file"}
            continue

        cases = builders[bench_name](dataframe)
        if args.limit:
            cases = cases[: args.limit]
        is_sample = "SAMPLE" in path.name
        print(f"\n== {bench_name}: {len(cases)} cases from {path.name}"
              + ("   (SAMPLE DATA - not real results!)" if is_sample else ""))

        bench_summary = {"data_file": path.name, "is_sample_data": is_sample, "models": {}}
        for model_name in model_names:
            if model_name == "drex" and not drex_client.is_configured():
                print("  [drex] skipped: not configured yet (needs DREX_API_KEY + real _call_drex)")
                bench_summary["models"]["drex"] = {"skipped": "drex not configured"}
                continue
            rows = run_cases(model_name, MODELS[model_name], cases)
            pd.DataFrame(rows).to_csv(RESULTS_DIR / f"{bench_name}_{model_name}.csv", index=False)
            bench_summary["models"][model_name] = summarize(rows)
        summary["benchmarks"][bench_name] = bench_summary

    output_path = RESULTS_DIR / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {output_path.relative_to(REPO_ROOT)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
