"""
Benchmark: Drex vs a frontier LLM on two reselling decisions.

    python bench/run_bench.py                      # both benchmarks, both models
    python bench/run_bench.py --models llm         # just the LLM (e.g. before Drex is wired up)
    python bench/run_bench.py --limit 3            # first 3 rows of each CSV (cheap smoke test)
    python bench/run_bench.py --bench negotiation  # one benchmark only
    python bench/run_bench.py --variant v2         # round 2 (see VARIANTS below)

Data:   bench/data/pricing.csv, bench/data/negotiation.csv
        (falls back to *_SAMPLE.csv if the real file is missing; skips if neither exists)
Output: bench/results/summary.json  +  one per-row CSV per (benchmark, model)
        (round 2 writes to bench/results/v2/ instead, so v1 stays untouched)

VARIANTS - every change gives BOTH models exactly the same extra input:
  v1  plain facts + option labels.
  v2  + pre-computed numbers ("code does the math"):
          negotiation: offer as % of list, offer minus floor
          pricing:     number of comps, median comp, median of same-condition comps, range
      + option descriptions (negotiation: the policy rule for each option)
      + pricing options from the 10th-90th percentile instead of 25th-75th
      + negotiation fact "90% of list price = $X"
      Also reported (not used as the answer): accuracy if Drex picked the option
      closest to its probability-weighted price.
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

# v2: what each negotiation option means - the seller policy, split per option.
NEGOTIATION_DESCRIPTIONS = {
    "accept": "Offer is at least 90% of the list price, and no scam signs",
    "counter": "Offer is at or above the floor price but below 90% of the list price, and no scam signs",
    "decline": "Offer is below the floor price, OR the buyer shows scam signs "
               "(off-platform payment like Zelle/Venmo/CashApp, asks to text/email, own shipping label)",
}

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

# v1 options sit between the 25th and 75th percentile of the comps, so the real price
# was often outside them and the right answer was just "the lowest" or "the highest"
# option (29 of 40 rows). v2 spreads them from the 10th to the 90th percentile.
PERCENTILES = {
    "v1": (0.25, 0.40, 0.50, 0.60, 0.75),
    "v2": (0.10, 0.30, 0.50, 0.70, 0.90),
}


def make_price_options(comp_prices: list, variant: str = "v1") -> list:
    """5 options from the comps' percentiles (see PERCENTILES), rounded to whole dollars."""
    series = pd.Series(comp_prices, dtype=float)
    options = []
    for percentile in PERCENTILES[variant]:
        price = int(round(series.quantile(percentile)))
        # Keep options distinct: if rounding made a duplicate, bump it up a dollar.
        while price in options:
            price += 1
        options.append(price)
    return options


def build_pricing_cases(dataframe: pd.DataFrame, variant: str = "v1") -> list:
    """One case per item. Comps = the OTHER items in the same category (leave-one-out)."""
    cases = []
    for _, row in dataframe.iterrows():
        comps = dataframe[(dataframe["category"] == row["category"]) & (dataframe["id"] != row["id"])]
        if len(comps) < 2:
            print(f"  skip pricing id={row['id']}: fewer than 2 comps in category '{row['category']}'")
            continue

        option_prices = make_price_options(comps["sold_price"].tolist(), variant)
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
        if variant == "v2":
            same_condition = comps[comps["condition"] == row["condition"]]["sold_price"]
            facts["number of comparable sales"] = len(comps)
            facts["median comparable sale price"] = f"${comps['sold_price'].median():.2f}"
            facts["comparable price range"] = f"${comps['sold_price'].min():.2f} to ${comps['sold_price'].max():.2f}"
            if len(same_condition) >= 2:
                facts[f"median price of comps in the same condition ({row['condition']})"] = (
                    f"${same_condition.median():.2f} (from {len(same_condition)} sales)")
        cases.append({"id": row["id"], "facts": facts, "question": PRICING_QUESTION,
                      "options": options, "correct": correct,
                      "actual_price": float(row["sold_price"])})
    return cases


# ---------------------------------------------------------------- negotiation cases

def build_negotiation_cases(dataframe: pd.DataFrame, variant: str = "v1") -> list:
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
        descriptions = None
        if variant == "v2":
            offer, list_price, floor = float(row["offer"]), float(row["list_price"]), float(row["floor_price"])
            facts["offer as % of list price"] = f"{100 * offer / list_price:.1f}%"
            facts["90% of list price"] = f"${0.9 * list_price:.2f}"
            gap = offer - floor
            facts["offer compared to floor"] = (f"${gap:.0f} above the floor" if gap >= 0
                                                else f"${-gap:.0f} BELOW the floor")
            descriptions = NEGOTIATION_DESCRIPTIONS
        cases.append({"id": row["id"], "facts": facts, "question": NEGOTIATION_QUESTION,
                      "options": NEGOTIATION_OPTIONS, "descriptions": descriptions,
                      "correct": str(row["correct_action"]).strip().lower()})
    return cases


# ---------------------------------------------------------------- running + metrics

def run_cases(model_name: str, client, cases: list, variant: str = "v1") -> list:
    rows = []
    for number, case in enumerate(cases, start=1):
        result = client.decide(case["facts"], case["question"], case["options"],
                               case.get("descriptions"))
        result["top_answer"] = result["answer"]
        if variant == "v2" and "actual_price" in case and result["probabilities"]:
            # Probability-weighted price, then the option closest to it.
            expected = sum(prob * float(str(option).strip("$"))
                           for option, prob in result["probabilities"].items())
            result["expected_option"] = min(case["options"], key=lambda o: abs(float(o.strip("$")) - expected))
            result["expected_price"] = round(expected, 2)
        is_correct = (not result["invalid"]) and result["answer"] == case["correct"]
        status = "ERROR" if result["error"] else ("ok " if is_correct else "MISS")
        print(f"  [{model_name}] {number}/{len(cases)} id={case['id']}: {status} "
              f"answer={result['answer']} correct={case['correct']} "
              f"{(result['latency_s'] or 0):.2f}s")
        if result["error"] and number == 1:
            print(f"    -> {result['error']}")
        # Pricing only: how many dollars off was the chosen price from the real sold price?
        dollar_error = None
        if "actual_price" in case and result["answer"]:
            dollar_error = abs(float(str(result["answer"]).strip("$")) - case["actual_price"])
        rows.append({
            "id": case["id"],
            "dollar_error": dollar_error,
            "correct_answer": case["correct"],
            "answer": result["answer"],
            "top_answer": result["top_answer"],
            "expected_price": result.get("expected_price"),
            "expected_option_correct": (result.get("expected_option") == case["correct"]
                                        if result.get("expected_option") else None),
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
    dollar_errors = [row["dollar_error"] for row in rows if row.get("dollar_error") is not None]
    extra = {}
    expected_hits = [row["expected_option_correct"] for row in rows if row.get("expected_option_correct") is not None]
    if expected_hits:
        extra["accuracy_expected_price_pick"] = round(sum(expected_hits) / len(expected_hits), 4)
    if dollar_errors:
        extra["mean_dollar_error"] = round(statistics.mean(dollar_errors), 2)
    return {
        **extra,
        "model": rows[0]["model"] if rows else None,
        "n": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else None,
        # Same, but ignoring calls that errored (e.g. rate limits) - those aren't wrong answers.
        "accuracy_on_answered": round(correct / (total - errors), 4) if total - errors else None,
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
    parser.add_argument("--variant", default="v1", choices=["v1", "v2"], help="see VARIANTS at the top")
    args = parser.parse_args()

    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    bench_names = [name.strip() for name in args.bench.split(",") if name.strip()]
    results_dir = RESULTS_DIR if args.variant == "v1" else RESULTS_DIR / args.variant
    results_dir.mkdir(parents=True, exist_ok=True)

    builders = {"pricing": build_pricing_cases, "negotiation": build_negotiation_cases}
    output_path = results_dir / "summary.json"

    # Keep results for benchmarks/models we are NOT re-running this time,
    # so "--bench pricing" doesn't wipe out the negotiation numbers.
    summary = {"benchmarks": {}}
    if output_path.exists():
        try:
            summary = json.loads(output_path.read_text())
        except json.JSONDecodeError:
            pass
    summary["generated_at"] = datetime.now().isoformat(timespec="seconds")
    summary["llm_model"] = f"{llm_client.get_provider()}:{llm_client.get_model()}"
    summary["variant"] = args.variant

    for bench_name in bench_names:
        dataframe, path = load_csv(bench_name)
        if dataframe is None:
            print(f"\n== {bench_name}: no data (bench/data/{bench_name}.csv missing) - skipping")
            summary["benchmarks"][bench_name] = {"skipped": "no data file"}
            continue

        cases = builders[bench_name](dataframe, args.variant)
        if args.limit:
            cases = cases[: args.limit]
        is_sample = "SAMPLE" in path.name
        print(f"\n== {bench_name}: {len(cases)} cases from {path.name}"
              + ("   (SAMPLE DATA - not real results!)" if is_sample else ""))

        previous_models = summary.get("benchmarks", {}).get(bench_name, {}).get("models", {})
        previous = summary.get("benchmarks", {}).get(bench_name, {})
        if previous.get("data_file") != path.name or previous.get("n_cases") != len(cases):
            previous_models = {}  # different data file -> old numbers don't apply
        bench_summary = {"data_file": path.name, "n_cases": len(cases), "is_sample_data": is_sample,
                         "models": dict(previous_models)}
        if bench_name == "pricing" and cases:
            # Dumb baseline: always pick the middle option (the comps' median).
            hits = sum(1 for case in cases if case["options"][2] == case["correct"])
            errors = [abs(float(case["options"][2].strip("$")) - case["actual_price"]) for case in cases]
            bench_summary["baseline_always_median"] = {
                "accuracy": round(hits / len(cases), 4),
                "mean_dollar_error": round(statistics.mean(errors), 2),
            }
            print(f"  [baseline] always-median: {hits}/{len(cases)} correct, "
                  f"${statistics.mean(errors):.2f} average error")
        for model_name in model_names:
            if model_name == "drex" and not drex_client.is_configured():
                print("  [drex] skipped: not configured yet (needs DREX_API_KEY + real _call_drex)")
                bench_summary["models"]["drex"] = {"skipped": "drex not configured"}
                continue
            rows = run_cases(model_name, MODELS[model_name], cases, args.variant)
            pd.DataFrame(rows).to_csv(results_dir / f"{bench_name}_{model_name}.csv", index=False)
            bench_summary["models"][model_name] = summarize(rows)
        summary["benchmarks"][bench_name] = bench_summary

    output_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {output_path.relative_to(REPO_ROOT)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
