"""
Turn bench/results/decisions_full.csv (from bench/hard_bench.py) into the results contract:

    bench/results/summary.csv          model, dataset, n, accuracy, ci_low, ci_high, scam_precision, scam_recall,
                                       consistency, median_latency_ms, p95_latency_ms, cost_per_1k, invalid,
                                       brier, ece, parallel100_seconds
    bench/results/summary_extended.csv summary.csv + scam_f1 + per-category rows for the hard set
    bench/results/coverage_curve.csv   model, dataset, threshold, coverage, accuracy
    bench/results/pricing_summary.csv  method, exact_acc, within_one_acc, median_abs_pct_err, within15_pct
    bench/results/headline.json        speedup_x, cost_ratio_x, drex_auto_coverage_at_100pct, threshold_used, ...

Definitions (also in RESULTS.md):
  n             = number of ITEMS. accuracy = mean over all decisions (items x repeats).
  ci_low/high   = 95% Wilson interval using n items (repeats are not independent, so we don't count them).
  scam P/R      = on items whose offer >= floor (there, the only valid reason to decline is a scam):
                  predicted scam = model declined; true scam = is_scam from the pre-registered set.
  consistency   = share of items where all repeats chose the same option.
  latency       = warm-up call excluded (runner does a throwaway call first).
  brier         = multi-class Brier on per-option probabilities (Drex native; LLM self-reported).
  ece           = expected calibration error, 10 bins, confidence = Drex top probability / LLM confidence/100.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from bench import run_bench  # noqa: E402

RESULTS = REPO_ROOT / "bench" / "results"
DATA = REPO_ROOT / "bench" / "data"
THRESHOLDS = [round(t, 2) for t in np.arange(0.50, 0.99, 0.01)] + [0.99]


def wilson(successes: float, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(centre - half, 4), round(centre + half, 4))


def brier(rows: pd.DataFrame):
    scores = []
    for _, r in rows.iterrows():
        if not r["probs_json"]:
            continue
        probs = json.loads(r["probs_json"])
        scores.append(sum((p - (1.0 if k == r["expected"] else 0.0)) ** 2 for k, p in probs.items()))
    return round(float(np.mean(scores)), 4) if scores else None


def ece(rows: pd.DataFrame, bins: int = 10):
    r = rows.dropna(subset=["top_prob_or_conf"])
    if r.empty:
        return None
    conf, ok = r["top_prob_or_conf"].astype(float).values, r["correct"].astype(float).values
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(conf[mask].mean() - ok[mask].mean())
    return round(float(total), 4)


def offer_at_or_above_floor() -> dict:
    out = {}
    for name, offer_col in (("negotiation.csv", "offer"), ("negotiation_hard.csv", "offer")):
        df = pd.read_csv(DATA / name)
        for _, r in df.iterrows():
            out[r["id"]] = float(r[offer_col]) >= float(r["floor_price"])
    return out


def block_metrics(rows: pd.DataFrame, above_floor: dict) -> dict:
    items = rows.groupby("item_id")
    n = items.ngroups
    acc = rows["correct"].mean()
    lo, hi = wilson(acc * n, n)
    consistency = float(np.mean([g["chosen"].nunique(dropna=False) == 1 for _, g in items])) if n else None
    sub = rows[rows["item_id"].map(above_floor).fillna(False).astype(bool)] if above_floor else rows.iloc[0:0]
    pred, true = sub["chosen"] == "decline", sub["is_scam"] == 1
    tp, fp, fn = int((pred & true).sum()), int((pred & ~true).sum()), int((~pred & true).sum())
    precision = round(tp / (tp + fp), 4) if tp + fp else None
    recall = round(tp / (tp + fn), 4) if tp + fn else None
    f1 = round(2 * precision * recall / (precision + recall), 4) if precision and recall else None
    lat = rows["latency_ms"].dropna()
    return {"n": n, "accuracy": round(acc, 4), "ci_low": lo, "ci_high": hi,
            "scam_precision": precision, "scam_recall": recall, "scam_f1": f1,
            "consistency": round(consistency, 4) if consistency is not None else None,
            "median_latency_ms": round(float(lat.median()), 1) if len(lat) else None,
            "p95_latency_ms": round(float(lat.quantile(0.95)), 1) if len(lat) else None,
            "cost_per_1k": round(1000 * rows["cost_usd"].mean(), 4),
            "invalid": int(rows["invalid"].sum()), "brier": brier(rows), "ece": ece(rows)}


def pricing_summary(d: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(DATA / "pricing.csv")
    actual = dict(zip(df["id"], df["sold_price"].astype(float)))
    nearest_median = {c["id"]: c["nearest_median"] for c in run_bench.build_pricing_cases(df, "v3")}
    rows = d[d["dataset"] == "pricing"]
    order = ("drex_top", "drex_ev", "llm_top", "llm_ev", "drex2_top", "drex2_ev", "code_baseline", "median_baseline")
    methods = {m: [] for m in order}

    def score(method, item, options, prices, correct, chosen_price):
        nearest = min(options, key=lambda o: abs(prices[o] - chosen_price))
        methods.setdefault(method, []).append({
            "exact": nearest == correct,
            "within_one": abs(options.index(nearest) - options.index(correct)) <= 1,
            "pct_err": abs(chosen_price - actual[item]) / actual[item]})

    for _, r in rows.iterrows():
        options, prices = json.loads(r["options"]), json.loads(r["option_prices"])
        model = r["model"]
        if r["chosen"] in prices:
            score(f"{model}_top", r["item_id"], options, prices, r["expected"], prices[r["chosen"]])
        if r["probs_json"]:
            probs = json.loads(r["probs_json"])
            ev = sum(p * prices[o] for o, p in probs.items() if o in prices)
            score(f"{model}_ev", r["item_id"], options, prices, r["expected"], ev)
        if model == "drex":  # baselines once per item
            score("code_baseline", r["item_id"], options, prices, r["expected"], nearest_median[r["item_id"]])
            score("median_baseline", r["item_id"], options, prices, r["expected"], prices[options[2]])
    out = []
    for method, items in methods.items():
        if not items:
            continue
        t = pd.DataFrame(items)
        out.append({"method": method, "n": len(t), "exact_acc": round(t["exact"].mean(), 4),
                    "within_one_acc": round(t["within_one"].mean(), 4),
                    "median_abs_pct_err": round(t["pct_err"].median(), 4),
                    "within15_pct": round((t["pct_err"] <= 0.15).mean(), 4)})
    return pd.DataFrame(out)


def main():
    d = pd.read_csv(RESULTS / "decisions_full.csv", keep_default_na=False, na_values=[""])
    d["probs_json"] = d["probs_json"].fillna("")
    d["option_prices"] = d["option_prices"].fillna("")
    d["correct"] = d["correct"].astype(str).str.lower() == "true"
    d["invalid"] = d["invalid"].astype(str).str.lower() == "true"
    parallel = json.loads((RESULTS / "parallel100.json").read_text()) if (RESULTS / "parallel100.json").exists() else {}
    above_floor = offer_at_or_above_floor()

    summary, extended, curve = [], [], []
    for (model, dataset), rows in d.groupby(["model", "dataset"]):
        m = block_metrics(rows, above_floor if dataset != "pricing" else {})
        m = {"model": model, "dataset": dataset, **m,
             "parallel100_seconds": parallel.get(model, {}).get("seconds")}
        summary.append(m)
        extended.append(m)
        if dataset == "hard":
            for cat, crow in rows.groupby("category"):
                extended.append({"model": model, "dataset": f"hard:{cat}", **block_metrics(crow, above_floor)})
        conf = rows.dropna(subset=["top_prob_or_conf"])
        for t in THRESHOLDS:
            kept = conf[conf["top_prob_or_conf"].astype(float) >= t]
            curve.append({"model": model, "dataset": dataset, "threshold": t,
                          "coverage": round(len(kept) / len(rows), 4) if len(rows) else None,
                          "accuracy": round(kept["correct"].mean(), 4) if len(kept) else None})

    cols = ["model", "dataset", "n", "accuracy", "ci_low", "ci_high", "scam_precision", "scam_recall",
            "consistency", "median_latency_ms", "p95_latency_ms", "cost_per_1k", "invalid", "brier", "ece",
            "parallel100_seconds"]
    pd.DataFrame(summary)[cols].to_csv(RESULTS / "summary.csv", index=False)
    pd.DataFrame(extended).to_csv(RESULTS / "summary_extended.csv", index=False)
    curve_df = pd.DataFrame(curve)
    curve_df.to_csv(RESULTS / "coverage_curve.csv", index=False)
    ps = pricing_summary(d)
    ps.to_csv(RESULTS / "pricing_summary.csv", index=False)

    s = pd.DataFrame(summary).set_index(["model", "dataset"])
    hard_drex = curve_df[(curve_df["model"] == "drex") & (curve_df["dataset"] == "hard")].dropna()
    perfect = hard_drex[hard_drex["accuracy"] >= 1.0]
    if len(perfect):
        best = perfect.sort_values(["coverage", "threshold"], ascending=[False, True]).iloc[0]
        note = "highest-coverage threshold where Drex is 100% accurate on the hard set"
    else:
        best = hard_drex.sort_values(["accuracy", "coverage"], ascending=False).iloc[0]
        note = "Drex never reaches 100% on the hard set at any threshold; this is its best-accuracy threshold"
    headline = {
        "speedup_x": round(s.loc[("llm", "hard"), "median_latency_ms"] / s.loc[("drex", "hard"), "median_latency_ms"], 1),
        "cost_ratio_x": round(s.loc[("llm", "hard"), "cost_per_1k"] / s.loc[("drex", "hard"), "cost_per_1k"], 0),
        "drex_auto_coverage_at_100pct": float(best["coverage"]),
        "threshold_used": float(best["threshold"]),
        "drex_accuracy_at_threshold": float(best["accuracy"]),
        "note": note,
        "dataset": "hard (80 pre-registered messages x 3 repeats)",
    }
    (RESULTS / "headline.json").write_text(json.dumps(headline, indent=2))
    print(pd.DataFrame(summary)[cols].to_string(index=False))
    print(ps.to_string(index=False))
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
