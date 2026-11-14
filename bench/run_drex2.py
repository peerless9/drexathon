"""
Run Drex 2.0 (open weights, on DeepInfra) through the SAME benchmark cases as
run_bench.py - same facts, same questions, same options, both variants.

    python bench/run_drex2.py            # v1 and v2, both benchmarks

It imports run_bench's case builders and metrics instead of copying them, so the
numbers are directly comparable. Results are written to their own files
(bench/results/drex2_summary.json + per-row CSVs) so nothing else is overwritten.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "bench"))

import run_bench  # noqa: E402
from agent import drex2_client  # noqa: E402


def main():
    if not drex2_client.is_configured():
        print("DEEPINFRA_API_KEY missing in .env")
        return

    summary = {"generated_at": datetime.now().isoformat(timespec="seconds"),
               "model": "deepinfra:nace-ai/drex-v2.0", "variants": {}}
    builders = {"pricing": run_bench.build_pricing_cases, "negotiation": run_bench.build_negotiation_cases}

    for variant in ("v1", "v2"):
        results_dir = run_bench.RESULTS_DIR if variant == "v1" else run_bench.RESULTS_DIR / variant
        results_dir.mkdir(parents=True, exist_ok=True)
        summary["variants"][variant] = {}
        for bench_name, build in builders.items():
            dataframe, path = run_bench.load_csv(bench_name)
            if dataframe is None:
                continue
            cases = build(dataframe, variant)
            print(f"\n== {variant} {bench_name}: {len(cases)} cases from {path.name}")
            rows = run_bench.run_cases("drex2", drex2_client, cases, variant)
            pd.DataFrame(rows).to_csv(results_dir / f"{bench_name}_drex2.csv", index=False)
            summary["variants"][variant][bench_name] = run_bench.summarize(rows)

    output_path = run_bench.RESULTS_DIR / "drex2_summary.json"
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {output_path.relative_to(REPO_ROOT)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
