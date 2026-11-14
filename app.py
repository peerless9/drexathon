"""
Drexathon showcase app: an AI agent that sells your used clothes.

    streamlit run app.py

Tabs:
  Sell       - photo + item details -> 5 comp-based price options -> Drex picks one -> listing text
  Negotiate  - same buyer decision sent to Drex and the frontier LLM side by side (the showstopper)
  Results    - benchmark numbers from bench/results/summary.json (or clearly-labeled SAMPLE numbers)

USE_MOCK = True  -> fake (but realistic) answers, no API keys needed.
USE_MOCK = False -> real calls through Levi's agent/drex_client.py and agent/llm_client.py.
"""

import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ============================================================== SETTINGS
USE_MOCK = False  # real Drex + LLM calls (set True to demo without API keys)

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "bench" / "data"
SUMMARY_PATH = REPO_ROOT / "bench" / "results" / "summary.json"

DREX_COLOR = "#6C3BFF"
LLM_COLOR = "#9AA0A6"

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
# Use the exact same wording as the benchmark if Levi's runner is importable.
try:
    from bench.run_bench import (  # noqa: E402
        NEGOTIATION_OPTIONS, NEGOTIATION_QUESTION, PRICING_QUESTION, SELLER_POLICY,
    )
except Exception:
    pass

SCAM_PATTERNS = [
    r"zelle", r"venmo", r"cash ?app", r"paypal f(&|and)f", r"friends (and|&) family",
    r"text me", r"text you", r"call me", r"your number", r"my number", r"email me", r"email it",
    r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b", r"own (shipping )?label", r"my label",
]


def scam_signals(message: str) -> list:
    text = (message or "").lower()
    return [p for p in SCAM_PATTERNS if re.search(p, text)]


def policy_action(list_price, floor_price, offer, message) -> str:
    if scam_signals(message):
        return "decline"
    if round(offer, 2) >= round(0.9 * list_price, 2):
        return "accept"
    if offer >= floor_price:
        return "counter"
    return "decline"


# ============================================================== MODEL CALLS
def _normalize(result: dict) -> dict:
    """Accept both key styles: probs/latency_ms and probabilities/latency_s."""
    probs = result.get("probs", result.get("probabilities"))
    if result.get("latency_ms") is not None:
        latency_ms = float(result["latency_ms"])
    elif result.get("latency_s") is not None:
        latency_ms = float(result["latency_s"]) * 1000
    else:
        latency_ms = None
    return {
        "answer": result.get("answer"),
        "probs": probs,
        "latency_ms": latency_ms,
        "cost_usd": float(result.get("cost_usd") or 0.0),
        "invalid": bool(result.get("invalid", result.get("answer") is None)),
        "error": result.get("error"),
        "model": result.get("model", ""),
    }


def _facts_text(facts) -> str:
    if isinstance(facts, dict):
        return " ".join(f"{k}: {v}" for k, v in facts.items())
    return str(facts)


def _mock_probs(facts, options) -> dict:
    """Plausible probabilities: follow the seller policy for negotiation, center the median for pricing."""
    if options == NEGOTIATION_OPTIONS and isinstance(facts, dict) and "_offer" in facts:
        best = policy_action(facts["_list"], facts["_floor"], facts["_offer"], facts.get("buyer message", ""))
        weights = {o: (0.86 if o == best else 0.07) for o in options}
    else:
        n = len(options)
        center = (n - 1) / 2
        text = str(facts.get("condition", "")).lower() if isinstance(facts, dict) else str(facts).lower()
        if "new" in text or "excellent" in text:
            center += 1
        elif "fair" in text or "poor" in text:
            center -= 1
        weights = {o: 1 / (1 + 1.6 * abs(i - center)) ** 2 for i, o in enumerate(options)}
    jitter = {o: w * random.uniform(0.9, 1.1) for o, w in weights.items()}
    total = sum(jitter.values())
    return {o: round(w / total, 3) for o, w in jitter.items()}


def mock_drex_decide(facts, question: str, options: list) -> dict:
    start = time.perf_counter()
    time.sleep(random.uniform(0.12, 0.28))
    probs = _mock_probs(facts, options)
    return {"answer": max(probs, key=probs.get), "probs": probs,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "cost_usd": 0.0004, "invalid": False, "model": "MOCK drex"}


def mock_llm_decide(facts, question: str, options: list) -> dict:
    start = time.perf_counter()
    time.sleep(random.uniform(1.8, 3.2))
    probs = _mock_probs(facts, options)
    answer = max(probs, key=probs.get)
    if random.random() < 0.12:  # LLMs occasionally miss
        answer = random.choice(options)
    return {"answer": answer, "probs": None,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "cost_usd": 0.0071, "invalid": False, "model": "MOCK frontier LLM"}


def _load_real_clients():
    from agent import drex_client, llm_client
    drex_fn = getattr(drex_client, "drex_decide", None) or drex_client.decide
    llm_fn = getattr(llm_client, "llm_decide", None) or llm_client.decide
    return drex_fn, llm_fn


def _strip_private(facts):
    if isinstance(facts, dict):
        return {k: v for k, v in facts.items() if not k.startswith("_")}
    return facts


def drex_decide(facts, question, options) -> dict:
    if USE_MOCK:
        return _normalize(mock_drex_decide(facts, question, options))
    return _normalize(_load_real_clients()[0](_strip_private(facts), question, options))


def llm_decide(facts, question, options) -> dict:
    if USE_MOCK:
        return _normalize(mock_llm_decide(facts, question, options))
    return _normalize(_load_real_clients()[1](_strip_private(facts), question, options))


def generate_listing(brand, title, size, condition, category, price) -> tuple:
    """LLM writes the listing text (Drex decides, the LLM writes). Falls back to a template."""
    fallback_title = f"{brand} {title} - Size {size} - {condition.title()}"[:80]
    fallback_desc = (
        f"{brand} {title} in {condition} condition, size {size}.\n\n"
        f"• Category: {category}\n• Pre-loved and ready for a new home\n"
        f"• Measurements and extra photos on request\n• Ships within 1 business day\n\n"
        f"Priced at ${price} based on recent sold comps. Offers welcome through the app!"
    )
    if USE_MOCK:
        return fallback_title, fallback_desc, "template (mock)"
    prompt = (
        f"Write a resale listing for: {brand} {title}, size {size}, {condition} condition, "
        f"category {category}, price ${price}. Reply as JSON: "
        '{"title": "<max 80 chars>", "description": "<3-5 short lines, friendly, honest>"}'
    )
    try:
        # Levi's client handles every provider (anthropic / openai / gemini) from .env.
        from agent.llm_client import write_text
        reply = write_text(prompt)
        data = json.loads(re.search(r"\{.*\}", reply, re.DOTALL).group(0))
        return data["title"], data["description"], "LLM"
    except Exception as error:
        return fallback_title, fallback_desc, f"template (LLM failed: {type(error).__name__})"


# ============================================================== DATA
def load_csv(name: str):
    """Real bench/data/<name>.csv if it has rows, else the SAMPLE file. Returns (df, is_sample)."""
    for path, is_sample in ((DATA_DIR / f"{name}.csv", False), (DATA_DIR / f"{name}_SAMPLE.csv", True)):
        if path.exists():
            try:
                df = pd.read_csv(path, comment="#")
            except Exception:
                continue
            if len(df) > 0:
                return df, is_sample
    return None, True


def make_price_options(prices: list) -> list:
    series = pd.Series(prices, dtype=float)
    options = []
    for q in (0.10, 0.30, 0.50, 0.70, 0.90):
        price = int(round(series.quantile(q)))
        while price in options:
            price += 1
        options.append(price)
    return options


SAMPLE_RESULTS = {
    "pricing": {"llm": {"accuracy": 0.55, "median_latency_s": 2.9, "cost_per_1000_usd": 7.40, "invalid": 1},
                "drex": {"accuracy": 0.68, "median_latency_s": 0.21, "cost_per_1000_usd": 0.40, "invalid": 0}},
    "negotiation": {"llm": {"accuracy": 0.83, "median_latency_s": 2.4, "cost_per_1000_usd": 6.10, "invalid": 0},
                    "drex": {"accuracy": 0.93, "median_latency_s": 0.18, "cost_per_1000_usd": 0.40, "invalid": 0}},
}


ROUNDS = {
    "Round 3 · prices as words + closest comps": REPO_ROOT / "bench" / "results" / "v3" / "summary.json",
    "Round 2 · code does the math": REPO_ROOT / "bench" / "results" / "v2" / "summary.json",
    "Round 1 · raw facts": SUMMARY_PATH,
}


def load_results(path=SUMMARY_PATH):
    """Returns (rows, is_sample, note)."""
    rows, is_sample, note = [], True, "No bench/results/summary.json yet."
    if path.exists():
        try:
            summary = json.loads(path.read_text())
            for bench, info in summary.get("benchmarks", {}).items():
                for model, m in (info.get("models") or {}).items():
                    if "skipped" in m or m.get("accuracy") is None:
                        continue
                    # put "model" last: summary.json rows carry their own long model id
                    rows.append({**m, "benchmark": bench, "model": model, "model_id": m.get("model"),
                                 "_sample_data": info.get("is_sample_data", False)})
            if rows:
                is_sample = any(r["_sample_data"] for r in rows)
                note = f"From summary.json (generated {summary.get('generated_at', '?')})"
                if is_sample:
                    note += " - ran on SAMPLE data, not the real benchmark CSVs"
            else:
                note = "summary.json exists but has no completed model runs yet."
        except Exception as error:
            note = f"Could not read summary.json: {error}"
    if not rows:
        for bench, models in SAMPLE_RESULTS.items():
            for model, m in models.items():
                rows.append({"benchmark": bench, "model": model, **m})
        is_sample = True
    return rows, is_sample, note


# ============================================================== UI HELPERS
# Everything below is PRESENTATION ONLY. Model calls, data loading and numbers live above.
import html as _html

st.set_page_config(page_title="ThriftAgent · Drex vs the Giants", page_icon="🧥", layout="wide")

RESULTS_DIR = REPO_ROOT / "bench" / "results"
ASSETS_DIR = REPO_ROOT / "assets"
ACTION_COLOR = {"accept": "#34D399", "counter": "#FBBF24", "decline": "#FB7185"}
ACTION_EMOJI = {"accept": "✅ ACCEPT", "counter": "🔁 COUNTER", "decline": "⛔ DECLINE"}
ACTION_WORD = {"accept": "ACCEPT", "counter": "COUNTER", "decline": "DECLINE"}

# ------------------------------------------------------------- STYLE (single block)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #0B0B10; --panel: rgba(255,255,255,.035); --panel-2: rgba(255,255,255,.06);
  --line: rgba(255,255,255,.09); --text: #ECECF1; --muted: #8B8B99;
  --drex: #6C3BFF; --drex-2: #B14BFF; --llm: #9AA0A6;
  --mint: #34D399; --amber: #FBBF24; --coral: #FB7185;
}
html, body, [class*="css"], .stMarkdown, .stText, p, label, input, textarea { font-family: 'Inter', sans-serif; }
.stApp {
  background:
    radial-gradient(1200px 600px at 85% -10%, rgba(108,59,255,.22), transparent 60%),
    radial-gradient(900px 500px at -10% 10%, rgba(177,75,255,.10), transparent 60%),
    var(--bg);
}
.block-container { padding-top: 1.6rem; max-width: 1280px; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.02em; }
[data-testid="stHeader"] { background: transparent; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
  font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 600; padding: 10px 18px;
  border-radius: 12px 12px 0 0; color: var(--muted);
}
.stTabs [aria-selected="true"] { color: var(--text) !important; background: var(--panel-2); }
.stTabs [data-baseweb="tab-highlight"] { background: linear-gradient(90deg, var(--drex), var(--drex-2)); height: 3px; }

/* buttons + inputs */
.stButton > button[kind="primary"] {
  background: linear-gradient(90deg, var(--drex), var(--drex-2)); border: 0; border-radius: 14px;
  font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.05rem; padding: .7rem 1rem;
  box-shadow: 0 8px 30px rgba(108,59,255,.35);
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.08); }
[data-testid="stFileUploaderDropzone"] {
  background: var(--panel); border: 1.5px dashed rgba(124,77,255,.45); border-radius: 18px; min-height: 150px;
}

/* hero */
.hero { display: grid; grid-template-columns: 1.6fr 1fr; gap: 28px; align-items: center; margin-bottom: 18px; }
.eyebrow { font-family: 'Space Grotesk'; font-size: .78rem; letter-spacing: .22em; text-transform: uppercase;
           color: #B9A6FF; margin-bottom: 8px; }
.hero h1 { font-size: 3.4rem; line-height: 1.02; margin: 0 0 10px; font-weight: 700; }
.hero h1 .grad { background: linear-gradient(90deg, #A78BFA, #E879F9); -webkit-background-clip: text; color: transparent; }
.hero .tag { font-size: 1.15rem; color: #C9C9D6; margin: 0 0 18px; }
.chips { display: flex; flex-wrap: wrap; gap: 10px; }
.chip { font-family: 'Space Grotesk'; font-weight: 600; font-size: 1rem; padding: 9px 14px; border-radius: 999px;
        background: var(--panel-2); border: 1px solid var(--line); font-variant-numeric: tabular-nums; }
.chip b { color: #C4B5FD; }
.hero-art { height: 210px; border-radius: 22px; border: 1px solid var(--line); overflow: hidden; position: relative;
            background: radial-gradient(circle at 30% 30%, rgba(177,75,255,.55), transparent 55%),
                        radial-gradient(circle at 75% 70%, rgba(108,59,255,.65), transparent 55%), #12121A; }
.hero-art .ph { position: absolute; inset: 0; display: grid; place-items: center; font-size: 4.5rem; }
.hero-art .ph small { position: absolute; bottom: 12px; font-size: .75rem; color: rgba(255,255,255,.55);
                      letter-spacing: .1em; text-transform: uppercase; }

/* generic cards */
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 20px; padding: 22px 24px;
        backdrop-filter: blur(8px); }
.label { font-family: 'Space Grotesk'; font-size: .78rem; letter-spacing: .18em; text-transform: uppercase; color: var(--muted); }
.section-title { font-family: 'Space Grotesk'; font-size: 1.5rem; font-weight: 700; margin: 6px 0 2px; }
.section-sub { color: var(--muted); margin-bottom: 14px; }

/* buyer chat bubble */
.chat { display: flex; gap: 12px; align-items: flex-start; margin: 8px 0 18px; }
.avatar { width: 40px; height: 40px; border-radius: 50%; display: grid; place-items: center; font-size: 1.2rem;
          background: var(--panel-2); border: 1px solid var(--line); flex: none; }
.bubble { background: #1C1C26; border: 1px solid var(--line); border-radius: 4px 18px 18px 18px; padding: 14px 18px;
          font-size: 1.12rem; max-width: 820px; }
.bubble .meta { color: var(--muted); font-size: .85rem; margin-top: 6px; font-variant-numeric: tabular-nums; }

/* race cards */
.race { position: relative; border-radius: 22px; padding: 22px 26px; min-height: 210px; border: 1px solid var(--line);
        background: var(--panel); overflow: hidden; }
.race.drex { border: 1.5px solid rgba(124,77,255,.75);
             background: linear-gradient(160deg, rgba(108,59,255,.20), rgba(177,75,255,.06) 60%, rgba(255,255,255,.02)); }
.race.drex.first { animation: pulse 1.6s ease-out 2; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(124,77,255,.65); } 100% { box-shadow: 0 0 0 26px rgba(124,77,255,0); } }
.race .who { display: flex; justify-content: space-between; align-items: center; }
.race .name { font-family: 'Space Grotesk'; font-weight: 700; letter-spacing: .14em; font-size: .9rem; }
.race.drex .name { color: #C4B5FD; } .race.llm .name { color: var(--llm); }
.badge { font-size: .72rem; font-weight: 700; letter-spacing: .1em; padding: 4px 10px; border-radius: 999px;
         background: rgba(124,77,255,.25); color: #DDD6FE; border: 1px solid rgba(124,77,255,.5); }
.action { font-family: 'Space Grotesk'; font-weight: 700; font-size: 3.6rem; line-height: 1; margin: 18px 0 12px; letter-spacing: -.02em; }
.timer { font-family: 'Space Grotesk'; font-size: 1.6rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.timer small { font-size: .95rem; color: var(--muted); font-weight: 500; margin-left: 10px; }
.pill { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: .82rem; font-weight: 600; }
.ok { background: rgba(52,211,153,.14); color: #6EE7B7; border: 1px solid rgba(52,211,153,.35); }
.bad { background: rgba(251,113,133,.14); color: #FDA4AF; border: 1px solid rgba(251,113,133,.35); }
.thinking { font-family: 'Space Grotesk'; font-size: 2.4rem; font-weight: 700; margin: 26px 0 14px;
            background: linear-gradient(90deg, #4B4B58 0%, #D4D4DE 50%, #4B4B58 100%); background-size: 200% 100%;
            -webkit-background-clip: text; color: transparent; animation: shimmer 1.3s linear infinite; }
@keyframes shimmer { from { background-position: 200% 0; } to { background-position: -200% 0; } }
.idle { font-family: 'Space Grotesk'; font-size: 2.4rem; color: #3A3A46; margin: 26px 0 14px; font-weight: 700; }

/* banner */
.banner { margin: 20px 0 8px; border-radius: 22px; padding: 26px; text-align: center; color: white;
          background: linear-gradient(100deg, #5B2BFF, #9B4BFF 55%, #D946EF); box-shadow: 0 18px 50px rgba(108,59,255,.35);
          animation: rise .6s cubic-bezier(.2,.9,.3,1.2); }
@keyframes rise { from { transform: translateY(10px) scale(.97); opacity: 0; } to { transform: none; opacity: 1; } }
.banner .big { font-family: 'Space Grotesk'; font-size: 2.6rem; font-weight: 700; letter-spacing: -.02em; }
.banner small { display: block; font-size: 1.02rem; opacity: .9; margin-top: 4px; font-variant-numeric: tabular-nums; }
.scam { margin-top: 12px; border-radius: 16px; padding: 14px 18px; font-weight: 600; font-size: 1.05rem;
        background: rgba(251,113,133,.12); border: 1px solid rgba(251,113,133,.45); color: #FECDD3; }
.policy { border-radius: 16px; padding: 14px 18px; background: var(--panel); border: 1px solid var(--line); }

/* KPI tiles */
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 8px 0 18px; }
.kpi { background: var(--panel); border: 1px solid var(--line); border-radius: 20px; padding: 18px 20px; }
.kpi.hot { background: linear-gradient(160deg, rgba(108,59,255,.22), rgba(255,255,255,.02)); border-color: rgba(124,77,255,.5); }
.kpi .v { font-family: 'Space Grotesk'; font-size: 2.5rem; font-weight: 700; line-height: 1.05; font-variant-numeric: tabular-nums; }
.kpi .s { color: var(--muted); font-size: .92rem; margin-top: 6px; }
.callout { border-radius: 18px; padding: 16px 20px; background: rgba(251,191,36,.08); border: 1px solid rgba(251,191,36,.35); }
.callout b { color: #FDE68A; }
.sample { background: rgba(251,191,36,.12); color: #FDE68A; border: 1px solid rgba(251,191,36,.45);
          border-radius: 14px; padding: 12px 16px; font-weight: 700; }

/* sell */
.pricetag { display: inline-flex; align-items: baseline; gap: 14px; padding: 18px 24px; border-radius: 20px;
            background: linear-gradient(140deg, rgba(108,59,255,.28), rgba(177,75,255,.08)); border: 1px solid rgba(124,77,255,.55); }
.pricetag .p { font-family: 'Space Grotesk'; font-size: 3.4rem; font-weight: 700; line-height: 1; }
.pricetag .m { color: #C9C9D6; font-variant-numeric: tabular-nums; }
.listing { border-radius: 20px; overflow: hidden; border: 1px solid var(--line); background: #13131B; }
.listing .img { height: 180px; background: radial-gradient(circle at 50% 40%, rgba(177,75,255,.35), transparent 60%), #1A1A24;
                display: grid; place-items: center; font-size: 4rem; }
.listing .body { padding: 16px 20px 20px; }
.listing .t { font-weight: 700; font-size: 1.08rem; }
.listing .pr { font-family: 'Space Grotesk'; font-size: 1.9rem; font-weight: 700; margin: 6px 0 8px; }
.listing .d { color: #C9C9D6; white-space: pre-line; line-height: 1.5; }
.listing .src { color: var(--muted); font-size: .8rem; margin-top: 10px; }

@media (max-width: 900px) { .hero { grid-template-columns: 1fr; } .kpis { grid-template-columns: repeat(2, 1fr); }
  .hero h1 { font-size: 2.4rem; } .action { font-size: 2.6rem; } }
</style>
""", unsafe_allow_html=True)


def _dark(chart):
    """Shared Altair styling: transparent background, quiet axes, no chart junk."""
    return (chart.configure_view(strokeWidth=0)
            .configure(background="transparent")
            .configure_axis(labelColor="#A1A1B0", titleColor="#A1A1B0", gridColor="rgba(255,255,255,.06)",
                            domainColor="rgba(255,255,255,.12)", tickColor="rgba(255,255,255,.12)",
                            labelFont="Inter", titleFont="Inter", labelFontSize=12)
            .configure_header(labelColor="#ECECF1", titleColor="#ECECF1", labelFont="Space Grotesk", labelFontSize=13)
            .configure_title(color="#ECECF1", font="Space Grotesk", fontSize=15, anchor="start"))


def prob_chart(probs: dict, pick: str, height: int = 220):
    df = pd.DataFrame({"option": list(probs.keys()), "probability": list(probs.values())})
    df["picked"] = df["option"] == pick
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=6, height=22).encode(
        y=alt.Y("option:N", sort=list(probs.keys()), title=None),
        x=alt.X("probability:Q", axis=alt.Axis(format="%", tickCount=5), scale=alt.Scale(domain=[0, 1]), title=None),
        color=alt.condition("datum.picked", alt.value("#8B5CF6"), alt.value("#2E2E3A")),
        tooltip=["option", alt.Tooltip("probability:Q", format=".1%")],
    )
    text = alt.Chart(df).mark_text(align="left", dx=6, color="#ECECF1", font="Inter", fontSize=12).encode(
        y=alt.Y("option:N", sort=list(probs.keys())), x="probability:Q", text=alt.Text("probability:Q", format=".0%"))
    return _dark((bars + text).properties(height=height))


MODEL_COLORS = alt.Scale(domain=["drex", "drex2", "llm"], range=["#8B5CF6", "#C084FC", "#6B7280"])


def metric_bar(df: pd.DataFrame, field: str, title: str, fmt: str):
    return _dark(alt.Chart(df).mark_bar(cornerRadiusEnd=6).encode(
        x=alt.X(f"{field}:Q", title=None, axis=alt.Axis(format=fmt, tickCount=4)),
        y=alt.Y("model:N", title=None),
        row=alt.Row("benchmark:N", title=None, header=alt.Header(labelFontWeight="bold", labelAngle=0, labelAlign="left")),
        color=alt.Color("model:N", scale=MODEL_COLORS, legend=None),
        tooltip=["benchmark", "model", alt.Tooltip(f"{field}:Q", format=fmt)],
    ).properties(title=title, height=64, width=260))


def _hard_summary():
    """Pre-registered hard-set results (bench/hard_bench.py), if present."""
    path = RESULTS_DIR / "summary.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _headline():
    """Numbers for the hero chips: hard set if available, else latest round, else clearly-marked defaults."""
    hs = _hard_summary()
    try:
        head = json.loads((RESULTS_DIR / "headline.json").read_text())
        hard = hs[hs["dataset"] == "hard"].set_index("model")
        return {"speed": f"{head['speedup_x']:.1f}×", "cost": f"{head['cost_ratio_x']:.0f}×",
                "acc": f"{hard.loc['drex', 'accuracy']:.0%}", "src": "80 pre-registered hard buyer messages × 3 runs"}
    except Exception:
        return {"speed": "6×", "cost": "250×", "acc": "100%", "src": "negotiation benchmark"}


def _img_or_placeholder(name: str, emoji: str, caption: str) -> str:
    for ext in ("png", "jpg", "jpeg", "webp"):
        path = ASSETS_DIR / f"{name}.{ext}"
        if path.exists():
            import base64
            data = base64.b64encode(path.read_bytes()).decode()
            return f'<img src="data:image/{ext};base64,{data}" style="width:100%;height:100%;object-fit:cover">'
    return f'<div class="ph">{emoji}<small>{caption}</small></div>'


# ------------------------------------------------------------- SIDEBAR
with st.sidebar:
    st.markdown("### 🧥 ThriftAgent")
    st.caption("Reselling is hundreds of tiny decisions. An LLM writes the text; **Drex makes the decisions.**")
    if USE_MOCK:
        st.warning("**MOCK MODE** - answers and timings are simulated. Set `USE_MOCK = False` in app.py "
                   "once the Drex + LLM clients work.")
    else:
        st.success("**LIVE MODE** - real Drex + frontier LLM calls.")
    if st.session_state.get("history"):
        hist = st.session_state["history"]
        drex_total = sum(h["drex_ms"] for h in hist)
        llm_total = sum(h["llm_ms"] for h in hist)
        st.metric("Decisions this session", len(hist))
        st.metric("Avg speedup", f"{llm_total / max(drex_total, 1):.1f}×")

# ------------------------------------------------------------- HERO
_h = _headline()
st.markdown(f"""
<div class="hero">
  <div>
    <div class="eyebrow">Drexathon · Drex vs the Giants</div>
    <h1>Thrift<span class="grad">Agent</span></h1>
    <p class="tag">An AI agent that sells your used clothes. <b>The LLM writes. Drex decides.</b></p>
    <div class="chips">
      <span class="chip">⚡ <b>{_h['speed']}</b> faster</span>
      <span class="chip">💸 <b>{_h['cost']}</b> cheaper</span>
      <span class="chip">🎯 <b>{_h['acc']}</b> accuracy</span>
    </div>
    <div style="color:var(--muted);font-size:.82rem;margin-top:10px">Measured on {_h['src']}, vs Claude Opus 5.5.</div>
  </div>
  <div class="hero-art">{_img_or_placeholder("hero", "🧥", "Adobe Firefly hero · assets/hero.png")}</div>
</div>
""", unsafe_allow_html=True)

if USE_MOCK:
    st.markdown('<div class="sample">⚠️ MOCK MODE - answers and timings below are simulated.</div>',
                unsafe_allow_html=True)

tab_negotiate, tab_sell, tab_results = st.tabs(["⚡ Negotiate", "🏷️ Sell", "📊 Results"])

# ============================================================== NEGOTIATE TAB
with tab_negotiate:
    neg_df, neg_is_sample = load_csv("negotiation")
    st.markdown('<div class="section-title">Same decision. Two models. One race.</div>'
                '<div class="section-sub">A buyer sends an offer. Drex and a frontier LLM decide in parallel - '
                'accept, counter or decline - under the same seller policy.</div>', unsafe_allow_html=True)

    preset_labels = ["✏️ Custom"]
    if neg_df is not None:
        preset_labels += [f"{r['id']} · {r['item']} · \"{str(r['buyer_message'])[:45]}…\""
                          for _, r in neg_df.iterrows()]
    chosen = st.selectbox("Load a buyer scenario", preset_labels)
    preset = None
    if chosen != preset_labels[0] and neg_df is not None:
        preset = neg_df.iloc[preset_labels.index(chosen) - 1]

    def default(field, fallback):
        return preset[field] if preset is not None else fallback

    key = f"neg_{chosen}"
    with st.expander("Listing & offer details", expanded=preset is None):
        c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1])
        item = c1.text_input("Item", default("item", "Patagonia Better Sweater Fleece (M)"), key=key + "item")
        list_price = c2.number_input("List $", min_value=1.0, value=float(default("list_price", 85)), key=key + "lp")
        floor_price = c3.number_input("Floor $", min_value=0.0, value=float(default("floor_price", 60)), key=key + "fp")
        days_listed = c4.number_input("Days listed", min_value=0, value=int(default("days_listed", 5)), key=key + "dl")
        c5, c6 = st.columns([3.2, 1])
        message = c5.text_area("💬 Buyer message", default("buyer_message", "hey would u take 70? can pay rn"),
                               height=90, key=key + "msg")
        offer = c6.number_input("Offer $", min_value=0.0, value=float(default("offer", 70)), key=key + "off")

    st.markdown(f"""
    <div class="chat"><div class="avatar">🛍️</div>
      <div class="bubble">{_html.escape(str(message))}
        <div class="meta">Offer <b>${offer:,.2f}</b> on <b>{_html.escape(str(item))}</b> · listed ${list_price:,.0f}
        · floor ${floor_price:,.0f} · {100 * offer / list_price:.1f}% of list</div></div></div>
    """, unsafe_allow_html=True)

    go = st.button("⚡ Send to Drex and the LLM", type="primary", use_container_width=True)

    facts = {
        "seller policy": SELLER_POLICY,
        "item": item,
        "list price": f"${list_price:g}",
        "floor price (lowest the seller will take)": f"${floor_price:g}",
        "days listed": days_listed,
        "buyer message": message,
        "offer": f"${offer:g}",
        "offer as % of list price": f"{100 * offer / list_price:.1f}%",
        "90% of list price": f"${0.9 * list_price:.2f}",
        # private hints for the mock only (stripped before real calls)
        "_list": list_price, "_floor": floor_price, "_offer": offer,
    }
    expected = policy_action(list_price, floor_price, offer, message)

    col_drex, col_llm = st.columns(2, gap="large")
    slot_drex, slot_llm = col_drex.empty(), col_llm.empty()
    banner_slot = st.empty()

    def render_card(slot, name, result, is_drex, first=False):
        cls = "race drex" if is_drex else "race llm"
        if first:
            cls += " first"
        badge = '<span class="badge">ANSWERED FIRST</span>' if first else ""
        head = f'<div class="who"><span class="name">{name.upper()}</span>{badge}</div>'
        if result is None:
            slot.markdown(f'<div class="{cls}">{head}<div class="thinking">thinking…</div>'
                          f'<div class="timer" style="color:var(--muted)">⏱ running</div></div>', unsafe_allow_html=True)
            return
        if result.get("answer") == "—":
            slot.markdown(f'<div class="{cls}">{head}<div class="idle">ready</div>'
                          f'<div class="timer" style="color:var(--muted)">⏱ – ms</div></div>', unsafe_allow_html=True)
            return
        if result["error"] or result["invalid"]:
            body = (f'<div class="action" style="color:var(--coral);font-size:2.2rem">invalid</div>'
                    f'<div style="color:var(--muted)">{_html.escape(str(result["error"] or "unparseable answer"))[:160]}</div>')
        else:
            ans = result["answer"]
            ok = ans == expected
            pill = ('<span class="pill ok">✓ matches policy</span>' if ok
                    else '<span class="pill bad">✗ wrong call</span>')
            body = (f'<div class="action" style="color:{ACTION_COLOR.get(ans, "#ECECF1")}">'
                    f'{ACTION_WORD.get(ans, str(ans).upper())}</div>{pill}')
        slot.markdown(
            f'<div class="{cls}">{head}{body}'
            f'<div class="timer" style="margin-top:14px">{(result["latency_ms"] or 0):,.0f} ms'
            f'<small>${result["cost_usd"]:.6f} per decision</small></div></div>', unsafe_allow_html=True)

    if go:
        render_card(slot_drex, "Drex", None, True)
        render_card(slot_llm, "Frontier LLM", None, False)
        results, order = {}, []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(drex_decide, facts, NEGOTIATION_QUESTION, NEGOTIATION_OPTIONS): "drex",
                       pool.submit(llm_decide, facts, NEGOTIATION_QUESTION, NEGOTIATION_OPTIONS): "llm"}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as error:
                    results[name] = {"answer": None, "probs": None, "latency_ms": None, "cost_usd": 0.0,
                                     "invalid": True, "error": f"{type(error).__name__}: {error}", "model": ""}
                order.append(name)
                if name == "drex":
                    render_card(slot_drex, "Drex", results[name], True, first=(order[0] == "drex"))
                else:
                    render_card(slot_llm, "Frontier LLM", results[name], False, first=False)
        st.session_state["neg"] = {"results": results, "expected": expected,
                                   "signals": scam_signals(message), "first": order[0] if order else None}
        d_ms, l_ms = results["drex"]["latency_ms"], results["llm"]["latency_ms"]
        if d_ms and l_ms and not (results["drex"]["invalid"] or results["llm"]["invalid"]):
            st.session_state.setdefault("history", []).append({"drex_ms": d_ms, "llm_ms": l_ms})
    elif st.session_state.get("neg"):
        res = st.session_state["neg"]["results"]
        render_card(slot_drex, "Drex", res["drex"], True, first=st.session_state["neg"].get("first") == "drex")
        render_card(slot_llm, "Frontier LLM", res["llm"], False)
    else:
        idle = {"answer": "—", "latency_ms": 0, "cost_usd": 0, "invalid": False, "error": None, "probs": None}
        render_card(slot_drex, "Drex", idle, True)
        render_card(slot_llm, "Frontier LLM", idle, False)

    neg = st.session_state.get("neg")
    if neg:
        d, l = neg["results"]["drex"], neg["results"]["llm"]
        both_valid = not (d["invalid"] or d["error"] or l["invalid"] or l["error"])
        if not both_valid:
            banner_slot.markdown('<div class="sample">⚠️ One model returned an error, so no speed comparison '
                                 'is shown for this run.</div>', unsafe_allow_html=True)
        elif d["latency_ms"] and l["latency_ms"]:
            speedup = l["latency_ms"] / d["latency_ms"]
            cheaper = (l["cost_usd"] / d["cost_usd"]) if d["cost_usd"] else None
            extra = f" · {cheaper:,.0f}× cheaper" if cheaper else ""
            banner_slot.markdown(
                f'<div class="banner"><div class="big">⚡ Drex was {speedup:.1f}× faster{extra}</div>'
                f'<small>{d["latency_ms"]:,.0f} ms vs {l["latency_ms"]:,.0f} ms on the exact same decision</small></div>',
                unsafe_allow_html=True)
        i1, i2 = st.columns([1.2, 1], gap="large")
        with i1:
            if d["probs"]:
                st.markdown('<div class="label" style="margin-top:10px">Drex · probability per option</div>',
                            unsafe_allow_html=True)
                st.altair_chart(prob_chart(d["probs"], d["answer"], height=130), use_container_width=True)
        with i2:
            exp = neg["expected"]
            st.markdown(f'<div class="policy" style="margin-top:12px"><div class="label">Seller policy says</div>'
                        f'<div class="action" style="font-size:2rem;margin:8px 0 2px;color:{ACTION_COLOR[exp]}">'
                        f'{ACTION_WORD[exp]}</div></div>', unsafe_allow_html=True)
            if neg["signals"]:
                st.markdown('<div class="scam">🚩 Scam shield triggered - off-platform payment, contact or '
                            'shipping-label request detected.</div>', unsafe_allow_html=True)
            st.caption("The LLM writes text and has no calibrated probabilities - Drex returns one per option.")

    if neg_is_sample and neg_df is not None:
        st.markdown('<div class="sample">⚠️ Scenarios are from the SAMPLE file.</div>', unsafe_allow_html=True)

# ============================================================== SELL TAB
with tab_sell:
    pricing_df, pricing_is_sample = load_csv("pricing")
    st.markdown('<div class="section-title">Listing studio</div>'
                '<div class="section-sub">Drop a photo, confirm the details. Code finds the sold comps, '
                'Drex picks the price, the LLM writes the listing.</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        photo = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp", "heic"])
        if photo is not None:
            st.image(photo, use_container_width=True)
        categories = sorted(pricing_df["category"].dropna().unique()) if pricing_df is not None else []
        if not categories:
            categories = ["jeans", "jackets", "sneakers", "sweaters"]
        c1, c2 = st.columns(2)
        brand = c1.text_input("Brand", "Levi's")
        size = c2.text_input("Size", "32x32")
        title = st.text_input("Item", "501 Original Fit Medium Wash")
        c3, c4 = st.columns(2)
        condition = c3.selectbox("Condition", ["new with tags", "like new", "good", "fair"], index=2)
        category = c4.selectbox("Category", categories,
                                index=categories.index("jeans") if "jeans" in categories else 0)
        price_clicked = st.button("💸 Price it with Drex", type="primary", use_container_width=True)

    with right:
        if pricing_df is None:
            st.info("No pricing data yet. Fill in bench/data/pricing.csv.")
        else:
            if pricing_is_sample:
                st.markdown('<div class="sample">⚠️ Using SAMPLE comps - fill in bench/data/pricing.csv.</div>',
                            unsafe_allow_html=True)
            comps = pricing_df[pricing_df["category"] == category]
            if len(comps) < 2:
                st.info(f"Need at least 2 sold comps in '{category}'.")
            elif price_clicked:
                options = [f"${p}" for p in make_price_options(comps["sold_price"].tolist())]
                comp_lines = [f"{r['brand']} {r['title']} (size {r['size']}, {r['condition']}): sold ${r['sold_price']}"
                              for _, r in comps.iterrows()]
                facts = {"category": category, "brand": brand, "title": title, "size": size,
                         "condition": condition, "recently sold comparable items": "; ".join(comp_lines)}
                with st.spinner("Drex is deciding…"):
                    result = drex_decide(facts, PRICING_QUESTION, options)
                st.session_state["sell"] = {"result": result, "options": options, "n_comps": len(comps),
                                            "item": (brand, title, size, condition, category)}

            sell = st.session_state.get("sell")
            if sell:
                result = sell["result"]
                if result["error"] or result["invalid"]:
                    st.error(f"Drex didn't return a valid answer. {result['error'] or ''}")
                else:
                    st.markdown(f'<div class="pricetag"><span class="p">{result["answer"]}</span>'
                                f'<span class="m">Drex picked in <b>{result["latency_ms"]:.0f} ms</b><br>'
                                f'from {sell["n_comps"]} sold comps</span></div>', unsafe_allow_html=True)
                    if result["probs"]:
                        st.altair_chart(prob_chart(result["probs"], result["answer"], height=190),
                                        use_container_width=True)
                    st.caption("Options = 10th / 30th / 50th / 70th / 90th percentile of recent sold comps. "
                               "Honest note: on our 76-item pricing benchmark the LLM is still more accurate.")

                    b, t, s, c, cat = sell["item"]
                    price = result["answer"].lstrip("$")
                    if "listing" not in sell:
                        with st.spinner("Writing listing…"):
                            sell["listing"] = generate_listing(b, t, s, c, cat, price)
                    l_title, l_desc, source = sell["listing"]
                    st.markdown(f"""
                    <div class="listing" style="margin-top:10px"><div class="img">👖</div>
                      <div class="body"><div class="label">Marketplace preview · {"written by the LLM" if source == "LLM" else "template (LLM unavailable)"}</div>
                        <div class="t" style="margin-top:6px">{_html.escape(str(l_title))}</div>
                        <div class="pr">${_html.escape(price)}</div>
                        <div class="d">{_html.escape(str(l_desc))}</div>
                        <div class="src">Text source: {_html.escape(str(source))}</div></div></div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="card" style="text-align:center;padding:48px 24px">'
                            '<div style="font-size:3rem">🏷️</div><div class="section-title">Ready to price</div>'
                            '<div class="section-sub">Fill in the item and hit <b>Price it with Drex</b>.</div></div>',
                            unsafe_allow_html=True)

# ============================================================== RESULTS TAB
with tab_results:
    hs = _hard_summary()
    if hs is not None and {"drex", "llm"} <= set(hs["model"]) and "hard" in set(hs["dataset"]):
        hard = hs[hs["dataset"] == "hard"].set_index("model")
        dx, lm = hard.loc["drex"], hard.loc["llm"]
        st.markdown('<div class="section-title">Scoreboard · pre-registered hard test</div>'
                    '<div class="section-sub">80 messy buyer messages (near-threshold offers, subtle scams, '
                    'prompt injection, Spanish, typos) committed to git before any model saw them · '
                    '3 runs each · identical inputs.</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kpis">
          <div class="kpi hot"><div class="label">Accuracy</div>
            <div class="v">{dx['accuracy']:.0%} <span style="font-size:1.1rem;color:var(--muted)">vs {lm['accuracy']:.0%}</span></div>
            <div class="s">95% CI {dx['ci_low']:.0%}–{dx['ci_high']:.0%} · same as the LLM</div></div>
          <div class="kpi hot"><div class="label">Speed</div>
            <div class="v">{lm['median_latency_ms'] / dx['median_latency_ms']:.1f}×</div>
            <div class="s">{dx['median_latency_ms']:,.0f} ms vs {lm['median_latency_ms']:,.0f} ms median</div></div>
          <div class="kpi hot"><div class="label">Cost</div>
            <div class="v">{lm['cost_per_1k'] / dx['cost_per_1k']:,.0f}×</div>
            <div class="s">${dx['cost_per_1k']:.3f} vs ${lm['cost_per_1k']:.2f} per 1,000 decisions</div></div>
          <div class="kpi"><div class="label">Calibration error ↓</div>
            <div class="v">{dx['ece']:.3f}</div>
            <div class="s">vs {lm['ece']:.3f} for the LLM's self-reported confidence</div></div>
        </div>
        """, unsafe_allow_html=True)
        show = hs[hs["dataset"].isin(["hard", "original"])].copy()
        st.dataframe(show[["model", "dataset", "n", "accuracy", "ci_low", "ci_high", "scam_precision", "scam_recall",
                           "consistency", "median_latency_ms", "p95_latency_ms", "cost_per_1k", "ece",
                           "parallel100_seconds"]],
                     hide_index=True, use_container_width=True, column_config={
                         "accuracy": st.column_config.NumberColumn(format="percent"),
                         "ci_low": st.column_config.NumberColumn("CI low", format="percent"),
                         "ci_high": st.column_config.NumberColumn("CI high", format="percent"),
                         "cost_per_1k": st.column_config.NumberColumn("$ / 1k", format="$%.3f"),
                         "parallel100_seconds": st.column_config.NumberColumn("100 parallel (s)", format="%.1f")})
        ps_path = RESULTS_DIR / "pricing_summary.csv"
        if ps_path.exists():
            ps = pd.read_csv(ps_path).set_index("method")
            try:
                st.markdown(f"""
                <div class="callout" style="margin:14px 0 22px"><b>Where the LLM still wins: pricing.</b>
                On 76 real eBay sold items the LLM picks the closest price {ps.loc['llm_top', 'exact_acc']:.0%} of the time
                vs Drex {ps.loc['drex_top', 'exact_acc']:.0%} (plain-code comps baseline
                {ps.loc['code_baseline', 'exact_acc']:.0%}). Pricing needs world knowledge - so in ThriftAgent the LLM
                prices, Drex decides, and code does the math.</div>""", unsafe_allow_html=True)
            except KeyError:
                pass
        st.divider()

    st.markdown('<div class="section-title">Round by round</div>'
                '<div class="section-sub">How we got here: each round changes only what plain code pre-computes - '
                'never the model.</div>', unsafe_allow_html=True)
    available = [name for name, path in ROUNDS.items() if path.exists()] or ["Round 1 · raw facts"]
    chosen_round = st.radio("Benchmark round", available, horizontal=True, label_visibility="collapsed")
    rows, is_sample, note = load_results(ROUNDS[chosen_round])

    existing = [name for name in reversed(list(ROUNDS)) if ROUNDS[name].exists()]
    if len(existing) >= 2:
        per_round = {name: {(r["benchmark"], r["model"]): r["accuracy"] for r in load_results(ROUNDS[name])[0]}
                     for name in existing}
        keys = sorted(set.intersection(*(set(v) for v in per_round.values())))
        if keys:
            with st.expander("📈 Accuracy round by round", expanded=True):
                short = {name: name.split(" · ")[0].lower() for name in existing}
                table = pd.DataFrame([{"benchmark": k[0], "model": k[1],
                                       **{short[n]: per_round[n][k] for n in existing}} for k in keys])
                st.dataframe(table, hide_index=True, use_container_width=True, column_config={
                    short[n]: st.column_config.NumberColumn(format="percent") for n in existing})
                st.caption("In every round both models get identical facts and options; "
                           "rounds differ only in what plain code pre-computes for them.")
    if is_sample:
        st.markdown('<div class="sample">⚠️ SAMPLE NUMBERS - not real results. Run '
                    '<code>python bench/run_bench.py</code> to fill bench/results/summary.json.</div>',
                    unsafe_allow_html=True)
    st.caption(note)

    df = pd.DataFrame(rows)
    df["median_latency_ms"] = df["median_latency_s"] * 1000
    df["model"] = df["model"].astype(str)

    agg = df.groupby("model")[["accuracy", "median_latency_ms", "cost_per_1000_usd"]].mean()
    if {"drex", "llm"} <= set(agg.index):
        h1, h2, h3 = st.columns(3)
        h1.metric("Accuracy (Drex, avg of benchmarks)", f"{agg.loc['drex', 'accuracy']:.0%}",
                  f"{(agg.loc['drex', 'accuracy'] - agg.loc['llm', 'accuracy']) * 100:+.0f} pts vs LLM")
        h2.metric("Speed", f"{agg.loc['llm', 'median_latency_ms'] / agg.loc['drex', 'median_latency_ms']:.1f}× faster",
                  f"{agg.loc['drex', 'median_latency_ms']:.0f} ms vs {agg.loc['llm', 'median_latency_ms']:.0f} ms",
                  delta_color="off")
        if agg.loc["drex", "cost_per_1000_usd"]:
            h3.metric("Cost", f"{agg.loc['llm', 'cost_per_1000_usd'] / agg.loc['drex', 'cost_per_1000_usd']:.0f}× cheaper",
                      f"${agg.loc['drex', 'cost_per_1000_usd']:.4f} vs ${agg.loc['llm', 'cost_per_1000_usd']:.2f} per 1k",
                      delta_color="off")

    table = df[["benchmark", "model", "accuracy", "median_latency_ms", "cost_per_1000_usd"]
               + [c for c in ("invalid", "n") if c in df.columns]].copy()
    st.dataframe(table, hide_index=True, use_container_width=True, column_config={
        "accuracy": st.column_config.NumberColumn("Accuracy", format="percent"),
        "median_latency_ms": st.column_config.NumberColumn("Median latency (ms)", format="%.0f"),
        "cost_per_1000_usd": st.column_config.NumberColumn("$ per 1,000 decisions", format="$%.4f"),
    })

    g1, g2, g3 = st.columns(3)
    g1.altair_chart(metric_bar(df, "accuracy", "Accuracy ↑", "%"), use_container_width=True)
    g2.altair_chart(metric_bar(df, "median_latency_ms", "Median latency, ms ↓", ",.0f"), use_container_width=True)
    g3.altair_chart(metric_bar(df, "cost_per_1000_usd", "$ per 1,000 decisions ↓", "$.2f"), use_container_width=True)
