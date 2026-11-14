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
DREX2_COLOR = "#B14BFF"  # Drex 2.0 (open weights, DeepInfra) - a lighter purple, same family
LLM_COLOR = "#9AA0A6"

# Short keys in summary.json -> names people read in the Results tab.
MODEL_LABELS = {
    "drex": "Drex Max (Nace)",
    "drex2": "Drex 2.0 open weights (DeepInfra)",
    "llm": "Claude Opus 5.5",
}

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


DREX_ENGINE = {"engine": "drex"}  # set from the sidebar: "drex" (Drex Max, Nace) or "drex2" (Drex 2.0, DeepInfra)


def drex_decide(facts, question, options) -> dict:
    if USE_MOCK:
        return _normalize(mock_drex_decide(facts, question, options))
    if DREX_ENGINE["engine"] == "drex2":
        from agent import drex2_client
        return _normalize(drex2_client.decide(_strip_private(facts), question, options))
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
ACTION_COLOR = {"accept": "#19E3D1", "counter": "#FFC857", "decline": "#FF4D6D"}
ACTION_EMOJI = {"accept": "✅ ACCEPT", "counter": "🔁 COUNTER", "decline": "⛔ DECLINE"}
ACTION_WORD = {"accept": "ACCEPT", "counter": "COUNTER", "decline": "DECLINE"}

# ------------------------------------------------------------- STYLE (single block)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@600;700;800&family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

/* ===================================================== 1. PALETTE (neon dusk) */
:root {
  --bg: #12061F; --bg-2: #1B0B33; --panel: rgba(18,6,31,.72); --panel-2: rgba(255,255,255,.07);
  --line: rgba(255,255,255,.12); --text: #FFF4F8; --muted: #B8ACCB;
  --pink: #FF2E88; --tangerine: #FF8A3D; --gold: #FFC857; --teal: #19E3D1; --coral: #FF4D6D;
  --lav: #A59BB8; --violet: #7B5CFF;
  --sunset: linear-gradient(90deg, #FF2E88, #FF8A3D, #FFC857, #FF8A3D, #FF2E88);
  --ocean: linear-gradient(90deg, #19E3D1, #7B5CFF);
  --display: 'Unbounded', 'Space Grotesk', sans-serif;
}
html, body, [class*="css"], .stMarkdown, p, label, input, textarea { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: var(--display) !important; letter-spacing: -.02em; }
[data-testid="stHeader"] { background: transparent; }

/* ===================================================== 2. LIVING BACKGROUND */
.stApp {
  background: linear-gradient(120deg, #12061F, #2A0B3D, #3B0D3A, #1B0B33, #0E1A3A, #12061F);
  background-size: 400% 400%; animation: drift 26s ease-in-out infinite;
}
@keyframes drift { 0%,100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
.bg-layer { position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden; }
.orb { position: absolute; border-radius: 50%; filter: blur(70px); opacity: .55; }
.orb.o1 { width: 520px; height: 520px; background: #FF2E88; top: -160px; right: -120px; animation: float1 22s ease-in-out infinite; }
.orb.o2 { width: 460px; height: 460px; background: #FF8A3D; bottom: -200px; left: -140px; opacity: .35; animation: float2 28s ease-in-out infinite; }
.orb.o3 { width: 380px; height: 380px; background: #19E3D1; top: 40%; left: 55%; opacity: .18; animation: float3 32s ease-in-out infinite; }
@keyframes float1 { 50% { transform: translate(-140px, 120px) scale(1.15); } }
@keyframes float2 { 50% { transform: translate(180px, -120px) scale(1.2); } }
@keyframes float3 { 50% { transform: translate(-220px, -80px) scale(.85); } }
.grid-floor { position: absolute; left: -50%; right: -50%; bottom: -10%; height: 46%;
  background-image: linear-gradient(rgba(255,46,136,.45) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,46,136,.45) 1px, transparent 1px);
  background-size: 64px 64px; transform: perspective(420px) rotateX(62deg); transform-origin: top;
  mask-image: linear-gradient(to bottom, transparent, #000 35%, #000 70%, transparent);
  -webkit-mask-image: linear-gradient(to bottom, transparent, #000 35%, #000 70%, transparent);
  animation: grid-run 2.4s linear infinite; opacity: .55; }
@keyframes grid-run { to { background-position: 0 64px, 0 0; } }
.sun { position: absolute; left: 50%; bottom: 30%; width: 340px; height: 340px; margin-left: -170px; border-radius: 50%;
  background: linear-gradient(#FFC857, #FF8A3D 45%, #FF2E88);
  mask-image: repeating-linear-gradient(to bottom, #000 0 18px, transparent 18px 24px);
  -webkit-mask-image: repeating-linear-gradient(to bottom, #000 0 18px, transparent 18px 24px);
  opacity: .16; filter: blur(1px); animation: sunpulse 6s ease-in-out infinite; }
@keyframes sunpulse { 50% { opacity: .24; transform: translateY(-8px); } }
.scan { position: absolute; inset: 0;
  background: repeating-linear-gradient(to bottom, rgba(255,255,255,.025) 0 1px, transparent 1px 3px);
  animation: scanroll 8s linear infinite; }
@keyframes scanroll { to { background-position: 0 300px; } }
.block-container { position: relative; z-index: 1; padding-top: 1.4rem; max-width: 1280px; }
[data-testid="stSidebar"] { background: rgba(18,6,31,.92); }

/* ===================================================== 3. HERO */
.hero { display: grid; grid-template-columns: 1.6fr 1fr; gap: 28px; align-items: center; margin-bottom: 10px; }
.eyebrow { font-family: var(--display); font-size: .72rem; letter-spacing: .3em; text-transform: uppercase; color: var(--gold);
           animation: fadeUp .7s ease both; }
.hero h1 { font-family: var(--display); font-size: 4.4rem; line-height: 1; margin: 8px 0 12px; font-weight: 800; }
.hero h1 .ltr { display: inline-block; background: var(--sunset); background-size: 300% 100%;
  -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: letterIn .7s cubic-bezier(.2,.9,.3,1.3) both, sweep 6s linear infinite; }
@keyframes letterIn { from { transform: translateY(60%) rotate(8deg); opacity: 0; filter: blur(6px); } to { transform: none; opacity: 1; filter: none; } }
@keyframes sweep { to { background-position: 300% 0; } }
.hero .tag { font-size: 1.2rem; color: #E9DDF5; margin: 0 0 18px; animation: fadeUp .8s ease .9s both; }
.hero .tag b { color: #fff; border-right: 2px solid var(--pink); padding-right: 4px; animation: caret 1s steps(1) infinite; }
@keyframes caret { 50% { border-color: transparent; } }
@keyframes fadeUp { from { transform: translateY(14px); opacity: 0; } to { transform: none; opacity: 1; } }
.chips { display: flex; flex-wrap: wrap; gap: 10px; }
.chip { position: relative; overflow: hidden; font-family: 'Space Grotesk'; font-weight: 700; font-size: 1.05rem; padding: 10px 16px;
  border-radius: 999px; background: rgba(18,6,31,.75); border: 1px solid rgba(255,46,136,.45);
  font-variant-numeric: tabular-nums; box-shadow: 0 0 22px rgba(255,46,136,.18);
  animation: popIn .6s cubic-bezier(.2,1.4,.4,1) both; }
.chip:nth-child(1) { animation-delay: 1.1s; } .chip:nth-child(2) { animation-delay: 1.3s; } .chip:nth-child(3) { animation-delay: 1.5s; }
.chip b { color: var(--gold); }
.chip::after { content: ""; position: absolute; inset: 0; transform: translateX(-120%);
  background: linear-gradient(100deg, transparent 30%, rgba(255,255,255,.35) 50%, transparent 70%);
  animation: shine 4.5s ease-in-out 2.2s infinite; }
@keyframes shine { 0% { transform: translateX(-120%); } 30%,100% { transform: translateX(120%); } }
@keyframes popIn { from { transform: scale(.4); opacity: 0; } to { transform: none; opacity: 1; } }
.hero-art { height: 220px; border-radius: 24px; border: 1px solid rgba(255,46,136,.4); overflow: hidden; position: relative;
  background: radial-gradient(circle at 30% 30%, rgba(255,46,136,.6), transparent 55%),
              radial-gradient(circle at 75% 75%, rgba(255,138,61,.55), transparent 55%), #1B0B33;
  box-shadow: 0 0 60px rgba(255,46,136,.25); animation: fadeUp .9s ease .4s both, bob 7s ease-in-out 1.3s infinite; }
@keyframes bob { 50% { transform: translateY(-8px) rotate(-.6deg); } }
.hero-art .ph { position: absolute; inset: 0; display: grid; place-items: center; font-size: 4.8rem; }
.hero-art .ph small { position: absolute; bottom: 12px; font-size: .72rem; color: rgba(255,255,255,.6); letter-spacing: .12em; text-transform: uppercase; }
.ticker { position: relative; overflow: hidden; white-space: nowrap; margin: 14px 0 18px; padding: 10px 0;
  border-top: 1px solid rgba(255,46,136,.35); border-bottom: 1px solid rgba(255,46,136,.35); background: rgba(18,6,31,.75); }
.ticker .track { display: inline-block; animation: marquee 32s linear infinite; font-family: 'Space Grotesk'; font-weight: 700;
  letter-spacing: .08em; font-size: .95rem; font-variant-numeric: tabular-nums; }
.ticker .track span { margin: 0 22px; } .ticker .up { color: var(--teal); } .ticker .hot { color: var(--gold); } .ticker .dim { color: var(--lav); }
@keyframes marquee { to { transform: translateX(-50%); } }

/* ===================================================== 4. TABS */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-family: var(--display); font-size: .95rem; font-weight: 700; padding: 10px 18px;
  border-radius: 12px 12px 0 0; color: var(--muted); transition: color .25s, background .25s; }
.stTabs [data-baseweb="tab"]:hover { color: #fff; background: rgba(255,46,136,.08); }
.stTabs [aria-selected="true"] { color: #fff !important; background: rgba(255,46,136,.12); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--sunset); background-size: 300% 100%; height: 3px;
  animation: sweep 3s linear infinite; transition: all .35s cubic-bezier(.2,.9,.3,1.2) !important; }
.stTabs [data-baseweb="tab-panel"] { animation: fadeUp .45s ease both; }

/* ===================================================== 5. RACE */
.section-title { font-family: var(--display); font-size: 1.6rem; font-weight: 700; margin: 6px 0 2px; }
.section-sub { color: var(--muted); margin-bottom: 14px; }
.label { font-family: 'Space Grotesk'; font-size: .78rem; letter-spacing: .2em; text-transform: uppercase; color: var(--muted); }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 22px; padding: 22px 24px; backdrop-filter: blur(12px); }
.chat { display: flex; gap: 12px; align-items: flex-start; margin: 10px 0 18px; animation: slideIn .5s cubic-bezier(.2,.9,.3,1.2) both; }
@keyframes slideIn { from { transform: translateX(-30px); opacity: 0; } to { transform: none; opacity: 1; } }
.avatar { width: 44px; height: 44px; border-radius: 50%; display: grid; place-items: center; font-size: 1.3rem; flex: none;
  background: var(--ocean); box-shadow: 0 0 18px rgba(25,227,209,.4); }
.bubble { background: rgba(28,12,48,.92); border: 1px solid rgba(25,227,209,.35); border-radius: 4px 20px 20px 20px; padding: 14px 18px;
  font-size: 1.15rem; max-width: 820px; backdrop-filter: blur(10px); }
.bubble .dots { display: inline-flex; gap: 4px; margin-right: 8px; vertical-align: middle; }
.bubble .dots i { width: 6px; height: 6px; border-radius: 50%; background: var(--teal); animation: blink 1.2s infinite; }
.bubble .dots i:nth-child(2) { animation-delay: .2s; } .bubble .dots i:nth-child(3) { animation-delay: .4s; }
@keyframes blink { 0%,80%,100% { opacity: .2; transform: translateY(0); } 40% { opacity: 1; transform: translateY(-3px); } }
.bubble .meta { color: var(--muted); font-size: .86rem; margin-top: 6px; font-variant-numeric: tabular-nums; }

.race { position: relative; border-radius: 24px; padding: 22px 26px 24px; min-height: 230px; overflow: hidden;
  background: rgba(18,6,31,.82); border: 1px solid var(--line); backdrop-filter: blur(14px);
  transition: transform .3s ease, box-shadow .3s ease; }
.race:hover { transform: translateY(-3px) rotate(-.3deg); }
.race.drex { border: 1.5px solid rgba(255,46,136,.75);
  background: linear-gradient(160deg, rgba(255,46,136,.22), rgba(255,138,61,.08) 55%, rgba(18,6,31,.9)); }
.race.llm { filter: saturate(.75); }
.race.drex.first { animation: flash .5s ease, scalePop .6s cubic-bezier(.2,1.6,.4,1), ring 1.8s ease-out .2s 3; }
@keyframes flash { 0% { background-color: rgba(255,255,255,.35); } 100% { background-color: transparent; } }
@keyframes scalePop { 0% { transform: scale(1); } 45% { transform: scale(1.045); } 100% { transform: scale(1); } }
@keyframes ring { 0% { box-shadow: 0 0 0 0 rgba(255,46,136,.75), 0 0 40px rgba(255,46,136,.35); }
                  100% { box-shadow: 0 0 0 30px rgba(255,46,136,0), 0 0 40px rgba(255,46,136,.35); } }
.race.drex.done { box-shadow: 0 0 40px rgba(255,46,136,.3); }
.race .who { display: flex; justify-content: space-between; align-items: center; }
.race .name { font-family: var(--display); font-weight: 700; letter-spacing: .14em; font-size: .85rem; }
.race.drex .name { color: #FF8FC0; } .race.llm .name { color: var(--lav); }
.badge { font-family: 'Space Grotesk'; font-size: .74rem; font-weight: 700; letter-spacing: .1em; padding: 5px 11px; border-radius: 999px;
  background: var(--sunset); color: #1B0B33; animation: dropIn .6s cubic-bezier(.2,1.6,.4,1) .15s both; }
@keyframes dropIn { from { transform: translateY(-26px) rotate(-8deg); opacity: 0; } to { transform: none; opacity: 1; } }
.action { position: relative; display: inline-block; font-family: var(--display); font-weight: 800; font-size: 3.6rem; line-height: 1;
  margin: 20px 0 12px; letter-spacing: -.02em; animation: slam .55s cubic-bezier(.2,1.5,.4,1) both; }
@keyframes slam { 0% { transform: scale(2.2); opacity: 0; filter: blur(8px); } 60% { transform: scale(.94); opacity: 1; filter: none; } 100% { transform: none; } }
.action.glitch::before, .action.glitch::after { content: attr(data-text); position: absolute; inset: 0; pointer-events: none; }
.action.glitch::before { color: var(--teal); mix-blend-mode: screen; animation: gl1 .9s steps(2) .1s 2 both; }
.action.glitch::after { color: var(--pink); mix-blend-mode: screen; animation: gl2 .9s steps(2) .1s 2 both; }
@keyframes gl1 { 0% { transform: translate(-4px,2px); clip-path: inset(0 0 60% 0); } 50% { transform: translate(3px,-2px); clip-path: inset(40% 0 20% 0); }
                 100% { transform: none; clip-path: inset(0 0 100% 0); } }
@keyframes gl2 { 0% { transform: translate(4px,-2px); clip-path: inset(50% 0 0 0); } 50% { transform: translate(-3px,2px); clip-path: inset(10% 0 60% 0); }
                 100% { transform: none; clip-path: inset(100% 0 0 0); } }
.timer { display: inline-block; font-family: 'Space Grotesk'; font-size: 1.7rem; font-weight: 700; font-variant-numeric: tabular-nums;
  padding: 4px 12px; border-radius: 10px; background: rgba(0,0,0,.45); border: 1px solid rgba(255,255,255,.08); }
.race.drex .timer { color: #FFD9E8; text-shadow: 0 0 12px rgba(255,46,136,.8); }
.race.llm .timer { color: #D9D2E6; }
.timer small { font-size: .85rem; color: var(--muted); font-weight: 500; margin-left: 10px; text-shadow: none; }
.pill { display: inline-block; padding: 5px 12px; border-radius: 999px; font-size: .84rem; font-weight: 700; animation: popIn .5s ease .25s both; }
.ok { background: rgba(25,227,209,.14); color: #7FF5EA; border: 1px solid rgba(25,227,209,.45); }
.bad { background: rgba(255,77,109,.16); color: #FFB3C1; border: 1px solid rgba(255,77,109,.5); }
.progress { height: 6px; border-radius: 6px; margin: 16px 0 18px; overflow: hidden; background: rgba(255,255,255,.06); }
.progress i { display: block; height: 100%; width: 100%;
  background: repeating-linear-gradient(45deg, rgba(255,255,255,.0) 0 10px, rgba(255,255,255,.25) 10px 20px), var(--sunset);
  background-size: 28px 28px, 300% 100%; animation: stripes .6s linear infinite, sweep 2s linear infinite; }
.race.llm .progress i { background: repeating-linear-gradient(45deg, transparent 0 10px, rgba(255,255,255,.18) 10px 20px), #6E6582;
  background-size: 28px 28px, 100% 100%; animation: stripes 1.4s linear infinite; }
@keyframes stripes { to { background-position: 28px 0, 0 0; } }
.scanner { position: absolute; left: 0; right: 0; top: 0; height: 40%; pointer-events: none;
  background: linear-gradient(to bottom, transparent, rgba(255,46,136,.28), transparent); animation: scanY .9s ease-in-out infinite alternate; }
@keyframes scanY { from { transform: translateY(-40%); } to { transform: translateY(220%); } }
.thinking { font-family: var(--display); font-size: 2.3rem; font-weight: 800;
  background: linear-gradient(90deg, #4B3E5E 0%, #F3E8FF 50%, #4B3E5E 100%); background-size: 200% 100%;
  -webkit-background-clip: text; background-clip: text; color: transparent; animation: shimmer 1.2s linear infinite; }
@keyframes shimmer { from { background-position: 200% 0; } to { background-position: -200% 0; } }
.idle { font-family: var(--display); font-size: 2.3rem; color: rgba(255,255,255,.18); margin: 24px 0 14px; font-weight: 800; }

.banner { position: relative; overflow: hidden; margin: 22px 0 10px; border-radius: 24px; padding: 28px; text-align: center; color: #1B0B33;
  background: var(--sunset); background-size: 300% 100%; box-shadow: 0 20px 70px rgba(255,46,136,.45);
  animation: burst .7s cubic-bezier(.2,1.5,.4,1) both, sweep 5s linear infinite; }
@keyframes burst { 0% { transform: scale(.6); opacity: 0; } 70% { transform: scale(1.03); opacity: 1; } 100% { transform: none; } }
.banner::after { content: ""; position: absolute; inset: 0; transform: translateX(-120%);
  background: linear-gradient(100deg, transparent 35%, rgba(255,255,255,.65) 50%, transparent 65%); animation: shine 3s ease-in-out .5s infinite; }
.banner .big { position: relative; z-index: 1; font-family: var(--display); font-size: 2.6rem; font-weight: 800; letter-spacing: -.02em; }
.banner small { position: relative; z-index: 1; display: block; font-size: 1.05rem; font-weight: 600; margin-top: 6px; font-variant-numeric: tabular-nums; }
.spark { position: absolute; width: 8px; height: 8px; border-radius: 2px; z-index: 1; opacity: 0; animation: sparkle 1.4s ease-out both; }
@keyframes sparkle { 0% { transform: translate(0,0) rotate(0) scale(.4); opacity: 1; }
                     100% { transform: translate(var(--dx), var(--dy)) rotate(260deg) scale(1); opacity: 0; } }
.scam { position: relative; overflow: hidden; margin-top: 12px; border-radius: 16px; padding: 16px 18px 16px 22px; font-weight: 700; font-size: 1.05rem;
  background: rgba(255,77,109,.16); border: 1.5px solid rgba(255,77,109,.7); color: #FFE1E7; animation: shake .5s ease both; }
.scam::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 10px;
  background: repeating-linear-gradient(45deg, #FF4D6D 0 8px, #FFC857 8px 16px); background-size: 22px 22px; animation: stripes .8s linear infinite; }
@keyframes shake { 0%,100% { transform: none; } 20% { transform: translateX(-7px); } 40% { transform: translateX(6px); } 60% { transform: translateX(-4px); } 80% { transform: translateX(2px); } }
.policy { border-radius: 18px; padding: 14px 18px; background: var(--panel); border: 1px solid var(--line); backdrop-filter: blur(10px); }

/* ===================================================== 6. RESULTS */
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 8px 0 18px; perspective: 900px; }
.kpi { background: var(--panel); border: 1px solid var(--line); border-radius: 22px; padding: 18px 20px; backdrop-filter: blur(12px);
  animation: flipIn .7s cubic-bezier(.2,1.2,.4,1) both; transition: transform .3s; }
.kpi:hover { transform: translateY(-4px); }
.kpi:nth-child(1) { animation-delay: .05s; } .kpi:nth-child(2) { animation-delay: .2s; } .kpi:nth-child(3) { animation-delay: .35s; } .kpi:nth-child(4) { animation-delay: .5s; }
@keyframes flipIn { from { transform: rotateX(-70deg) translateY(20px); opacity: 0; } to { transform: none; opacity: 1; } }
.kpi.hot { background: linear-gradient(160deg, rgba(255,46,136,.25), rgba(255,138,61,.08) 60%, rgba(18,6,31,.85)); border-color: rgba(255,46,136,.6);
  box-shadow: 0 0 30px rgba(255,46,136,.18); }
.kpi .v { font-family: var(--display); font-size: 2.6rem; font-weight: 800; line-height: 1.05; font-variant-numeric: tabular-nums;
  display: inline-block; animation: countUp 1.2s cubic-bezier(.2,.9,.3,1) .3s both; }
@keyframes countUp { 0% { transform: translateY(60%); opacity: 0; filter: blur(10px); } 60% { filter: blur(0); } 100% { transform: none; opacity: 1; } }
.kpi.hot .v { background: var(--sunset); background-size: 300% 100%; -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: countUp 1.2s cubic-bezier(.2,.9,.3,1) .3s both, sweep 6s linear infinite; }
.kpi .s { color: var(--muted); font-size: .92rem; margin-top: 6px; }
.callout { border-radius: 18px; padding: 16px 20px; background: rgba(18,6,31,.9); border: 1px solid rgba(255,200,87,.5); }
.callout b { color: var(--gold); }
.sample { position: relative; z-index: 1; background: rgba(40,20,0,.85); color: #FFE7A8; border: 1.5px solid rgba(255,200,87,.7);
  border-radius: 14px; padding: 12px 16px; font-weight: 700; }
.timeline { display: flex; align-items: center; gap: 0; margin: 6px 0 16px; padding: 18px 20px; border-radius: 20px;
  background: var(--panel); border: 1px solid var(--line); backdrop-filter: blur(10px); }
.tl-node { flex: 1; text-align: center; position: relative; animation: fadeUp .6s ease both; }
.tl-node:nth-child(2) { animation-delay: .2s; } .tl-node:nth-child(3) { animation-delay: .4s; } .tl-node:nth-child(4) { animation-delay: .6s; }
.tl-dot { width: 18px; height: 18px; border-radius: 50%; margin: 0 auto 8px; background: var(--pink);
  box-shadow: 0 0 0 4px rgba(255,46,136,.2), 0 0 20px rgba(255,46,136,.8); animation: nodePulse 2.4s ease-in-out infinite; }
.tl-node.best .tl-dot { background: var(--gold); box-shadow: 0 0 0 4px rgba(255,200,87,.25), 0 0 24px rgba(255,200,87,.9); }
@keyframes nodePulse { 50% { transform: scale(1.25); } }
.tl-node .r { font-family: 'Space Grotesk'; font-size: .78rem; letter-spacing: .14em; color: var(--muted); text-transform: uppercase; }
.tl-node .a { font-family: var(--display); font-size: 1.5rem; font-weight: 800; font-variant-numeric: tabular-nums; }
.tl-node .d { font-size: .82rem; color: var(--muted); }
.tl-line { flex: 0 0 40px; height: 2px; background: var(--sunset); background-size: 300% 100%; animation: sweep 3s linear infinite; }

/* ===================================================== 7. SELL */
.pricetag { display: inline-flex; align-items: center; gap: 16px; padding: 18px 26px 18px 34px; border-radius: 22px 22px 22px 22px;
  background: linear-gradient(140deg, rgba(255,46,136,.35), rgba(255,138,61,.18)); border: 1.5px solid rgba(255,46,136,.7);
  position: relative; transform-origin: 30px 0; animation: swing 1.6s cubic-bezier(.2,1.6,.4,1) both; box-shadow: 0 14px 40px rgba(255,46,136,.3); }
.pricetag::before { content: ""; position: absolute; left: 14px; top: 50%; width: 10px; height: 10px; margin-top: -5px; border-radius: 50%;
  background: #12061F; border: 2px solid rgba(255,255,255,.6); }
@keyframes swing { 0% { transform: rotate(-8deg); opacity: 0; } 20% { opacity: 1; } 45% { transform: rotate(5deg); } 70% { transform: rotate(-2deg); } 100% { transform: none; } }
.pricetag .p { font-family: var(--display); font-size: 3.4rem; font-weight: 800; line-height: 1; }
.pricetag .m { color: #F3E6FA; font-variant-numeric: tabular-nums; }
.listing { border-radius: 22px; overflow: hidden; border: 1px solid var(--line); background: rgba(18,6,31,.92);
  animation: slideUp .7s cubic-bezier(.2,1.2,.4,1) .3s both; box-shadow: 0 24px 60px rgba(0,0,0,.45), 0 0 50px rgba(255,46,136,.12); }
@keyframes slideUp { from { transform: translateY(40px); opacity: 0; } to { transform: none; opacity: 1; } }
.listing .img { height: 180px; display: grid; place-items: center; font-size: 4rem;
  background: radial-gradient(circle at 50% 40%, rgba(255,138,61,.45), transparent 60%), radial-gradient(circle at 20% 80%, rgba(25,227,209,.25), transparent 50%), #1B0B33; }
.listing .body { padding: 16px 20px 20px; }
.listing .t { font-weight: 700; font-size: 1.08rem; }
.listing .pr { font-family: var(--display); font-size: 1.9rem; font-weight: 800; margin: 6px 0 8px; color: var(--gold); }
.listing .d { color: #E2D6EE; white-space: pre-line; line-height: 1.5; }
.listing .src { color: var(--muted); font-size: .8rem; margin-top: 10px; }

/* ===================================================== 8. MICRO-INTERACTIONS */
.stButton > button[kind="primary"] { background: var(--sunset); background-size: 300% 100%; border: 0; border-radius: 16px;
  font-family: var(--display); font-weight: 700; font-size: 1rem; letter-spacing: .02em; color: #1B0B33; padding: .8rem 1rem;
  box-shadow: 0 10px 34px rgba(255,46,136,.45); animation: sweep 6s linear infinite; transition: transform .2s, box-shadow .2s; }
.stButton > button[kind="primary"]:hover { transform: translateY(-2px) scale(1.01); box-shadow: 0 16px 44px rgba(255,46,136,.65); color: #1B0B33; }
.stButton > button[kind="primary"] p { color: #1B0B33; font-weight: 800; }
input:focus, textarea:focus, [data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within, [data-baseweb="select"]:focus-within {
  box-shadow: 0 0 0 2px var(--teal) !important; border-radius: 10px; }
[data-testid="stFileUploaderDropzone"] { background: rgba(18,6,31,.8); border: 1.5px dashed rgba(25,227,209,.55); border-radius: 20px; min-height: 150px;
  transition: border-color .3s, box-shadow .3s; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--teal); box-shadow: 0 0 30px rgba(25,227,209,.25); }
[data-testid="stExpander"], [data-testid="stDataFrame"] { background: rgba(18,6,31,.78); border-radius: 14px; }

@media (max-width: 900px) { .hero { grid-template-columns: 1fr; } .kpis { grid-template-columns: repeat(2, 1fr); }
  .hero h1 { font-size: 2.8rem; } .action { font-size: 2.6rem; } }

/* ===================================================== 9. REDUCED MOTION */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition: none !important; }
  .grid-floor, .scan, .orb, .ticker .track { animation: none !important; }
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------- THEME SWITCH
# "mono" = Levi's Figma design (black/white, sharp, blackletter logo), layered on top of the
# neon CSS above. Set to "neon" to get the neon-dusk look back. Presentation only.
UI_THEME = "mono"
CHART = {"pick": "#FF2E88", "unpick": "#3A2A4F", "models": ["#FF2E88", "#FF8A3D", "#A59BB8"]}
if UI_THEME == "mono":
    from ui_theme import MONO_CHART, MONO_CSS
    st.markdown(MONO_CSS, unsafe_allow_html=True)
    CHART = MONO_CHART

# Background layers (decorative, fixed, behind content, never intercept clicks)
st.markdown('<div class="bg-layer"><div class="orb o1"></div><div class="orb o2"></div><div class="orb o3"></div>'
            '<div class="sun"></div><div class="grid-floor"></div><div class="scan"></div></div>',
            unsafe_allow_html=True)


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
        color=alt.condition("datum.picked", alt.value(CHART["pick"]), alt.value(CHART["unpick"])),
        tooltip=["option", alt.Tooltip("probability:Q", format=".1%")],
    )
    text = alt.Chart(df).mark_text(align="left", dx=6, color="#ECECF1", font="Inter", fontSize=12).encode(
        y=alt.Y("option:N", sort=list(probs.keys())), x="probability:Q", text=alt.Text("probability:Q", format=".0%"))
    return _dark((bars + text).properties(height=height))


MODEL_COLORS = alt.Scale(domain=[MODEL_LABELS["drex"], MODEL_LABELS["drex2"], MODEL_LABELS["llm"]],
                         range=CHART["models"])


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


def _recorded_pair():
    """A random hard-set item where both Drex Max and Claude have a recorded (cached, real) answer."""
    path = RESULTS_DIR / "decisions_full.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path, keep_default_na=False)
    d = d[(d["dataset"] == "hard") & (d["repeat"].astype(int) == 0) & (d["error"] == "")]
    both = set(d[d["model"] == "drex"]["item_id"]) & set(d[d["model"] == "llm"]["item_id"])
    if not both:
        return None
    hard = pd.read_csv(DATA_DIR / "negotiation_hard.csv").set_index("id")
    item = random.choice(sorted(both))
    row = hard.loc[item]

    def as_result(model):
        r = d[(d["model"] == model) & (d["item_id"] == item)].iloc[0]
        return {"answer": r["chosen"] or None, "probs": json.loads(r["probs_json"]) if r["probs_json"] and model == "drex" else None,
                "latency_ms": float(r["latency_ms"]), "cost_usd": float(r["cost_usd"]),
                "invalid": str(r["invalid"]).lower() == "true", "error": None, "model": model}
    return {"id": item, "category": row["category"], "message": str(row["message"]), "item": str(row["item"]),
            "offer": float(row["offer"]), "list": float(row["list_price"]), "floor": float(row["floor_price"]),
            "expected": row["expected_action"], "is_scam": int(row["is_scam"]),
            "drex": as_result("drex"), "llm": as_result("llm")}


def _sparks(n: int = 22) -> str:
    """Pure-CSS confetti: small squares flying out from the banner centre (deterministic positions)."""
    colors = ["#FFFFFF", "#19E3D1", "#FFC857", "#12061F", "#FF2E88"]
    out = []
    for i in range(n):
        angle = i * 360 / n
        import math
        dx, dy = 260 * math.cos(math.radians(angle)), 90 * math.sin(math.radians(angle))
        out.append(f'<span class="spark" style="left:50%;top:50%;background:{colors[i % len(colors)]};'
                   f'--dx:{dx:.0f}px;--dy:{dy:.0f}px;animation-delay:{0.35 + (i % 5) * 0.06:.2f}s"></span>')
    return "".join(out)


# ------------------------------------------------------------- SIDEBAR
with st.sidebar:
    st.markdown("### 🧥 ThriftAgent")
    st.caption("Reselling is hundreds of tiny decisions. An LLM writes the text; **Drex makes the decisions.**")
    if USE_MOCK:
        st.warning("**MOCK MODE** - answers and timings are simulated. Set `USE_MOCK = False` in app.py "
                   "once the Drex + LLM clients work.")
    else:
        st.success("**LIVE MODE** - real Drex + frontier LLM calls.")
    engine_label = st.radio("Drex engine", ["Drex Max (Nace API)", "Drex 2.0 open weights (DeepInfra)"],
                            help="Switch if one provider is down. The active engine is named on every result.")
    DREX_ENGINE["engine"] = "drex2" if engine_label.startswith("Drex 2.0") else "drex"
    if st.session_state.get("history"):
        hist = st.session_state["history"]
        drex_total = sum(h["drex_ms"] for h in hist)
        llm_total = sum(h["llm_ms"] for h in hist)
        st.metric("Decisions this session", len(hist))
        st.metric("Avg speedup", f"{llm_total / max(drex_total, 1):.1f}×")

# ------------------------------------------------------------- HERO
_h = _headline()
_letters = "".join(f'<span class="ltr" style="animation-delay:{0.15 + i * 0.05:.2f}s, 0s">{c}</span>'
                   for i, c in enumerate("ThriftAgent"))
st.markdown(f"""
<div class="hero">
  <div>
    <div class="eyebrow">Drexathon · Drex vs the Giants</div>
    <h1>{_letters}</h1>
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

def _ticker_items():
    items = [f'<span class="hot">⚡ DREX {_h["speed"]} FASTER</span>', f'<span class="hot">💸 {_h["cost"]} CHEAPER</span>',
             f'<span class="up">🎯 {_h["acc"]} ON 80 PRE-REGISTERED HARD MESSAGES</span>']
    hs_ = _hard_summary()
    try:
        hard_ = hs_[hs_["dataset"] == "hard"].set_index("model")
        items += [f'<span class="up">DREX {hard_.loc["drex", "median_latency_ms"]:,.0f} MS ▲</span>',
                  f'<span class="dim">CLAUDE {hard_.loc["llm", "median_latency_ms"]:,.0f} MS</span>']
    except Exception:
        pass
    try:
        sc = json.loads((RESULTS_DIR / "scale1000.json").read_text())["drex"]
        items.append(f'<span class="hot">1,000 DECISIONS IN {sc["seconds"]:.0f}S FOR ${sc["total_cost_usd"]:.3f}</span>')
    except Exception:
        pass
    items.append('<span class="dim">PRICING: THE LLM STILL WINS · ROUTED TO THE LLM</span>')
    return " · ".join(items)


_t = _ticker_items()
st.markdown(f'<div class="ticker"><div class="track">{_t} · {_t} · </div></div>', unsafe_allow_html=True)

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
      <div class="bubble"><span class="dots"><i></i><i></i><i></i></span>{_html.escape(str(message))}
        <div class="meta">Offer <b>${offer:,.2f}</b> on <b>{_html.escape(str(item))}</b> · listed ${list_price:,.0f}
        · floor ${floor_price:,.0f} · {100 * offer / list_price:.1f}% of list</div></div></div>
    """, unsafe_allow_html=True)

    DREX_NAME = "Drex 2.0" if DREX_ENGINE["engine"] == "drex2" else "Drex Max"
    bcol1, bcol2 = st.columns([3, 1.2])
    go = bcol1.button("⚡ Send to Drex and the LLM", type="primary", use_container_width=True)
    replay = bcol2.button("▶ Replay a recorded run", use_container_width=True,
                          help="Plays back REAL cached responses from the benchmark, at their recorded speed.")

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
            scanner = '<div class="scanner"></div>' if is_drex else ""
            slot.markdown(f'<div class="{cls}">{scanner}{head}<div class="progress"><i></i></div>'
                          f'<div class="thinking">deciding…</div>'
                          f'<div class="timer" style="margin-top:14px">⏱ live</div></div>', unsafe_allow_html=True)
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
            word = ACTION_WORD.get(ans, str(ans).upper())
            body = (f'<div class="action glitch" data-text="{word}" style="color:{ACTION_COLOR.get(ans, "#FFF4F8")};'
                    f'text-shadow:0 0 28px {ACTION_COLOR.get(ans, "#FFF4F8")}66">{word}</div><br>{pill}')
            if is_drex:
                cls += " done"
        slot.markdown(
            f'<div class="{cls}">{head}{body}'
            f'<div class="timer" style="margin-top:14px">{(result["latency_ms"] or 0):,.0f} ms'
            f'<small>${result["cost_usd"]:.6f} per decision</small></div></div>', unsafe_allow_html=True)

    rec = _recorded_pair() if replay else None
    if replay and rec is None:
        st.warning("No recorded runs found (bench/results/decisions_full.csv).")
    if rec is not None:
        # REPLAY: real cached API responses from the benchmark, played back at their recorded latency.
        expected = rec["expected"]
        banner_slot.markdown(
            f'<div class="sample">▶ REPLAY of a recorded benchmark run · {rec["id"]} ({rec["category"].replace("_", " ")}) · '
            f'real Drex Max + Claude responses captured earlier today, shown at their real speed.<br>'
            f'<span style="font-weight:500">💬 “{_html.escape(rec["message"])}” · offer ${rec["offer"]:g} on '
            f'{_html.escape(rec["item"])} (list ${rec["list"]:g}, floor ${rec["floor"]:g})</span></div>',
            unsafe_allow_html=True)
        render_card(slot_drex, "Drex Max · replay", None, True)
        render_card(slot_llm, "Claude Opus 5.5 · replay", None, False)
        d_rec, l_rec = rec["drex"], rec["llm"]
        time.sleep(d_rec["latency_ms"] / 1000)
        render_card(slot_drex, "Drex Max · replay", d_rec, True, first=not d_rec["invalid"])
        time.sleep(max(0.0, (l_rec["latency_ms"] - d_rec["latency_ms"]) / 1000))
        render_card(slot_llm, "Claude Opus 5.5 · replay", l_rec, False)
        st.session_state["neg"] = {"results": {"drex": d_rec, "llm": l_rec}, "expected": rec["expected"],
                                   "signals": ["recorded scam"] if rec["is_scam"] else [], "first": "drex",
                                   "replay": rec}
    elif go:
        st.session_state.pop("neg", None)
        render_card(slot_drex, DREX_NAME, None, True)
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
                    render_card(slot_drex, DREX_NAME, results[name], True,
                                    first=(order[0] == "drex" and not results[name]["invalid"]))
                else:
                    render_card(slot_llm, "Frontier LLM", results[name], False, first=False)
        st.session_state["neg"] = {"results": results, "expected": expected,
                                   "signals": scam_signals(message),
                                   "first": order[0] if order and not results[order[0]]["invalid"] else None}
        d_ms, l_ms = results["drex"]["latency_ms"], results["llm"]["latency_ms"]
        if d_ms and l_ms and not (results["drex"]["invalid"] or results["llm"]["invalid"]):
            st.session_state.setdefault("history", []).append({"drex_ms": d_ms, "llm_ms": l_ms})
    elif st.session_state.get("neg"):
        res = st.session_state["neg"]["results"]
        is_replay = "replay" in st.session_state["neg"]
        if is_replay:
            expected = st.session_state["neg"]["expected"]
        render_card(slot_drex, "Drex Max · replay" if is_replay else DREX_NAME, res["drex"], True,
                    first=st.session_state["neg"].get("first") == "drex")
        render_card(slot_llm, "Claude Opus 5.5 · replay" if is_replay else "Frontier LLM", res["llm"], False)
    else:
        idle = {"answer": "—", "latency_ms": 0, "cost_usd": 0, "invalid": False, "error": None, "probs": None}
        render_card(slot_drex, DREX_NAME, idle, True)
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
                f'<div class="banner">{_sparks()}<div class="big">⚡ Drex was {speedup:.1f}× faster{extra}</div>'
                f'<small>{d["latency_ms"]:,.0f} ms vs {l["latency_ms"]:,.0f} ms on the exact same decision'
                f'{" · REPLAY of recorded real responses" if "replay" in neg else ""}</small></div>',
                unsafe_allow_html=True)
        if "replay" in neg:
            rp = neg["replay"]
            st.markdown(f'<div class="chat"><div class="avatar">🛍️</div><div class="bubble">'
                        f'<span class="label" style="color:var(--gold)">Replayed message · {rp["id"]} · '
                        f'{rp["category"].replace("_", " ")}</span><br>{_html.escape(rp["message"])}'
                        f'<div class="meta">Offer <b>${rp["offer"]:g}</b> on <b>{_html.escape(rp["item"])}</b> · '
                        f'list ${rp["list"]:g} · floor ${rp["floor"]:g} · recorded real API responses</div></div></div>',
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
        show["model"] = show["model"].map(MODEL_LABELS).fillna(show["model"])
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
            nodes = []
            for n in existing:
                neg_acc = per_round[n].get(("negotiation", "drex"))
                pr_acc = per_round[n].get(("pricing", "drex"))
                if neg_acc is not None:
                    nodes.append((n.split(" · ")[0], neg_acc, f"pricing {pr_acc:.0%}" if pr_acc is not None else ""))
            if hs is not None and "hard" in set(hs["dataset"]):
                try:
                    nodes.append(("Hard set", float(hs[(hs["dataset"] == "hard") & (hs["model"] == "drex")]["accuracy"].iloc[0]),
                                  "80 pre-registered"))
                except IndexError:
                    pass
            if nodes:
                best = max(a for _, a, _ in nodes)
                parts = []
                for i, (r, a, d) in enumerate(nodes):
                    if i:
                        parts.append('<div class="tl-line"></div>')
                    parts.append(f'<div class="tl-node{" best" if a == best else ""}"><div class="tl-dot"></div>'
                                 f'<div class="r">{r}</div><div class="a">{a:.0%}</div><div class="d">{d}</div></div>')
                st.markdown('<div class="label" style="margin-top:4px">Drex · negotiation accuracy by round</div>'
                            f'<div class="timeline">{"".join(parts)}</div>', unsafe_allow_html=True)
            with st.expander("📈 Accuracy round by round", expanded=True):
                short = {name: name.split(" · ")[0].lower() for name in existing}
                table = pd.DataFrame([{"benchmark": k[0], "model": MODEL_LABELS.get(k[1], k[1]),
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

    # Readable model names for the table and charts (headline above still uses the short keys).
    df["model"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
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
