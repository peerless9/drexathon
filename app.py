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
    for q in (0.25, 0.40, 0.50, 0.60, 0.75):
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


def load_results():
    """Returns (rows, is_sample, note)."""
    rows, is_sample, note = [], True, "No bench/results/summary.json yet."
    if SUMMARY_PATH.exists():
        try:
            summary = json.loads(SUMMARY_PATH.read_text())
            for bench, info in summary.get("benchmarks", {}).items():
                for model, m in (info.get("models") or {}).items():
                    if "skipped" in m or m.get("accuracy") is None:
                        continue
                    rows.append({"benchmark": bench, "model": model, **m,
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
st.set_page_config(page_title="ThriftAgent · Drex vs the Giants", page_icon="🧥", layout="wide")

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2rem; max-width: 1200px; }}
  .hero h1 {{ font-size: 2.4rem; margin-bottom: 0; }}
  .hero p {{ color: #6b7280; font-size: 1.05rem; margin-top: .25rem; }}
  .card {{ border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 18px 20px; height: 100%; }}
  .card.drex {{ border: 2px solid {DREX_COLOR}; }}
  .label {{ font-size: .8rem; letter-spacing: .08em; text-transform: uppercase; color: #6b7280; }}
  .answer {{ font-size: 2.2rem; font-weight: 800; margin: .2rem 0; }}
  .latency {{ font-size: 1.1rem; font-variant-numeric: tabular-nums; }}
  .banner {{ background: linear-gradient(90deg, {DREX_COLOR}, #B14BFF); color: white; border-radius: 16px;
             padding: 22px; text-align: center; font-size: 2rem; font-weight: 800; margin: 18px 0 6px; }}
  .banner small {{ display:block; font-size: 1rem; font-weight: 500; opacity: .9; }}
  .pill {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .8rem; font-weight: 600; }}
  .ok {{ background:#DCFCE7; color:#166534; }} .bad {{ background:#FEE2E2; color:#991B1B; }}
  .sample {{ background:#FEF3C7; color:#92400E; border-radius:10px; padding:10px 14px; font-weight:700; }}
</style>
""", unsafe_allow_html=True)

ACTION_EMOJI = {"accept": "✅ ACCEPT", "counter": "🔁 COUNTER", "decline": "⛔ DECLINE"}


def prob_chart(probs: dict, pick: str, height: int = 220):
    df = pd.DataFrame({"option": list(probs.keys()), "probability": list(probs.values())})
    df["picked"] = df["option"] == pick
    return alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("option:N", sort=list(probs.keys()), title=None),
        x=alt.X("probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title=None),
        color=alt.condition("datum.picked", alt.value(DREX_COLOR), alt.value("#D1D5DB")),
        tooltip=["option", alt.Tooltip("probability:Q", format=".1%")],
    ).properties(height=height)


def metric_bar(df: pd.DataFrame, field: str, title: str, fmt: str):
    return alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X(f"{field}:Q", title=None, axis=alt.Axis(format=fmt)),
        y=alt.Y("model:N", title=None),
        row=alt.Row("benchmark:N", title=None, header=alt.Header(labelFontWeight="bold")),
        color=alt.Color("model:N", scale=alt.Scale(domain=["drex", "llm"], range=[DREX_COLOR, LLM_COLOR]),
                        legend=None),
        tooltip=["benchmark", "model", alt.Tooltip(f"{field}:Q", format=fmt)],
    ).properties(title=title, height=60, width=300)


# ============================================================== SIDEBAR
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

st.markdown('<div class="hero"><h1>🧥 ThriftAgent</h1>'
            '<p>An AI agent that sells your used clothes — priced and negotiated by <b>Drex</b>, '
            'benchmarked against a frontier LLM.</p></div>', unsafe_allow_html=True)

tab_sell, tab_negotiate, tab_results = st.tabs(["🏷️ Sell", "🤝 Negotiate", "📊 Results"])

# ============================================================== SELL TAB
with tab_sell:
    pricing_df, pricing_is_sample = load_csv("pricing")
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        st.subheader("1 · Your item")
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
        category = c4.selectbox("Category", categories)
        price_clicked = st.button("💸 Price it with Drex", type="primary", use_container_width=True)

    with right:
        st.subheader("2 · Drex picks the price")
        if pricing_df is None:
            st.info("No pricing data yet. Fill in bench/data/pricing.csv.")
        else:
            if pricing_is_sample:
                st.caption("⚠️ Using SAMPLE comps - fill in bench/data/pricing.csv for real ones.")
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
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Drex's price", result["answer"])
                    m2.metric("Decided in", f"{result['latency_ms']:.0f} ms")
                    m3.metric("Comps used", sell["n_comps"])
                    if result["probs"]:
                        st.altair_chart(prob_chart(result["probs"], result["answer"]), use_container_width=True)
                    st.caption("Options = 25th / 40th / 50th / 60th / 75th percentile of recent sold comps.")

                    st.subheader("3 · Listing (written by the LLM)")
                    b, t, s, c, cat = sell["item"]
                    price = result["answer"].lstrip("$")
                    if "listing" not in sell:
                        with st.spinner("Writing listing…"):
                            sell["listing"] = generate_listing(b, t, s, c, cat, price)
                    l_title, l_desc, source = sell["listing"]
                    with st.container(border=True):
                        st.markdown(f"**{l_title}**  \n### ${price}")
                        st.write(l_desc)
                        st.caption(f"Text source: {source}")
            else:
                st.caption("Fill in the item and hit **Price it with Drex**.")

# ============================================================== NEGOTIATE TAB
with tab_negotiate:
    neg_df, neg_is_sample = load_csv("negotiation")
    st.subheader("Same decision. Two models. Who's faster?")

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
    c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1])
    item = c1.text_input("Item", default("item", "Patagonia Better Sweater Fleece (M)"), key=key + "item")
    list_price = c2.number_input("List $", min_value=1.0, value=float(default("list_price", 85)), key=key + "lp")
    floor_price = c3.number_input("Floor $", min_value=0.0, value=float(default("floor_price", 60)), key=key + "fp")
    days_listed = c4.number_input("Days listed", min_value=0, value=int(default("days_listed", 5)), key=key + "dl")
    c5, c6 = st.columns([3.2, 1])
    message = c5.text_area("💬 Buyer message", default("buyer_message", "hey would u take 70? can pay rn"),
                           height=90, key=key + "msg")
    offer = c6.number_input("Offer $", min_value=0.0, value=float(default("offer", 70)), key=key + "off")

    go = st.button("⚡ Send to Drex and the LLM", type="primary", use_container_width=True)

    facts = {
        "seller policy": SELLER_POLICY,
        "item": item,
        "list price": f"${list_price:g}",
        "floor price (lowest the seller will take)": f"${floor_price:g}",
        "days listed": days_listed,
        "buyer message": message,
        "offer": f"${offer:g}",
        # private hints for the mock only (stripped before real calls)
        "_list": list_price, "_floor": floor_price, "_offer": offer,
    }
    expected = policy_action(list_price, floor_price, offer, message)

    col_drex, col_llm = st.columns(2, gap="large")
    slot_drex, slot_llm = col_drex.empty(), col_llm.empty()
    banner_slot = st.empty()

    def render_card(slot, name, result, is_drex):
        cls = "card drex" if is_drex else "card"
        if result is None:
            slot.markdown(f'<div class="{cls}"><div class="label">{name}</div>'
                          f'<div class="answer">⏳ thinking…</div></div>', unsafe_allow_html=True)
            return
        if result["error"] or result["invalid"]:
            body = f'<div class="answer">⚠️ invalid</div><div>{result["error"] or "unparseable answer"}</div>'
        else:
            ok = result["answer"] == expected
            pill = '<span class="pill ok">matches policy</span>' if ok else '<span class="pill bad">wrong call</span>'
            body = (f'<div class="answer">{ACTION_EMOJI.get(result["answer"], result["answer"])}</div>{pill}')
        slot.markdown(
            f'<div class="{cls}"><div class="label">{name}</div>{body}'
            f'<div class="latency">⏱ <b>{(result["latency_ms"] or 0):,.0f} ms</b> · '
            f'${result["cost_usd"]:.4f}</div></div>', unsafe_allow_html=True)

    if go:
        render_card(slot_drex, "Drex", None, True)
        render_card(slot_llm, "Frontier LLM", None, False)
        results = {}
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
                if name == "drex":
                    render_card(slot_drex, "Drex", results[name], True)
                else:
                    render_card(slot_llm, "Frontier LLM", results[name], False)
        st.session_state["neg"] = {"results": results, "expected": expected,
                                   "signals": scam_signals(message)}
        d_ms, l_ms = results["drex"]["latency_ms"], results["llm"]["latency_ms"]
        if d_ms and l_ms:
            st.session_state.setdefault("history", []).append({"drex_ms": d_ms, "llm_ms": l_ms})
    elif st.session_state.get("neg"):
        res = st.session_state["neg"]["results"]
        render_card(slot_drex, "Drex", res["drex"], True)
        render_card(slot_llm, "Frontier LLM", res["llm"], False)
    else:
        render_card(slot_drex, "Drex", {"answer": "—", "latency_ms": 0, "cost_usd": 0, "invalid": False,
                                        "error": None, "probs": None}, True)
        render_card(slot_llm, "Frontier LLM", {"answer": "—", "latency_ms": 0, "cost_usd": 0, "invalid": False,
                                               "error": None, "probs": None}, False)

    neg = st.session_state.get("neg")
    if neg:
        d, l = neg["results"]["drex"], neg["results"]["llm"]
        if d["latency_ms"] and l["latency_ms"]:
            speedup = l["latency_ms"] / d["latency_ms"]
            cheaper = (l["cost_usd"] / d["cost_usd"]) if d["cost_usd"] else None
            extra = f" · {cheaper:.0f}× cheaper" if cheaper else ""
            banner_slot.markdown(
                f'<div class="banner">⚡ Drex was {speedup:.1f}× faster{extra}'
                f'<small>{d["latency_ms"]:,.0f} ms vs {l["latency_ms"]:,.0f} ms on the exact same decision</small></div>',
                unsafe_allow_html=True)
        i1, i2 = st.columns(2)
        with i1:
            if d["probs"]:
                st.markdown("**Drex's confidence**")
                st.altair_chart(prob_chart(d["probs"], d["answer"], height=140), use_container_width=True)
        with i2:
            st.markdown(f"**Seller policy says:** {ACTION_EMOJI[neg['expected']]}")
            if neg["signals"]:
                st.error("🚩 Scam signals in the message (off-platform payment / contact / own label).")
            st.caption("The LLM writes text and has no calibrated probabilities - Drex returns one per option.")

    if neg_is_sample and neg_df is not None:
        st.caption("⚠️ Scenarios are from the SAMPLE file.")

# ============================================================== RESULTS TAB
with tab_results:
    rows, is_sample, note = load_results()
    st.subheader("Benchmark: Drex vs the frontier LLM")
    if is_sample:
        st.markdown('<div class="sample">⚠️ SAMPLE NUMBERS - not real results. Run '
                    '<code>python bench/run_bench.py</code> to fill bench/results/summary.json.</div>',
                    unsafe_allow_html=True)
    st.caption(note)

    df = pd.DataFrame(rows)
    df["median_latency_ms"] = df["median_latency_s"] * 1000
    df["model"] = df["model"].astype(str)

    # headline numbers (averaged across benchmarks)
    agg = df.groupby("model")[["accuracy", "median_latency_ms", "cost_per_1000_usd"]].mean()
    if {"drex", "llm"} <= set(agg.index):
        h1, h2, h3 = st.columns(3)
        h1.metric("Accuracy (Drex)", f"{agg.loc['drex', 'accuracy']:.0%}",
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
