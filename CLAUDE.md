# drexathon: Project Rules for Claude

## Who we are / how to work
- Two beginners at a hackathon, each on our own Mac: Levi (GitHub: peerless9, repo owner) and svanaparthy22.
- Do as much as possible yourself (run commands, create files, install things). Only stop when you truly need a human: passwords, sign-ins, API keys, decisions. When a human must click something, give exact step-by-step clicks.
- Keep explanations short and beginner-friendly.
- ALWAYS `git pull` before starting a chunk of work, and commit + push after each working step with clear messages. Remind the human to push/pull.
- Never commit `.env` (API keys). It is in `.gitignore`.

## Event
- "Drexathon" - Nace AI's Decision Model Hackathon x Adobe EDU, Frontier Tower SF, Sat Nov 14 2026.
- Build starts 11:00 AM. HARD submission deadline 3:00 PM (we submit by 2:45). Finalist demos 3:15 (5-minute limit).
- Track: "Apex Predator: Drex vs. the Giants" - take decisions normally handled by a frontier LLM, give them to Drex, and prove it's faster/cheaper/more accurate. Judged on MEASURED results and the size of the improvement. **The benchmark IS the product; the app is the showcase.**
- Bonus points: using NDI (Nace's document parser) for real-world documents. Separate "Best Use of Adobe" prize (Firefly/Express/Photoshop in project or demo).

## About Drex (from nace.ai/drex)
- Drex is a decision model, NOT a text generator. You send case facts + typed questions; it returns a probability for each option in one pass. It can only answer with options you supply.
- Question types: choice among named options, yes/no, or a score on a scale. Several questions can share one pass.
- Models: Drex Max v1.1 (`drex-max`, most accurate, via Nace Console and OpenRouter), Drex 2.0 (open weights), Drex 1.5 (light).
- Docs: console.nace.ai/docs. **Don't guess the API schema** - Levi pastes the exact request format; `agent/drex_client.py` is a stub until then.
- Architecture: an LLM writes text (listings, chat replies); Drex makes the decisions.

## Project: AI agent that sells your used clothes
Story: "Reselling = hundreds of small decisions (what price, accept an offer, is this buyer a scammer). We gave them to a frontier LLM and to Drex, measured both, and Drex won by __."

### Benchmark 1 - Pricing (`bench/data/pricing.csv`, ~40 rows)
Collected by hand from eBay SOLD listings, 4 categories x 10 items.
Columns: `id, category, brand, title, size, condition, sold_price, url`
- For each item, the other items in its category are the "comps" (leave-one-out; the model never sees the item's own sold_price).
- Build 5 price options from comps (25th/40th/50th/60th/75th percentiles, rounded). Both models get the SAME facts + SAME options.
- Correct = option closest to the actual sold_price.

### Benchmark 2 - Negotiation (`bench/data/negotiation.csv`, ~30 rows)
Columns: `id, item, list_price, floor_price, days_listed, buyer_message, offer, correct_action`
- Policy: accept if offer >= 90% of list; counter if between floor and 90%; decline if below floor OR scam ("pay on Zelle", "text me at...", "I'll send my own shipping label"). ~6 scam rows.
- Options: accept / counter / decline (+ optional follow-up "counter at?" with 3-4 price options).

### Metrics (both models)
Accuracy, median latency (skip the first warm-up call), cost per 1,000 decisions, invalid/unparseable answers. Saved to `bench/results/summary.json`.
Fairness matters: same prompt wording, same options, strongest frontier model we have keys for.

### App (`app.py`, Streamlit, 3 tabs)
1. **Sell**: upload photo -> LLM vision extracts brand/size/condition (NDI on a tag photo if time) -> Drex picks price (show probability bars) -> LLM writes listing. Use CSV comps; NO live scraping in the demo.
2. **Negotiate** (the showstopper): type a buyer message; Drex and the LLM answer side by side with live timers.
3. **Results**: benchmark table + bar charts (latency, cost, accuracy).

Do NOT automate real marketplaces (Depop/Poshmark/etc.) - mock it.

## Repo layout & file ownership (avoid merge conflicts)
- **Levi**: `agent/`, `bench/run_bench.py`
- **Friend (svanaparthy22)**: `app.py`, `bench/data/`, `assets/`
- `CLAUDE.md`, `README.md`, `requirements.txt` are shared - pull first, keep edits small.

Key modules:
- `agent/llm_client.py` - frontier LLM. `decide(facts, question, options) -> dict`.
- `agent/drex_client.py` - Drex. Same `decide()` signature and same return dict, so the bench and app can swap them.
- Return dict: `answer` (one of options, or None), `probabilities` (dict or None), `latency_s`, `input_tokens`, `output_tokens`, `cost_usd`, `invalid` (bool), `raw`, `error`.
- `bench/run_bench.py` - runs both benchmarks, writes `bench/results/summary.json`.
- Sample data: `bench/data/*_SAMPLE.csv` (fake, 5 rows each). The bench uses the real CSV if present, else the sample.

## Setup / running (macOS)
```bash
cd ~/Documents/drexathon
/opt/homebrew/bin/python3.11 -m venv .venv      # once (needs Python 3.10+; macOS system python is 3.9 - too old)
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
python bench/run_bench.py --models llm          # or: --models llm,drex   --limit 3
```

`.env` in the repo root (never committed):
```
DREX_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
LLM_PROVIDER=openai          # or anthropic
LLM_MODEL=gpt-4o             # whichever is the strongest we have keys for
```
