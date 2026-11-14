"""
Monochrome look from Levi's Figma design (black, white, grey; sharp corners;
blackletter logo; tight bold headlines; wide spaced-out uppercase labels).

It is a LAYER on top of the neon theme in app.py: app.py injects its own CSS first,
then this file's MONO_CSS, which overrides colours, fonts and shapes and switches
off the neon background. Nothing here changes data, numbers or model calls.

Turn it off in app.py with UI_THEME = "neon".
"""

# Colours the Altair charts use when this theme is on (CSS can't reach inside charts).
MONO_CHART = {
    "pick": "#FFFFFF",        # the option Drex picked
    "unpick": "#3A3A3A",      # the other options
    "models": ["#FFFFFF", "#BDBDBD", "#5C5C5C"],  # Drex Max, Drex 2.0, Claude
    "text": "#EDEDED",
    "muted": "#8A8A8A",
}

MONO_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..900&family=Inter:wght@400;500;600;700&family=UnifrakturMaguntia&display=swap');

/* ---------------------------------------------------------------- palette */
:root {
  --bg: #0A0A0A; --bg-2: #111111; --panel: #111111; --panel-2: #161616;
  --line: #2A2A2A; --text: #F2F2F2; --muted: #8A8A8A;
  --pink: #FFFFFF; --tangerine: #FFFFFF; --gold: #FFFFFF; --teal: #FFFFFF; --coral: #FFFFFF;
  --lav: #8A8A8A; --violet: #FFFFFF;
  --sunset: #FFFFFF; --ocean: #FFFFFF;
  --display: 'Archivo', 'Inter', sans-serif;
}
.stApp { background: var(--bg) !important; animation: none !important; color: var(--text); }
.bg-layer { display: none !important; }                     /* no orbs / grid / sun / scanlines */
[data-testid="stSidebar"] { background: #0D0D0D !important; border-right: 1px solid var(--line); }
.block-container { max-width: 1320px; padding-top: 1.2rem; }

/* Wide, spaced-out uppercase labels (the "LIVE MODEL SHOWDOWN" style) */
.eyebrow, .label, .race .name, .tl-node .r, .badge {
  font-family: 'Archivo', sans-serif !important; font-stretch: 125%; font-weight: 800 !important;
  letter-spacing: .22em !important; text-transform: uppercase; color: #F2F2F2 !important;
}
.label { color: #CFCFCF !important; font-size: .74rem !important; }

/* Tight, heavy headlines ("Let the agents handle the buyer.") */
h1, h2, h3, h4, .section-title {
  font-family: 'Archivo', sans-serif !important; font-weight: 800 !important; letter-spacing: -.045em !important;
  color: #FFFFFF !important;
}
.section-sub, .s, .tl-node .d { color: var(--muted) !important; }

/* ---------------------------------------------------------------- hero / logo */
.hero h1 { font-family: 'UnifrakturMaguntia', serif !important; font-weight: 400 !important;
           font-size: 5rem !important; letter-spacing: .02em !important; }
.hero h1 .ltr { background: none !important; color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important;
                animation: fadeUp .6s ease both !important; }
.hero .tag { color: #BDBDBD !important; }
.hero .tag b { color: #FFFFFF; border-right-color: #FFFFFF; }
.chip { border-radius: 0 !important; background: #111 !important; border: 1px solid var(--line) !important;
        box-shadow: none !important; font-family: 'Archivo', sans-serif !important; }
.chip b { color: #FFFFFF !important; }
.chip::after { display: none; }
.hero-art { border-radius: 0 !important; border: 1px solid var(--line) !important; box-shadow: none !important;
            background: #111 !important; animation: fadeUp .8s ease both !important; }
.ticker { background: #0A0A0A !important; border-color: var(--line) !important; }
.ticker .up, .ticker .hot { color: #FFFFFF !important; } .ticker .dim { color: var(--muted) !important; }
.ticker .track { font-family: 'Archivo', sans-serif !important; font-stretch: 125%; letter-spacing: .14em; font-size: .8rem; }

/* ---------------------------------------------------------------- tabs (SELL · NEGOTIATE · RESULTS) */
.stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid var(--line); gap: 28px; justify-content: center; }
.stTabs [data-baseweb="tab"] { font-family: 'Archivo', sans-serif !important; font-stretch: 125%; font-weight: 800;
  letter-spacing: .16em; text-transform: uppercase; border-radius: 0; color: #6E6E6E; background: transparent !important; }
.stTabs [data-baseweb="tab"]:hover { color: #FFFFFF; }
.stTabs [aria-selected="true"] { color: #FFFFFF !important; }
.stTabs [data-baseweb="tab-highlight"] { background: #FFFFFF !important; animation: none !important; height: 3px; }

/* newer Streamlit tabs (react-aria): data-testid="stTab" + a sliding SelectionIndicator */
[role="tablist"] { justify-content: center; gap: 36px; border-bottom: 1px solid var(--line); }
[data-testid="stTab"] { background: transparent !important; padding: 14px 4px !important; }
[data-testid="stTab"] p { font-family: 'Archivo', sans-serif !important; font-stretch: 125%; font-weight: 800 !important;
  letter-spacing: .18em; text-transform: uppercase; font-size: 1rem !important; color: #6E6E6E; }
[data-testid="stTab"][aria-selected="true"] p, [data-testid="stTab"]:hover p { color: #FFFFFF !important; }
.react-aria-SelectionIndicator { background: #FFFFFF !important; border-color: #FFFFFF !important; height: 3px !important; }

/* ---------------------------------------------------------------- cards: square, thin grey border */
.card, .race, .kpi, .policy, .timeline, .listing, .callout, [data-testid="stExpander"], [data-testid="stDataFrame"] {
  border-radius: 0 !important; background: var(--panel) !important; border: 1px solid var(--line) !important;
  box-shadow: none !important; backdrop-filter: none !important;
}
.race:hover, .kpi:hover { transform: none !important; }
/* the winner / Drex card gets the thick white frame */
.race.drex, .kpi.hot { border: 2px solid #FFFFFF !important; background: var(--panel) !important; }
.race.drex.first { animation: none !important; }
.race.drex.done { box-shadow: none !important; }
.race.llm { filter: none !important; }
.race.drex .name, .race.llm .name { color: #FFFFFF !important; }
.badge { background: transparent !important; color: #FFFFFF !important; border: 0; padding: 0; font-size: .7rem !important; }

/* the big answer word: white, boxed, no neon glitch */
.action { color: #FFFFFF !important; font-family: 'Archivo', sans-serif !important; font-weight: 900 !important;
          letter-spacing: .02em !important; font-size: 2.6rem !important; border: 1px solid #FFFFFF;
          padding: 10px 18px; animation: fadeUp .4s ease both !important; }
.action.glitch::before, .action.glitch::after { display: none !important; }
.timer { background: transparent !important; border: 0 !important; padding: 0 !important; color: #FFFFFF !important;
         text-shadow: none !important; font-family: 'Archivo', sans-serif !important; font-weight: 800; font-size: 2.6rem !important; }
.timer small { color: var(--muted) !important; }
.thinking { background: none !important; color: #5C5C5C !important; -webkit-text-fill-color: #5C5C5C !important;
            font-family: 'Archivo', sans-serif !important; }
.idle { color: #2E2E2E !important; font-family: 'Archivo', sans-serif !important; }
.progress { border-radius: 0 !important; background: #1E1E1E !important; height: 4px; }
.progress i, .race.llm .progress i { background: #FFFFFF !important; animation: none !important; }
.race.llm .progress i { background: #5C5C5C !important; }
.scanner { display: none !important; }
.pill { border-radius: 0 !important; }
.ok  { background: transparent !important; color: #FFFFFF !important; border: 1px solid #FFFFFF !important; }
.bad { background: transparent !important; color: #8A8A8A !important; border: 1px dashed #5C5C5C !important; }

/* chat line ("B  Would you take $30?") */
.avatar { background: transparent !important; border: 1.5px solid #FFFFFF; box-shadow: none !important; color: #FFF;
          font-family: 'Archivo', sans-serif; font-weight: 800; }
.bubble { border-radius: 0 !important; background: #111 !important; border: 1px solid var(--line) !important; }
.bubble .dots i { background: #FFFFFF !important; }

/* "Drex wins this round" banner: solid white block, black text */
.banner { background: #FFFFFF !important; color: #0A0A0A !important; border-radius: 0 !important; box-shadow: none !important;
          animation: fadeUp .5s ease both !important; text-align: left; }
.banner::after, .spark { display: none !important; }
.banner .big { font-family: 'Archivo', sans-serif !important; font-weight: 800 !important; letter-spacing: -.03em; color: #0A0A0A; }
.banner small { color: #333 !important; }

/* scam warning: stays loud, but in monochrome */
.scam { background: #FFFFFF !important; color: #0A0A0A !important; border: 0 !important; border-radius: 0 !important; }
.scam::before { background: repeating-linear-gradient(45deg, #0A0A0A 0 8px, #FFFFFF 8px 16px) !important; }

/* ---------------------------------------------------------------- results */
.kpi .v { background: none !important; color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important;
          font-family: 'Archivo', sans-serif !important; font-weight: 800 !important; letter-spacing: -.04em; font-size: 3.4rem !important; }
.tl-dot, .tl-node.best .tl-dot { background: #FFFFFF !important; box-shadow: none !important; animation: none !important; }
.tl-line { background: var(--line) !important; animation: none !important; }
.tl-node .a { font-family: 'Archivo', sans-serif !important; }
.callout b { color: #FFFFFF !important; }
.sample { background: #111 !important; color: #FFFFFF !important; border: 1px dashed #FFFFFF !important; border-radius: 0 !important; }

/* ---------------------------------------------------------------- sell */
.pricetag { background: transparent !important; border: 2px solid #FFFFFF !important; border-radius: 0 !important;
            box-shadow: none !important; }
.pricetag::before { display: none; }
.pricetag .p { font-family: 'Archivo', sans-serif !important; font-weight: 900 !important; letter-spacing: -.04em; color: #FFF; }
.pricetag .m { color: #BDBDBD !important; }
.listing .img { background: #161616 !important; }
.listing .pr { color: #FFFFFF !important; font-family: 'Archivo', sans-serif !important; }
.listing .d { color: #CFCFCF !important; }

/* ---------------------------------------------------------------- inputs & buttons */
.stButton > button[kind="primary"] { background: #FFFFFF !important; color: #0A0A0A !important; border: 0 !important;
  border-radius: 0 !important; box-shadow: none !important; animation: none !important; padding: .95rem 1rem;
  font-family: 'Archivo', sans-serif !important; font-stretch: 125%; letter-spacing: .16em; text-transform: uppercase; }
.stButton > button[kind="primary"] p { color: #0A0A0A !important; font-weight: 800 !important; }
.stButton > button[kind="primary"]:hover { background: #DADADA !important; transform: none !important; }
.stButton > button:not([kind="primary"]) { border-radius: 0 !important; border: 1px solid var(--line) !important;
  background: #111 !important; color: #DADADA !important; }
.stButton > button:not([kind="primary"]):hover { border-color: #FFFFFF !important; color: #FFFFFF !important; }
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {
  border-radius: 0 !important; background: #111 !important; border-color: var(--line) !important; }
input:focus, textarea:focus, [data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within,
[data-baseweb="select"]:focus-within { box-shadow: 0 0 0 1px #FFFFFF !important; border-radius: 0 !important; }
[data-testid="stWidgetLabel"] p { font-family: 'Archivo', sans-serif; font-stretch: 125%; font-weight: 700;
  letter-spacing: .14em; text-transform: uppercase; font-size: .78rem; color: #9A9A9A; }
[data-testid="stFileUploaderDropzone"] { background: #111 !important; border: 1.5px dashed #5C5C5C !important; border-radius: 0 !important;
  box-shadow: none !important; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: #FFFFFF !important; }
[data-testid="stMetricValue"] { font-family: 'Archivo', sans-serif; font-weight: 800; letter-spacing: -.03em; }
[role="radiogroup"] label { border-radius: 0 !important; }
</style>
"""
