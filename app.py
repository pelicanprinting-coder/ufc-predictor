
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import re
from datetime import datetime
from bs4 import BeautifulSoup

# ── CONFIGURAÇÃO ──────────────────────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("ODDS_API_KEY", "")

NOME_MAP = {
    "Paulo Henrique Costa": "Paulo Costa",
    "Junior dos Santos": "Junior Dos Santos",
    "Weili Zhang": "Zhang Weili",
    "Timmy Cuamba": "Timothy Cuamba",
    "Tommy Gantt": "Thomas Gantt",
}

def normalizar_nome(nome):
    return NOME_MAP.get(nome, nome)

@st.cache_data
def load_data():
    with open("ufc_ensemble_final.pkl", "rb") as f:
        models = pickle.load(f)
    with open("features_final.pkl", "rb") as f:
        features = pickle.load(f)
    lookup = pd.read_csv("fighter_lookup_final.csv")

    models_dec, models_o25 = None, None
    features_dec, features_o25 = None, None
    try:
        with open("model_decision.pkl", "rb") as f:
            models_dec = pickle.load(f)
        with open("model_over25.pkl", "rb") as f:
            models_o25 = pickle.load(f)
        with open("features_method_v2.pkl", "rb") as f:
            features_dec = pickle.load(f)
        with open("features_method.pkl", "rb") as f:
            features_o25 = pickle.load(f)
    except:
        pass
    return models, features, lookup, models_dec, models_o25, features_dec, features_o25

@st.cache_data(ttl=3600)
def get_upcoming_odds():
    try:
        url = (f"https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
               f"?regions=eu&markets=h2h&oddsFormat=decimal&apiKey={API_KEY}")
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []
        dados = r.json()
        vistos = {}
        for c in sorted(dados, key=lambda x: x["commence_time"]):
            par = tuple(sorted([c["home_team"], c["away_team"]]))
            if par not in vistos:
                vistos[par] = c
        return list(vistos.values())
    except:
        return []

@st.cache_data(ttl=3600)
def get_card_ufcstats():
    try:
        url = "http://www.ufcstats.com/statistics/events/upcoming"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        eventos = []
        for e in soup.find_all("tr", class_="b-statistics__table-row"):
            link = e.find("a")
            if not link:
                continue
            nome_evento = link.text.strip()
            url_evento  = link["href"]
            cells       = e.find_all("td")
            cell0_text  = cells[0].get_text(separator=" ").strip()
            local_str   = cells[1].text.strip() if len(cells) > 1 else ""
            match = re.search(r"([A-Z][a-z]+ \d+, \d{4})", cell0_text)
            if not match:
                continue
            try:
                data_dt = datetime.strptime(match.group(1), "%B %d, %Y")
            except:
                continue
            card = get_card_evento(url_evento)
            if card:
                eventos.append({
                    "nome": nome_evento,
                    "data": data_dt,
                    "local": local_str,
                    "card": card,
                })
        return eventos
    except:
        return []

def get_card_evento(url_evento):
    try:
        r = requests.get(url_evento, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        combates = []
        for linha in soup.find_all("tr", class_="b-fight-details__table-row"):
            fighteres = linha.find_all("a", class_="b-link b-link_style_black")
            if len(fighteres) >= 2:
                f1 = fighteres[0].text.strip()
                f2 = fighteres[1].text.strip()
                if f1 and f2:
                    combates.append((f1, f2))
        return combates
    except:
        return []

models, features, lookup, models_dec, models_o25, features_dec, features_o25 = load_data()
fighter_names = sorted(lookup["name"].dropna().tolist())

# ── FUNÇÕES ───────────────────────────────────────────────────────
def decimal_to_prob(odds):
    try:
        odds = float(odds)
        if np.isnan(odds) or odds <= 1.0: return None
        return 1 / odds
    except:
        return None

def get_fighter_row(name):
    row = lookup[lookup["name"] == name]
    if not row.empty:
        return row.iloc[0]
    name_norm = normalizar_nome(name)
    row = lookup[lookup["name"] == name_norm]
    return row.iloc[0] if not row.empty else None

def build_features(r, b, r_odds=None, b_odds=None):
    def sd(col):
        try: return float(r.get(col, 0) or 0) - float(b.get(col, 0) or 0)
        except: return 0.0
    def sv(row, col):
        try: return float(row.get(col, 0) or 0)
        except: return 0.0

    feats = {
        "win_streak_dif":         sd("win_streak"),
        "lose_streak_dif":        sd("lose_streak"),
        "longest_win_streak_dif": sd("longest_win_streak"),
        "win_dif":                sd("wins"),
        "loss_dif":               sd("losses"),
        "total_round_dif":        sd("total_rounds"),
        "total_title_bout_dif":   sd("title_bouts"),
        "ko_dif":                 sd("ko_wins"),
        "sub_dif":                sd("sub_wins"),
        "height_dif":             sd("height"),
        "reach_dif":              sd("reach"),
        "age_dif":                sd("age"),
    }

    rolling = ["SLpM","sig_str_acc","td_avg","td_acc","sub_avg","ctrl_avg",
               "kd_per_fight","head_ratio","body_ratio","leg_ratio",
               "distance_ratio","clinch_ratio","ground_ratio",
               "sig_pct_fade","td_fade","ctrl_fade",
               "finish_rate","ko_rate","sub_rate",
               "SApM","str_def","td_def","rev_avg"]
    for c in rolling:
        feats[f"{c}_diff"] = sd(c)

    r_dist = sv(r,"distance_ratio"); b_dist = sv(b,"distance_ratio")
    r_clin = sv(r,"clinch_ratio");   b_clin = sv(b,"clinch_ratio")
    r_gnd  = sv(r,"ground_ratio");   b_gnd  = sv(b,"ground_ratio")
    r_head = sv(r,"head_ratio");     b_head = sv(b,"head_ratio")
    r_body = sv(r,"body_ratio");     b_body = sv(b,"body_ratio")
    r_leg  = sv(r,"leg_ratio");      b_leg  = sv(b,"leg_ratio")

    feats["style_clash_position"] = np.sqrt((r_dist-b_dist)**2+(r_clin-b_clin)**2+(r_gnd-b_gnd)**2)
    feats["style_clash_target"]   = np.sqrt((r_head-b_head)**2+(r_body-b_body)**2+(r_leg-b_leg)**2)
    feats["grappling_advantage"]  = (r_gnd-b_gnd)*feats["style_clash_position"]

    r_rank = sv(r,"rank") if pd.notna(r.get("rank")) else 16
    b_rank = sv(b,"rank") if pd.notna(b.get("rank")) else 16
    feats["R_ranked"]  = int(pd.notna(r.get("rank")))
    feats["B_ranked"]  = int(pd.notna(b.get("rank")))
    feats["rank_dif"]  = b_rank - r_rank

    r_prob = decimal_to_prob(r_odds) or 0.5
    b_prob = decimal_to_prob(b_odds) or 0.5
    feats["odds_prob_diff"] = r_prob - b_prob

    return pd.DataFrame([feats])[features].fillna(0)

def prever(f1_name, f2_name, odds_f1=None, odds_f2=None,
           r_ko_odds=None, r_sub_odds=None, r_dec_odds=None,
           b_ko_odds=None, b_sub_odds=None, b_dec_odds=None,
           no_of_rounds=3):
    r = get_fighter_row(f1_name)
    b = get_fighter_row(f2_name)
    if r is None or b is None:
        return None
    X = build_features(r, b, odds_f1, odds_f2)
    probs = np.array([m.predict_proba(X)[0][1] for m in models.values()])
    prob_winner = probs.mean()

    # Previsões secundárias
    prob_decision, prob_over25 = None, None

    if models_dec and features_dec:
        try:
            # Adicionar features de método
            X_dec = X.copy()
            def op(o): return (1/float(o)) if o and float(o) > 1 else 0.5

            extra = {
                'r_ko_prob':  op(r_ko_odds),  'r_sub_prob': op(r_sub_odds),
                'r_dec_prob': op(r_dec_odds),  'b_ko_prob':  op(b_ko_odds),
                'b_sub_prob': op(b_sub_odds),  'b_dec_prob': op(b_dec_odds),
                'no_of_rounds': no_of_rounds,
            }
            for col, val in extra.items():
                X_dec[col] = val

            # Adicionar historial de decisões do lookup
            for col in ['R_win_by_Decision_Unanimous','R_win_by_Decision_Split',
                        'R_win_by_Decision_Majority','R_win_by_KO/TKO','R_win_by_Submission',
                        'B_win_by_Decision_Unanimous','B_win_by_Decision_Split',
                        'B_win_by_Decision_Majority','B_win_by_KO/TKO','B_win_by_Submission']:
                X_dec[col] = 0

            X_dec_final = X_dec.reindex(columns=features_dec, fill_value=0).fillna(0)
            probs_dec = np.array([m.predict_proba(X_dec_final)[0][1]
                                  for m in models_dec.values()])
            prob_decision = probs_dec.mean()
        except:
            pass

    if models_o25 and features_o25:
        try:
            X_o25 = X.copy()
            for col, val in extra.items():
                X_o25[col] = val
            X_o25_final = X_o25.reindex(columns=features_o25, fill_value=0).fillna(0)
            probs_o25 = np.array([m.predict_proba(X_o25_final)[0][1]
                                  for m in models_o25.values()])
            prob_over25 = probs_o25.mean()
        except:
            pass

    return prob_winner, 1 - prob_winner, r, b, prob_decision, prob_over25

def conviction_label(prob):
    if prob >= 0.80: return 0, "HIGH CONVICTION", "#D4AF37", "high"
    if prob >= 0.75: return 1, "MODERATE CONVICTION", "#C0A030", "moderate"
    if prob >= 0.65: return 2, "SLIGHT FAVOURITE", "#888888", "slight"
    return 3, "TOO CLOSE TO CALL", "#555555", "close"


def conviction_order(prob):
    if prob >= 0.80: return 0
    if prob >= 0.75: return 1
    if prob >= 0.65: return 2
    return 3

def fmt_stat(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    if isinstance(v, float): return f"{v:.3f}"
    try: return str(int(v))
    except: return str(v)

# ── CSS GLOBAL ────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800;900&family=Barlow:wght@300;400;500;600&display=swap');

    /* ── Root & Body ── */
    :root {
        --gold:    #C9A84C;
        --gold-lt: #e0b96a;
        --red:     #C8102E;
        --red-lt:  #e8253f;
        --blue:    #1a6cff;
        --bg:      #0a0a0b;
        --bg2:     #111114;
        --bg3:     #18181d;
        --bg4:     #1f1f26;
        --border:  #2a2a33;
        --text:    #f0f0f0;
        --muted:   #7a7a8a;
        --green:   #22c55e;
    }

    html, body, [class*="css"] {
        font-family: 'Barlow', sans-serif;
        background-color: var(--bg) !important;
        color: var(--text);
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg2); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg2);
        border-radius: 10px;
        padding: 4px;
        gap: 2px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: var(--muted) !important;
        border-radius: 8px;
        padding: 8px 20px;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--gold) !important;
        color: #000 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }

    /* ── Buttons ── */
    .stButton > button {
        font-family: 'Barlow Condensed', sans-serif;
        font-weight: 800;
        letter-spacing: 0.08em;
        font-size: 1rem;
        background: var(--gold);
        color: #000;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.2s;
        text-transform: uppercase;
    }
    .stButton > button:hover {
        background: var(--gold-lt);
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(201,168,76,0.35);
    }

    /* ── Selectboxes & inputs ── */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background: var(--bg3) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 8px !important;
    }
    .stSelectbox > label,
    .stNumberInput > label {
        color: var(--muted) !important;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ── Metrics ── */
    [data-testid="metric-container"] {
        background: var(--bg3);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="metric-container"] label {
        color: var(--muted) !important;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--gold) !important;
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1.8rem !important;
        font-weight: 800;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: var(--bg3) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        font-family: 'Barlow Condensed', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: var(--text) !important;
        padding: 14px 18px !important;
    }
    .streamlit-expanderHeader:hover {
        border-color: var(--gold) !important;
    }
    .streamlit-expanderContent {
        background: var(--bg2) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
        padding: 18px !important;
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: var(--gold) !important; }

    /* ── Alerts ── */
    .stAlert { border-radius: 8px !important; }

    /* ── Dataframe ── */
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    [data-testid="stDataFrame"] * { font-family: 'Barlow', sans-serif !important; font-size: 0.85rem !important; }

    /* ── Custom components ── */
    .ufc-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 0.5rem;
    }
    .ufc-logo-text {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: -0.01em;
        line-height: 1;
        background: linear-gradient(135deg, #C9A84C 0%, #e0b96a 50%, #C9A84C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .ufc-subtitle {
        font-size: 0.78rem;
        color: var(--muted);
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 500;
        margin-top: 2px;
    }
    .badge-stat {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--bg3);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--muted);
        margin-right: 6px;
    }
    .badge-stat span { color: var(--gold); font-weight: 700; }

    /* ── Event header card ── */
    .event-card {
        background: linear-gradient(135deg, var(--bg3) 0%, var(--bg4) 100%);
        border: 1px solid var(--border);
        border-left: 4px solid var(--gold);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .event-name {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--text);
        letter-spacing: 0.02em;
        margin: 0 0 4px 0;
    }
    .event-meta {
        font-size: 0.82rem;
        color: var(--muted);
        letter-spacing: 0.04em;
    }
    .event-meta strong { color: var(--gold); }

    /* ── Fight card ── */
    .fight-card {
        background: var(--bg3);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 18px 20px 14px;
        margin-bottom: 10px;
        transition: border-color 0.2s;
        position: relative;
        overflow: hidden;
    }
    .fight-card:hover { border-color: var(--gold); }
    .fight-card-high { border-left: 4px solid #22c55e; }
    .fight-card-med  { border-left: 4px solid var(--gold); }
    .fight-card-low  { border-left: 4px solid var(--blue); }
    .fight-card-draw { border-left: 4px solid var(--muted); }

    .fight-names {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: 0.03em;
        color: var(--text);
        margin-bottom: 12px;
    }
    .fight-names .vs {
        color: var(--muted);
        font-size: 1rem;
        font-weight: 400;
        margin: 0 8px;
    }
    .fight-names .fav { color: var(--gold); }

    .conviction-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .conv-HIGH  { background: rgba(34,197,94,0.15);  color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
    .conv-MED   { background: rgba(201,168,76,0.15); color: var(--gold); border: 1px solid rgba(201,168,76,0.3); }
    .conv-LOW   { background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
    .conv-DRAW  { background: rgba(107,114,128,0.15); color: #9ca3af; border: 1px solid rgba(107,114,128,0.3); }

    /* ── Probability bar ── */
    .prob-bar-wrap {
        margin: 10px 0 6px;
    }
    .prob-bar-labels {
        display: flex;
        justify-content: space-between;
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .prob-bar-labels .name-left  { color: var(--red-lt); }
    .prob-bar-labels .name-right { color: #60a5fa; text-align: right; }
    .prob-bar-labels .pct-left   { color: var(--red-lt); margin-left: 4px; }
    .prob-bar-labels .pct-right  { color: #60a5fa; margin-right: 4px; }
    .prob-bar-outer {
        height: 10px;
        background: var(--bg4);
        border-radius: 999px;
        overflow: hidden;
        border: 1px solid var(--border);
    }
    .prob-bar-inner {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--red) 0%, var(--red-lt) 45%, transparent 45%,
                    transparent 55%, #3b82f6 55%, var(--blue) 100%);
        transition: width 0.4s ease;
    }

    /* ── Odds row ── */
    .odds-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 8px;
        font-size: 0.8rem;
        color: var(--muted);
    }
    .odds-chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: var(--bg4);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .odds-chip .ov { color: var(--gold); }

    /* ── Section divider ── */
    .section-divider {
        border: none;
        border-top: 1px solid var(--border);
        margin: 24px 0;
    }

    /* ── Predict tab ── */
    .corner-label {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1.1rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 6px 14px;
        border-radius: 6px;
        margin-bottom: 12px;
        display: inline-block;
    }
    .corner-red  { background: rgba(200,16,46,0.15); color: var(--red-lt); border: 1px solid rgba(200,16,46,0.3); }
    .corner-blue { background: rgba(26,108,255,0.15); color: #60a5fa; border: 1px solid rgba(26,108,255,0.3); }

    .result-card {
        background: var(--bg3);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 24px;
        margin: 16px 0;
        text-align: center;
    }
    .result-winner {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 2rem;
        font-weight: 900;
        letter-spacing: 0.02em;
        color: var(--gold);
    }
    .result-pct {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--muted);
        margin-top: 2px;
    }

    .vs-divider {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1.5rem;
        font-weight: 900;
        color: var(--muted);
        text-align: center;
        margin: 8px 0;
    }

    /* Edge indicator */
    .edge-pos { color: #22c55e; font-weight: 700; }
    .edge-neg { color: var(--red-lt); font-weight: 700; }
    .edge-neu { color: var(--muted); font-weight: 700; }

    /* Stats table override */
    [data-testid="stDataFrame"] th {
        background: var(--bg4) !important;
        color: var(--muted) !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stDataFrame"] td { color: var(--text) !important; }
    </style>
    """, unsafe_allow_html=True)

# ── HELPERS DE RENDER ─────────────────────────────────────────────
def render_fight_card(c):
    fav    = c["f1"] if c["p1"] >= c["p2"] else c["f2"]
    em, lbl, color, level = conviction_label(c["prob_fav"])
    card_class = {"high": "fight-card-high", "moderate": "fight-card-med",
                  "slight": "fight-card-low", "close": "fight-card-draw"}[level]

    p1_pct = c["p1"] * 100
    p2_pct = c["p2"] * 100

    # Pre-compute name classes to avoid quote conflicts in f-string
    f1_class = "fav" if c["p1"] >= c["p2"] else "plain-name"
    f2_class = "fav" if c["p2"] > c["p1"] else "plain-name"
    f1_name  = c["f1"]
    f2_name  = c["f2"]

    # Odds strings
    odds_f1_str = f"{c['odds_f1']:.2f}" if c["odds_f1"] else "—"
    odds_f2_str = f"{c['odds_f2']:.2f}" if c["odds_f2"] else "—"
    mkt_f1_str  = f"{decimal_to_prob(c['odds_f1']):.0%}" if c["odds_f1"] else "—"
    mkt_f2_str  = f"{decimal_to_prob(c['odds_f2']):.0%}" if c["odds_f2"] else "—"

    # Edge calculation — fully pre-computed, no f-string nesting
    edge_f1_html = ""
    edge_f2_html = ""
    if c["odds_f1"]:
        diff = c["p1"] - decimal_to_prob(c["odds_f1"])
        cls  = "edge-pos" if diff > 0.03 else ("edge-neg" if diff < -0.03 else "edge-neu")
        sign = f"{diff:+.0%}"
        edge_f1_html = f'<span class="{cls}">({sign})</span>'
    if c["odds_f2"]:
        diff = c["p2"] - decimal_to_prob(c["odds_f2"])
        cls  = "edge-pos" if diff > 0.03 else ("edge-neg" if diff < -0.03 else "edge-neu")
        sign = f"{diff:+.0%}"
        edge_f2_html = f'<span class="{cls}">({sign})</span>'

    p1_str = f"{p1_pct:.1f}%"
    p2_str = f"{p2_pct:.1f}%"
    p1_w   = f"{p1_pct:.1f}%"
    p2_w   = f"{p2_pct:.1f}%"

    # Warnings para title fights e 5 rounds
    warnings = []
    if c.get("title_bout"):
        warnings.append('<span style="background:rgba(200,16,46,0.15); color:#e8253f; '
                        'border:1px solid rgba(200,16,46,0.3); border-radius:4px; '
                        'padding:2px 8px; font-size:0.72rem; font-weight:700; '
                        'letter-spacing:0.05em;">🏆 TITLE FIGHT — model less reliable (64%)</span>')
    if c.get("rounds") == 5 and not c.get("title_bout"):
        warnings.append('<span style="background:rgba(255,165,0,0.15); color:#ffa500; '
                        'border:1px solid rgba(255,165,0,0.3); border-radius:4px; '
                        'padding:2px 8px; font-size:0.72rem; font-weight:700; '
                        'letter-spacing:0.05em;">⏱️ 5 ROUNDS — model less reliable (62%)</span>')
    warnings_html = " ".join(warnings)

    html = (
        f'<div class="fight-card {card_class}">'
        f'  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">'
        f'    <div class="fight-names">'
        f'      <span class="{f1_class}">{f1_name}</span>'
        f'      <span class="vs">VS</span>'
        f'      <span class="{f2_class}">{f2_name}</span>'
        f'    </div>'
        f'    <div class="conviction-badge conv-{level}">{em} {lbl}</div>'
        f'  </div>'
        f'  <div class="prob-bar-wrap">'
        f'    <div class="prob-bar-labels">'
        f'      <span><span class="name-left">{f1_name}</span>'
        f'            <span class="pct-left"> {p1_str}</span></span>'
        f'      <span><span class="pct-right">{p2_str} </span>'
        f'            <span class="name-right">{f2_name}</span></span>'
        f'    </div>'
        f'    <div class="prob-bar-outer">'
        f'      <div style="height:100%; display:flex;">'
        f'        <div style="width:{p1_w}; background:linear-gradient(90deg,#C8102E,#e8253f);'
        f'             border-radius:999px 0 0 999px;"></div>'
        f'        <div style="width:{p2_w}; background:linear-gradient(90deg,#1a6cff,#60a5fa);'
        f'             border-radius:0 999px 999px 0; margin-left:auto;"></div>'
        f'      </div>'
        f'    </div>'
        f'  </div>'
        f'  <div class="odds-row">'
        f'    <div>'
        f'      <span class="odds-chip">&#128202; Odds <span class="ov">{odds_f1_str}</span></span>'
        f'      <span class="odds-chip">Market <span class="ov">{mkt_f1_str}</span></span>'
        f'      {edge_f1_html}'
        f'    </div>'
        f'    <div style="font-size:0.75rem; color:var(--muted);">Favourite:'
        f'      <strong style="color:var(--gold);">{fav}</strong>'
        f'    </div>'
        f'    <div>'
        f'      {edge_f2_html}'
        f'      <span class="odds-chip">Market <span class="ov">{mkt_f2_str}</span></span>'
        f'      <span class="odds-chip">&#128202; Odds <span class="ov">{odds_f2_str}</span></span>'
        f'    </div>'
        f'  </div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

# ── INIT ──────────────────────────────────────────────────────────
st.set_page_config(page_title="UFC Fight Predictor", page_icon="🥊", layout="wide")
inject_css()

# ── HEADER ────────────────────────────────────────────────────────
st.markdown("""
<div class="ufc-header">
  <div>
    <div class="ufc-logo-text">🥊 UFC FIGHT PREDICTOR</div>
    <div class="ufc-subtitle">Ensemble v4 · Powered by Machine Learning</div>
  </div>
</div>
<div style="margin: 10px 0 22px;">
  <span class="badge-stat">🎯 Accuracy <span>68.33%</span></span>
  <span class="badge-stat">🤖 Modelos <span>5 ensemble</span></span>
  <span class="badge-stat">⚡ Conviction 80%+ <span>90.2% acc</span></span>
  <span class="badge-stat">📈 ROI 80%+ <span>+3.9%</span></span>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📅  UPCOMING EVENTS", "🔍  PREDICT A FIGHT"])

# ── TAB 1 ─────────────────────────────────────────────────────────
with tab1:
    col_t, col_r = st.columns([5, 1])
    with col_t:
        st.markdown("""
        <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.6rem;
             font-weight:800; letter-spacing:0.04em; color:var(--text); margin-bottom:4px;">
          UPCOMING FIGHTS
        </div>
        """, unsafe_allow_html=True)
    with col_r:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Loading card and odds..."):
        eventos_ufc  = get_card_ufcstats()
        combates_api = get_upcoming_odds()

    # Lookup de odds
    odds_lookup = {}
    for c in combates_api:
        f1_api = normalizar_nome(c["home_team"])
        f2_api = normalizar_nome(c["away_team"])
        odds_f1_list, odds_f2_list = [], []
        for bm in c.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt["key"] == "h2h":
                    for outcome in mkt["outcomes"]:
                        nome = normalizar_nome(outcome["name"])
                        if nome == f1_api:
                            odds_f1_list.append(outcome["price"])
                        elif nome == f2_api:
                            odds_f2_list.append(outcome["price"])
        odds_f1 = float(np.median(odds_f1_list)) if odds_f1_list else None
        odds_f2 = float(np.median(odds_f2_list)) if odds_f2_list else None
        par = tuple(sorted([f1_api.lower(), f2_api.lower()]))
        odds_lookup[par] = (f1_api, f2_api, odds_f1, odds_f2)

    if not eventos_ufc:
        st.error("⚠️ Could not load UFC card.")
    else:
        for evento in eventos_ufc:
            # Event header
            st.markdown(f"""
            <div class="event-card">
              <div class="event-name">{evento['nome']}</div>
              <div class="event-meta">
                📍 {evento['local']} &nbsp;·&nbsp;
                <strong>{evento['data'].strftime('%d %B %Y')}</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)

            combates_proc = []
            sem_dados = []

            for f1_ufc, f2_ufc in evento["card"]:
                par = tuple(sorted([f1_ufc.lower(), f2_ufc.lower()]))
                odds_info = odds_lookup.get(par)
                odds_f1, odds_f2 = (odds_info[2], odds_info[3]) if odds_info else (None, None)

                res = prever(f1_ufc, f2_ufc, odds_f1, odds_f2)
                if res is None:
                    res = prever(f2_ufc, f1_ufc, odds_f2, odds_f1)
                    if res is not None:
                        p2, p1, _, _, _, _ = res
                    else:
                        sem_dados.append(f"{f1_ufc} vs {f2_ufc}")
                        continue
                else:
                    p1, p2, _, _, _, _ = res

                prob_fav = max(p1, p2)
                combates_proc.append({
                    "f1": f1_ufc, "f2": f2_ufc,
                    "p1": p1, "p2": p2,
                    "prob_fav": prob_fav,
                    "odds_f1": odds_f1, "odds_f2": odds_f2,
                    "ordem": conviction_order(prob_fav),
                    "title_bout": False,
                    "rounds": 3,
                })

            combates_proc.sort(key=lambda x: (x["ordem"], -x["prob_fav"]))

            # Section labels
            levels_seen = set()
            for c in combates_proc:
                _, lbl, color, level = conviction_label(c["prob_fav"])
                if level not in levels_seen:
                    levels_seen.add(level)
                    icon_map = {"high": "🟢", "moderate": "🟡", "slight": "🔵", "close": "⚪"}
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; gap:8px; margin: 18px 0 8px;">
                      <span style="font-size:0.9rem;">{icon_map[level]}</span>
                      <span style="font-family:'Barlow Condensed',sans-serif; font-size:0.85rem;
                           font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
                           color:{color};">{lbl}</span>
                      <div style="flex:1; height:1px; background:var(--border);"></div>
                    </div>
                    """, unsafe_allow_html=True)
                render_fight_card(c)

            if sem_dados:
                st.markdown(f"""
                <div style="background:var(--bg3); border:1px solid var(--border); border-radius:10px;
                     padding:12px 16px; margin-top:12px; color:var(--muted); font-size:0.85rem;">
                  ⚠️ <strong style="color:var(--text);">{len(sem_dados)} fights without sufficient data</strong><br>
                  {'  ·  '.join(sem_dados)}
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── TAB 2 ─────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.6rem;
         font-weight:800; letter-spacing:0.04em; color:var(--text); margin-bottom:20px;">
      PREVISÃO MANUAL
    </div>
    """, unsafe_allow_html=True)

    col1, col_vs, col2 = st.columns([5, 1, 5])

    with col1:
        st.markdown('<span class="corner-label corner-red">🔴 RED CORNER</span>', unsafe_allow_html=True)
        red_name     = st.selectbox("Fighter", fighter_names, key="red", label_visibility="collapsed")
        r_odds_input = st.number_input("Decimal odds", min_value=1.0, value=1.0, step=0.05, key="rodds")

    with col_vs:
        st.markdown('<div class="vs-divider" style="margin-top:36px;">VS</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<span class="corner-label corner-blue">🔵 BLUE CORNER</span>', unsafe_allow_html=True)
        blue_name    = st.selectbox("Fighter", fighter_names, key="blue", label_visibility="collapsed")
        b_odds_input = st.number_input("Decimal odds", min_value=1.0, value=1.0, step=0.05, key="bodds")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("🔮  PREDICT A FIGHT", use_container_width=True):
        if red_name == blue_name:
            st.warning("⚠️ Select two different fighters!")
        else:
            odds_r = r_odds_input if r_odds_input > 1.0 else None
            odds_b = b_odds_input if b_odds_input > 1.0 else None
            res = prever(red_name, blue_name, odds_r, odds_b,
                         r_ko_odds=r_ko_odds_in if r_ko_odds_in > 1.0 else None,
                         r_sub_odds=r_sub_odds_in if r_sub_odds_in > 1.0 else None,
                         r_dec_odds=r_dec_odds_in if r_dec_odds_in > 1.0 else None,
                         b_ko_odds=b_ko_odds_in if b_ko_odds_in > 1.0 else None,
                         b_sub_odds=b_sub_odds_in if b_sub_odds_in > 1.0 else None,
                         b_dec_odds=b_dec_odds_in if b_dec_odds_in > 1.0 else None,
                         no_of_rounds=rounds_in)

            if res is None:
                st.error("❌ Insufficient data for one or both fighters.")
            else:
                prob_r, prob_b, red, blue, prob_decision, prob_over25 = res
                winner  = red_name if prob_r > prob_b else blue_name
                w_corner = "🔴" if prob_r > prob_b else "🔵"
                em, lbl, color, level = conviction_label(max(prob_r, prob_b))

                # Result card
                st.markdown(f"""
                <div class="result-card" style="border-top: 3px solid {color};">
                  <div style="margin-bottom:6px;">
                    <span class="conviction-badge conv-{level}">{em} {lbl}</span>
                  </div>
                  <div class="result-winner">{w_corner} {winner}</div>
                  <div class="result-pct">Probabilidade: {max(prob_r, prob_b):.1%}</div>
                </div>
                """, unsafe_allow_html=True)

                # Probability bar
                p1_pct = prob_r * 100
                p2_pct = prob_b * 100
                st.markdown(f"""
                <div class="prob-bar-wrap">
                  <div class="prob-bar-labels">
                    <span>
                      <span class="name-left">{red_name}</span>
                      <span class="pct-left">{p1_pct:.1f}%</span>
                    </span>
                    <span>
                      <span class="pct-right">{p2_pct:.1f}%</span>
                      <span class="name-right">{blue_name}</span>
                    </span>
                  </div>
                  <div class="prob-bar-outer">
                    <div style="height:100%; display:flex;">
                      <div style="width:{p1_pct:.1f}%; background:linear-gradient(90deg,#C8102E,#e8253f); border-radius:999px 0 0 999px;"></div>
                      <div style="width:{p2_pct:.1f}%; background:linear-gradient(90deg,#1a6cff,#60a5fa); border-radius:0 999px 999px 0; margin-left:auto;"></div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Model vs Market
                if odds_r and odds_b:
                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                    st.markdown("""
                    <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.1rem;
                         font-weight:800; letter-spacing:0.08em; text-transform:uppercase;
                         color:var(--muted); margin-bottom:12px;">
                      💰 Model vs Market
                    </div>
                    """, unsafe_allow_html=True)

                    m1, m2 = st.columns(2)
                    prob_mkt_r = decimal_to_prob(odds_r)
                    prob_mkt_b = decimal_to_prob(odds_b)
                    diff_r = prob_r - prob_mkt_r
                    diff_b = prob_b - prob_mkt_b

                    with m1:
                        cls_r = "edge-pos" if diff_r > 0.03 else ("edge-neg" if diff_r < -0.03 else "edge-neu")
                        st.markdown(f"""
                        <div style="background:var(--bg3); border:1px solid rgba(200,16,46,0.3);
                             border-radius:10px; padding:16px;">
                          <div style="font-family:'Barlow Condensed',sans-serif; font-size:1rem;
                               font-weight:700; color:var(--red-lt); margin-bottom:10px;">🔴 {red_name}</div>
                          <div style="display:flex; flex-direction:column; gap:5px; font-size:0.85rem;">
                            <div style="display:flex; justify-content:space-between;">
                              <span style="color:var(--muted);">Modelo</span>
                              <strong style="color:var(--text);">{prob_r:.1%}</strong>
                            </div>
                            <div style="display:flex; justify-content:space-between;">
                              <span style="color:var(--muted);">Market</span>
                              <strong style="color:var(--text);">{prob_mkt_r:.1%}</strong>
                            </div>
                            <div style="display:flex; justify-content:space-between; border-top:1px solid var(--border); padding-top:5px;">
                              <span style="color:var(--muted);">Edge</span>
                              <strong class="{cls_r}">{diff_r:+.1%}</strong>
                            </div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with m2:
                        cls_b = "edge-pos" if diff_b > 0.03 else ("edge-neg" if diff_b < -0.03 else "edge-neu")
                        st.markdown(f"""
                        <div style="background:var(--bg3); border:1px solid rgba(26,108,255,0.3);
                             border-radius:10px; padding:16px;">
                          <div style="font-family:'Barlow Condensed',sans-serif; font-size:1rem;
                               font-weight:700; color:#60a5fa; margin-bottom:10px;">🔵 {blue_name}</div>
                          <div style="display:flex; flex-direction:column; gap:5px; font-size:0.85rem;">
                            <div style="display:flex; justify-content:space-between;">
                              <span style="color:var(--muted);">Modelo</span>
                              <strong style="color:var(--text);">{prob_b:.1%}</strong>
                            </div>
                            <div style="display:flex; justify-content:space-between;">
                              <span style="color:var(--muted);">Market</span>
                              <strong style="color:var(--text);">{prob_mkt_b:.1%}</strong>
                            </div>
                            <div style="display:flex; justify-content:space-between; border-top:1px solid var(--border); padding-top:5px;">
                              <span style="color:var(--muted);">Edge</span>
                              <strong class="{cls_b}">{diff_b:+.1%}</strong>
                            </div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Stats table
                # Previsões secundárias
                if prob_decision is not None or prob_over25 is not None:
                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                    st.markdown("""
                    <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.1rem;
                         font-weight:800; letter-spacing:0.08em; text-transform:uppercase;
                         color:var(--muted); margin-bottom:12px;">
                      🎯 Secondary Markets
                    </div>
                    """, unsafe_allow_html=True)

                    sec_cols = st.columns(2)
                    if prob_decision is not None:
                        with sec_cols[0]:
                            dec_label = "LIKELY DECISION" if prob_decision > 0.5 else "LIKELY FINISH"
                            dec_color = "#D4AF37" if prob_decision > 0.5 else "#e8253f"
                            st.markdown(f"""
                            <div style="background:var(--bg3); border:1px solid var(--border);
                                 border-radius:10px; padding:16px; text-align:center;">
                              <div style="font-size:0.75rem; color:var(--muted);
                                   letter-spacing:2px; text-transform:uppercase;
                                   margin-bottom:6px;">Goes to Decision</div>
                              <div style="font-family:'Barlow Condensed',sans-serif;
                                   font-size:2rem; font-weight:900;
                                   color:{dec_color};">{prob_decision*100:.0f}%</div>
                              <div style="font-size:0.75rem; color:{dec_color};
                                   font-weight:700; letter-spacing:1px;
                                   margin-top:4px;">{dec_label}</div>
                              <div style="font-size:0.7rem; color:var(--muted);
                                   margin-top:4px;">Model accuracy: 59.45%</div>
                            </div>
                            """, unsafe_allow_html=True)
                    if prob_over25 is not None:
                        with sec_cols[1]:
                            o25_label = "LIKELY OVER" if prob_over25 > 0.5 else "LIKELY UNDER"
                            o25_color = "#22c55e" if prob_over25 > 0.5 else "#60a5fa"
                            st.markdown(f"""
                            <div style="background:var(--bg3); border:1px solid var(--border);
                                 border-radius:10px; padding:16px; text-align:center;">
                              <div style="font-size:0.75rem; color:var(--muted);
                                   letter-spacing:2px; text-transform:uppercase;
                                   margin-bottom:6px;">Over / Under 2.5 Rounds</div>
                              <div style="font-family:'Barlow Condensed',sans-serif;
                                   font-size:2rem; font-weight:900;
                                   color:{o25_color};">{prob_over25*100:.0f}%</div>
                              <div style="font-size:0.75rem; color:{o25_color};
                                   font-weight:700; letter-spacing:1px;
                                   margin-top:4px;">{o25_label}</div>
                              <div style="font-size:0.7rem; color:var(--muted);
                                   margin-top:4px;">Model accuracy: 62.43%</div>
                            </div>
                            """, unsafe_allow_html=True)

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("""
                <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.1rem;
                     font-weight:800; letter-spacing:0.08em; text-transform:uppercase;
                     color:var(--muted); margin-bottom:12px;">
                  📊 Comparação de Estatísticas
                </div>
                """, unsafe_allow_html=True)

                stats_map = [
                    ("Vitórias",             "wins"),
                    ("Derrotas",             "losses"),
                    ("Win Streak",           "win_streak"),
                    ("KO Wins",              "ko_wins"),
                    ("Sub Wins",             "sub_wins"),
                    ("Altura (cm)",          "height"),
                    ("Reach (cm)",           "reach"),
                    ("Idade",                "age"),
                    ("SLpM",                 "SLpM"),
                    ("SApM",                 "SApM"),
                    ("Str. Accuracy",        "sig_str_acc"),
                    ("Str. Defence",         "str_def"),
                    ("TD avg",               "td_avg"),
                    ("TD Defence",           "td_def"),
                    ("Finish Rate",          "finish_rate"),
                    ("KO Rate",              "ko_rate"),
                    ("Sub Rate",             "sub_rate"),
                    ("Win Rate recente",     "winrate_recente"),
                    ("KO sofridos (ult. 3)", "ko_sofrido_recente"),
                    ("Dias inativo",         "dias_inativo"),
                ]
                rows = []
                for label, col in stats_map:
                    rows.append({
                        "Estatística":      label,
                        f"🔴 {red_name}":   fmt_stat(red.get(col)),
                        f"🔵 {blue_name}":  fmt_stat(blue.get(col)),
                    })
                st.dataframe(
                    pd.DataFrame(rows).set_index("Estatística"),
                    use_container_width=True
                )
