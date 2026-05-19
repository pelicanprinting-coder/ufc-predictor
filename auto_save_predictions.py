import os
import pickle
import requests
import numpy as np
import pandas as pd
import base64
import io
from datetime import date, datetime, timedelta
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO  = os.getenv("GITHUB_REPO", "jdanielbcosta/ufc-predictor")
API_KEY      = os.getenv("ODDS_API_KEY", "97d30892355e2d15de1257c0aa526a50")
HIST_FILE    = "historico.csv"

NOME_MAP = {
    "Paulo Henrique Costa": "Paulo Costa",
    "Junior dos Santos": "Junior Dos Santos",
    "Weili Zhang": "Zhang Weili",
    "Timmy Cuamba": "Timothy Cuamba",
    "Tommy Gantt": "Thomas Gantt",
}

def normalizar_nome(nome):
    return NOME_MAP.get(nome, nome)

def decimal_to_prob(odds):
    try:
        odds = float(odds)
        if np.isnan(odds) or odds <= 1.0: return None
        return 1 / odds
    except:
        return None

def load_models():
    with open("ufc_ensemble_v3.pkl", "rb") as f:
        models = pickle.load(f)
    with open("features_ensemble_v3.pkl", "rb") as f:
        features = pickle.load(f)
    lookup = pd.read_csv("fighter_lookup_final.csv")
    return models, features, lookup

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

def get_card_ufcstats():
    try:
        url = "http://www.ufcstats.com/statistics/events/upcoming"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        eventos = []
        import re
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
    except Exception as e:
        print(f"Erro UFCStats: {e}")
        return []

def get_card_evento(url_evento):
    try:
        r = requests.get(url_evento, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        combates = []
        for linha in soup.find_all("tr", class_="b-fight-details__table-row"):
            fighters = linha.find_all("a", class_="b-link b-link_style_black")
            if len(fighters) >= 2:
                f1 = fighters[0].text.strip()
                f2 = fighters[1].text.strip()
                if f1 and f2:
                    combates.append((f1, f2))
        return combates
    except:
        return []

def get_fighter_row(lookup, name):
    row = lookup[lookup["name"] == name]
    if not row.empty:
        return row.iloc[0]
    name_norm = normalizar_nome(name)
    row = lookup[lookup["name"] == name_norm]
    return row.iloc[0] if not row.empty else None

def build_features(r, b, features, r_odds=None, b_odds=None):
    def sd(col):
        try: return float(r.get(col, 0) or 0) - float(b.get(col, 0) or 0)
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

    r_dist = float(r.get("distance_ratio", 0) or 0)
    b_dist = float(b.get("distance_ratio", 0) or 0)
    r_clin = float(r.get("clinch_ratio", 0) or 0)
    b_clin = float(b.get("clinch_ratio", 0) or 0)
    r_gnd  = float(r.get("ground_ratio", 0) or 0)
    b_gnd  = float(b.get("ground_ratio", 0) or 0)
    r_head = float(r.get("head_ratio", 0) or 0)
    b_head = float(b.get("head_ratio", 0) or 0)
    r_body = float(r.get("body_ratio", 0) or 0)
    b_body = float(b.get("body_ratio", 0) or 0)
    r_leg  = float(r.get("leg_ratio", 0) or 0)
    b_leg  = float(b.get("leg_ratio", 0) or 0)

    feats["style_clash_position"] = np.sqrt((r_dist-b_dist)**2+(r_clin-b_clin)**2+(r_gnd-b_gnd)**2)
    feats["style_clash_target"]   = np.sqrt((r_head-b_head)**2+(r_body-b_body)**2+(r_leg-b_leg)**2)
    feats["grappling_advantage"]  = (r_gnd-b_gnd)*feats["style_clash_position"]

    r_rank = float(r.get("rank", 16) or 16)
    b_rank = float(b.get("rank", 16) or 16)
    feats["R_ranked"]  = int(pd.notna(r.get("rank")))
    feats["B_ranked"]  = int(pd.notna(b.get("rank")))
    feats["rank_dif"]  = b_rank - r_rank

    r_prob = decimal_to_prob(r_odds) or 0.5
    b_prob = decimal_to_prob(b_odds) or 0.5
    feats["odds_prob_diff"] = r_prob - b_prob

    r_rest = float(r.get("dias_inativo", 180) or 180)
    b_rest = float(b.get("dias_inativo", 180) or 180)
    feats["rest_diff"] = r_rest - b_rest

    return pd.DataFrame([feats])[features].fillna(0)

def prever(models, features, lookup, f1_name, f2_name, odds_f1=None, odds_f2=None):
    r = get_fighter_row(lookup, f1_name)
    b = get_fighter_row(lookup, f2_name)
    if r is None or b is None:
        return None
    X = build_features(r, b, features, odds_f1, odds_f2)
    probs = np.array([m.predict_proba(X)[0][1] for m in models.values()])
    return probs.mean(), 1 - probs.mean()

def conviction_label(prob):
    if prob >= 0.75: return "HIGH CONVICTION 83%"
    if prob >= 0.70: return "MODERATE CONVICTION 79%"
    if prob >= 0.60: return "SLIGHT FAVOURITE 74%"
    return "TOO CLOSE TO CALL"

def gh_get_historico():
    if not GITHUB_TOKEN:
        return pd.DataFrame()
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{HIST_FILE}"
    r = requests.get(url, headers={"Authorization": f"token {GITHUB_TOKEN}"}, timeout=10)
    if r.status_code == 200:
        content = base64.b64decode(r.json()["content"]).decode("utf-8")
        return pd.read_csv(io.StringIO(content))
    return pd.DataFrame()

def gh_save_historico(df):
    if not GITHUB_TOKEN:
        return False
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{HIST_FILE}"
    csv_content = df.to_csv(index=False)
    encoded = base64.b64encode(csv_content.encode()).decode()
    r = requests.get(url, headers={"Authorization": f"token {GITHUB_TOKEN}"}, timeout=10)
    payload = {"message": "Auto-save predictions", "content": encoded}
    if r.status_code == 200:
        payload["sha"] = r.json()["sha"]
    r2 = requests.put(url, headers={"Authorization": f"token {GITHUB_TOKEN}"},
                      json=payload, timeout=30)
    return r2.status_code in [200, 201]

def main():
    print(f"UFC Auto-Save — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    models, features, lookup = load_models()
    eventos = get_card_ufcstats()
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

    hist = gh_get_historico()
    existing = set()
    if not hist.empty and "fighter_1" in hist.columns:
        existing = set(zip(hist["fighter_1"], hist["fighter_2"], hist["event_name"]))

    new_rows = []
    today = date.today()

    for evento in eventos:
        # Guardar eventos nos próximos 7 dias
        dias_para_evento = (evento["data"].date() - today).days
        if dias_para_evento < 0 or dias_para_evento > 7:
            continue

        print(f"  Evento: {evento['nome']} ({dias_para_evento} dias)")

        for f1_ufc, f2_ufc in evento["card"]:
            if (f1_ufc, f2_ufc, evento["nome"]) in existing:
                continue

            par = tuple(sorted([f1_ufc.lower(), f2_ufc.lower()]))
            odds_info = odds_lookup.get(par)
            odds_f1, odds_f2 = (odds_info[2], odds_info[3]) if odds_info else (None, None)

            res = prever(models, features, lookup, f1_ufc, f2_ufc, odds_f1, odds_f2)
            if res is None:
                res = prever(models, features, lookup, f2_ufc, f1_ufc, odds_f2, odds_f1)
                if res is not None:
                    p2, p1 = res
                else:
                    continue
            else:
                p1, p2 = res

            winner = f1_ufc if p1 >= p2 else f2_ufc
            new_rows.append({
                "event_name":       evento["nome"],
                "fighter_1":        f1_ufc,
                "fighter_2":        f2_ufc,
                "prob_f1":          round(p1, 4),
                "prob_f2":          round(p2, 4),
                "predicted_winner": winner,
                "odds_f1":          odds_f1 or "",
                "odds_f2":          odds_f2 or "",
                "conviction":       conviction_label(max(p1, p2)),
                "actual_winner":    "",
                "correct":          "",
                "saved_at":         pd.Timestamp.now().strftime("%Y-%m-%d"),
            })
            print(f"    {f1_ufc} vs {f2_ufc} → {winner} ({max(p1,p2):.1%})")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([hist, new_df], ignore_index=True) if not hist.empty else new_df
        if gh_save_historico(combined):
            print(f"✅ {len(new_rows)} previsões guardadas")
        else:
            print("❌ Erro ao guardar")
    else:
        print("ℹ️ Sem novas previsões para guardar")

if __name__ == "__main__":
    main()
