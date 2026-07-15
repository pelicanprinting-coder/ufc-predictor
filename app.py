
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
import base64
import io
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
    # Tentar v5 (ufc_age + win_rate + odds movement), fallback para v3
    try:
        with open("ufc_ensemble_v5.pkl", "rb") as f:
            models = pickle.load(f)
        with open("features_ensemble_v5.pkl", "rb") as f:
            features = pickle.load(f)
    except:
        with open("ufc_ensemble_v3.pkl", "rb") as f:
            models = pickle.load(f)
        with open("features_ensemble_v3.pkl", "rb") as f:
            features = pickle.load(f)
    lookup = pd.read_csv("fighter_lookup_final.csv")

    # BFO movement features
    bfo_features = None
    try:
        bfo_features = pd.read_csv("bfo_aligned_features.csv")
    except:
        pass

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
    meta_model, meta_feats = None, None
    try:
        with open("ufc_meta_model.pkl", "rb") as f:
            meta_model = pickle.load(f)
        with open("meta_features.pkl", "rb") as f:
            meta_feats = pickle.load(f)
    except:
        pass
    beta_cal = None
    try:
        with open("beta_calibrator.pkl", "rb") as f:
            beta_cal = pickle.load(f)
    except:
        pass
    return models, features, lookup, models_dec, models_o25, features_dec, features_o25, meta_model, meta_feats, beta_cal, bfo_features

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
def get_card_ufc_com():
    """Scraper do UFC.com para obter o card do próximo evento"""
    try:
        # Encontrar o próximo evento UFC
        r = requests.get(
            "https://www.ufc.com/events",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=10
        )
        soup = BeautifulSoup(r.text, 'html.parser')

        # Encontrar link do próximo evento
        event_links = soup.find_all('a', href=lambda x: x and '/event/' in str(x))
        next_event_url = None
        next_event_nome = None

        for l in event_links:
            href = l.get('href', '')
            if '/event/' in href and 'ufc' in href.lower():
                if href.startswith('/'):
                    href = 'https://www.ufc.com' + href
                next_event_url = href
                next_event_nome = l.text.strip()
                break

        if not next_event_url:
            return []

        # Aceder à página do evento
        r2 = requests.get(
            next_event_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=10
        )
        soup2 = BeautifulSoup(r2.text, 'html.parser')

        # Extrair data
        import re
        date_match = re.search(r'(\w+ \d+, \d{4})', r2.text)
        data_dt = datetime.strptime(date_match.group(1), "%B %d, %Y") if date_match else datetime.now()

        # Extrair local
        local_str = ""
        local_tag = soup2.find('div', class_=lambda x: x and 'location' in str(x).lower())
        if local_tag:
            local_str = local_tag.text.strip()

        # Extrair fighters
        fighter_links = soup2.find_all('a', href=lambda x: x and '/athlete/' in str(x))
        fighters_clean = []
        seen_hrefs = set()
        for l in fighter_links:
            href = l.get('href', '')
            name = ' '.join(l.text.strip().split())
            if not name or href in seen_hrefs:
                continue
            seen_hrefs.add(href)
            fighters_clean.append(name)

        # Criar pares
        card = [(fighters_clean[i], fighters_clean[i+1]) 
                for i in range(0, len(fighters_clean)-1, 2)]

        if not card:
            return []

        # Nome do evento — procurar texto curto que seja o nome real
        nome_evento = None
        for l in event_links:
            txt = l.text.strip()
            href = l.get('href', '')
            if next_event_url.endswith(href) or href in next_event_url:
                # Preferir textos curtos como "Song vs Figueiredo"
                if txt and 'vs' in txt.lower() and len(txt) < 50:
                    nome_evento = f"UFC Fight Night: {txt}"
                    break
        if not nome_evento:
            # Extrair do URL: /event/ufc-fight-night-may-30-2026 → UFC Fight Night May 30 2026
            slug = next_event_url.split('/event/')[-1]
            parts = slug.replace('-', ' ').title()
            nome_evento = parts

        return [{
            "nome": nome_evento,
            "data": data_dt,
            "local": local_str,
            "card": card,
        }]
    except Exception as e:
        return []

def get_card_ufcstats():
    # Tentar UFC.com primeiro (mais fiável)
    try:
        eventos = get_card_ufc_com()
        if eventos:
            return eventos
    except:
        pass

    # Tentar UFCStats
    try:
        url = "http://www.ufcstats.com/statistics/events/upcoming"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        eventos = []
        rows = soup.find_all("tr")
        for e in rows:
            link = e.find("a")
            if not link:
                continue
            nome_evento = link.text.strip()
            if not nome_evento or "UFC" not in nome_evento:
                continue
            url_evento  = link["href"]
            cells       = e.find_all("td")
            cell0_text  = cells[0].get_text(separator=" ").strip() if cells else ""
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
        if eventos:
            return eventos
    except:
        pass
    # Fallback: Odds API
    try:
        return get_card_from_odds_api()
    except:
        return []

def get_card_from_odds_api():
    from collections import defaultdict
    r = requests.get(
        f"https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
        f"?regions=eu&markets=h2h&oddsFormat=decimal&apiKey={API_KEY}",
        timeout=10
    )
    if r.status_code != 200:
        return []
    data = r.json()

    # Identificar eventos UFC pelo nome dos grupos
    # Usar o evento da Odds API que tem mais combates num mesmo dia
    eventos_por_data = defaultdict(list)
    for c in data:
        data_str = c["commence_time"][:10]
        f1 = normalizar_nome(c["home_team"])
        f2 = normalizar_nome(c["away_team"])
        # Filtrar eventos claramente não-UFC (menos fighters conhecidos)
        eventos_por_data[data_str].append((f1, f2))

    eventos = []
    today = datetime.now()

    for data_str, combates in sorted(eventos_por_data.items()):
        data_dt = datetime.strptime(data_str, "%Y-%m-%d")
        dias = (data_dt - today).days
        if dias < -1 or dias > 14:
            continue
        # UFC events têm tipicamente 10-15 combates
        # Pegar só o maior evento do dia (mais combates = mais provável ser UFC)
        if len(combates) < 8:
            continue
        # Verificar se algum fighter está no lookup
        fighters_conhecidos = sum(
            1 for f1, f2 in combates
            if lookup[lookup['name'] == f1].shape[0] > 0 or
               lookup[lookup['name'] == f2].shape[0] > 0
        )
        if fighters_conhecidos < 3:
            continue
        nome = f"UFC Fight Night — {data_dt.strftime('%B %d, %Y')}"
        eventos.append({
            "nome": nome,
            "data": data_dt,
            "local": "",
            "card": combates,
        })

    return eventos

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


@st.cache_data(ttl=1800)
def get_polymarket_odds():
    """Busca odds UFC do Polymarket — apenas eventos não fechados"""
    import json
    try:
        markets = {}
        r = requests.get(
            "https://gamma-api.polymarket.com/events?tag_slug=ufc&limit=200&closed=false",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=10
        )
        if r.status_code != 200:
            return {}
        batch = r.json()
        for ev in batch:
            for m in ev.get('markets', []):
                outcomes_raw = m.get('outcomes', '[]')
                prices_raw = m.get('outcomePrices', '[]')
                try:
                    outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
                    prices = [float(p) for p in (json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw)]
                except:
                    continue
                if len(outcomes) != 2 or 'Yes' in outcomes or 'No' in outcomes:
                    continue
                if not prices or prices[0] in [0.0, 1.0] or prices[1] in [0.0, 1.0]:
                    continue
                key = tuple(sorted([outcomes[0].lower().split()[-1],
                                    outcomes[1].lower().split()[-1]]))
                vol = float(m.get('volume', 0) or 0)
                if key not in markets or vol > markets[key]['volume']:
                    markets[key] = {
                        'f1': outcomes[0], 'f2': outcomes[1],
                        'p1': prices[0], 'p2': prices[1],
                        'volume': vol,
                        'question': m.get('question', ''),
                    }
        return markets
    except:
        return {}
def match_polymarket(f1_name, f2_name, pm_markets):
    """Tenta fazer match de um combate com mercados do Polymarket"""
    # Tentar por último nome
    key = tuple(sorted([f1_name.lower().split()[-1], f2_name.lower().split()[-1]]))
    if key in pm_markets:
        m = pm_markets[key]
        # Garantir que p1 corresponde a f1
        if f1_name.lower().split()[-1] == m['f1'].lower().split()[-1]:
            return m['p1'], m['p2'], m['volume']
        else:
            return m['p2'], m['p1'], m['volume']
    return None, None, None

models, features, lookup, models_dec, models_o25, features_dec, features_o25, meta_model, meta_feats, beta_cal, bfo_features = load_data()
fighter_names = sorted(lookup["name"].dropna().tolist())

# ── FUNÇÕES ───────────────────────────────────────────────────────
def decimal_to_prob(odds):
    try:
        odds = float(odds)
        if np.isnan(odds) or odds <= 1.0: return None
        return 1 / odds
    except:
        return None

def calc_ev(odds, prob):
    """Expected Value por unidade apostada em odds decimais"""
    try:
        odds = float(odds)
        prob = float(prob)
        if odds <= 1.0 or prob <= 0:
            return None
        return round(odds * prob - 1, 4)
    except:
        return None


def calc_kelly(odds, prob, fraction=0.25):
    """Kelly Criterion (fractional) — % do bankroll a apostar"""
    try:
        odds = float(odds)
        prob = float(prob)
        if odds <= 1.0 or prob <= 0 or prob >= 1:
            return None
        b = odds - 1
        q = 1 - prob
        kelly = (prob * b - q) / b
        if kelly <= 0:
            return None
        return round(kelly * fraction, 4)  # Quarter Kelly por defeito
    except:
        return None

# ── HISTÓRICO (GitHub) ──────────────────────────────────────────
GITHUB_TOKEN = os.getenv("TOKEN_GITHUB", "")
GITHUB_REPO  = "jdanielbcosta/ufc-predictor"
HIST_FILE    = "historico.csv"

def gh_get_historico():
    """Descarrega o CSV de histórico do GitHub"""
    if not GITHUB_TOKEN:
        return pd.DataFrame()
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{HIST_FILE}"
    r = requests.get(url, headers={"Authorization": f"token {GITHUB_TOKEN}"}, timeout=10)
    if r.status_code == 200:
        content = base64.b64decode(r.json()["content"]).decode("utf-8")
        return pd.read_csv(io.StringIO(content))
    return pd.DataFrame()

def gh_save_historico(df):
    """Guarda o CSV de histórico no GitHub"""
    if not GITHUB_TOKEN:
        return False
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{HIST_FILE}"
    csv_content = df.to_csv(index=False)
    encoded = base64.b64encode(csv_content.encode()).decode()
    # Ver se ficheiro já existe (para obter SHA)
    r = requests.get(url, headers={"Authorization": f"token {GITHUB_TOKEN}"}, timeout=10)
    payload = {
        "message": "Update historico.csv",
        "content": encoded,
    }
    if r.status_code == 200:
        payload["sha"] = r.json()["sha"]
    r2 = requests.put(url, headers={"Authorization": f"token {GITHUB_TOKEN}"},
                      json=payload, timeout=10)
    return r2.status_code in [200, 201]

def gh_save_previsoes(combates, event_name):
    """Guarda as previsões antes do evento"""
    if not combates:
        return
    hist = gh_get_historico()
    rows = []
    for c in combates:
        rows.append({
            "event_name": event_name,
            "fighter_1": c["f1"],
            "fighter_2": c["f2"],
            "prob_f1": round(c["p1"], 4),
            "prob_f2": round(c["p2"], 4),
            "predicted_winner": c["f1"] if c["p1"] >= c["p2"] else c["f2"],
            "odds_f1": c.get("odds_f1", ""),
            "odds_f2": c.get("odds_f2", ""),
            "conviction": conviction_label(max(c["p1"], c["p2"]))[1],
            "actual_winner": "",
            "correct": "",
            "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d"),
        })
    new_rows = pd.DataFrame(rows)
    if not hist.empty:
        # Não duplicar combates já guardados
        existing = set(zip(hist["fighter_1"], hist["fighter_2"], hist["event_name"]))
        new_rows = new_rows[~new_rows.apply(
            lambda r: (r["fighter_1"], r["fighter_2"], r["event_name"]) in existing, axis=1)]
    combined = pd.concat([hist, new_rows], ignore_index=True) if not hist.empty else new_rows
    gh_save_historico(combined)

def gh_update_resultados():
    """Tenta actualizar resultados via UFCStats para combates sem resultado"""
    hist = gh_get_historico()
    if hist.empty:
        return hist
    sem_resultado = hist[hist["actual_winner"] == ""]
    if sem_resultado.empty:
        return hist
    # Scrape últimos resultados do UFCStats
    try:
        from bs4 import BeautifulSoup
        r = requests.get("http://ufcstats.com/statistics/events/completed?page=all",
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        # Pegar nos últimos 3 eventos
        event_links = [a["href"] for a in soup.select("a.b-link_style_black")[:3]]
        resultados = {}
        for ev_url in event_links:
            r2 = requests.get(ev_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            soup2 = BeautifulSoup(r2.text, "html.parser")
            for row in soup2.select("tr.b-fight-details__table-row"):
                cols = row.select("td")
                if len(cols) < 2:
                    continue
                fighters = row.select("a.b-link_style_black")
                if len(fighters) >= 2:
                    winner = fighters[0].text.strip()
                    loser  = fighters[1].text.strip()
                    resultados[f"{winner}_{loser}"] = winner
                    resultados[f"{loser}_{winner}"] = winner
        # Actualizar histórico
        updated = False
        for idx, row in hist.iterrows():
            if row["actual_winner"] != "":
                continue
            key1 = f"{row['fighter_1']}_{row['fighter_2']}"
            key2 = f"{row['fighter_2']}_{row['fighter_1']}"
            winner = resultados.get(key1) or resultados.get(key2)
            if winner:
                hist.at[idx, "actual_winner"] = winner
                hist.at[idx, "correct"] = "✅" if winner == row["predicted_winner"] else "❌"
                updated = True
        if updated:
            gh_save_historico(hist)
    except Exception as e:
        print(f"Erro a actualizar resultados: {e}")
    return hist

def get_historico():
    return gh_get_historico()

def gh_save_totals_odds():
    """Recolhe e guarda odds de over/under antes de cada evento"""
    if not API_KEY or not GITHUB_TOKEN:
        return
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds",
            params={"regions": "eu", "markets": "totals",
                    "oddsFormat": "decimal", "apiKey": API_KEY},
            timeout=15
        )
        if r.status_code != 200:
            return
        data = r.json()
        rows = []
        now = pd.Timestamp.now().strftime("%Y-%m-%d")
        for e in data:
            if not e.get('bookmakers'):
                continue
            for b in e['bookmakers']:
                for m in b.get('markets', []):
                    if m['key'] != 'totals':
                        continue
                    for outcome in m['outcomes']:
                        rows.append({
                            "fighter_1": e['home_team'],
                            "fighter_2": e['away_team'],
                            "bookmaker": b['title'],
                            "type": outcome['name'],
                            "point": outcome['point'],
                            "price": outcome['price'],
                            "fetched_at": now,
                            "commence_time": e.get('commence_time','')[:10],
                        })
        if not rows:
            return
        new_df = pd.DataFrame(rows)

        # Ler histórico existente
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/totals_odds_history.csv"
        r2 = requests.get(url, headers={"Authorization": f"token {GITHUB_TOKEN}"}, timeout=10)
        if r2.status_code == 200:
            content = base64.b64decode(r2.json()["content"]).decode("utf-8")
            existing = pd.read_csv(io.StringIO(content))
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["fighter_1","fighter_2","bookmaker","type","point","fetched_at"],
                keep="first"
            )
            sha = r2.json()["sha"]
        else:
            combined = new_df
            sha = None

        csv_content = combined.to_csv(index=False)
        encoded = base64.b64encode(csv_content.encode()).decode()
        payload = {"message": f"Update totals odds {now}", "content": encoded}
        if sha:
            payload["sha"] = sha
        requests.put(url, headers={"Authorization": f"token {GITHUB_TOKEN}"},
                     json=payload, timeout=30)
    except Exception:
        pass



def gh_save_moneyline_odds(combates_api):
    """Guarda snapshot diário das odds de moneyline no GitHub"""
    if not GITHUB_TOKEN or not combates_api:
        return
    try:
        rows = []
        now = pd.Timestamp.now().strftime("%Y-%m-%d")
        for c in combates_api:
            f1 = normalizar_nome(c["home_team"])
            f2 = normalizar_nome(c["away_team"])
            odds_f1_list, odds_f2_list = [], []
            for bm in c.get("bookmakers", []):
                for mkt in bm.get("markets", []):
                    if mkt["key"] == "h2h":
                        for outcome in mkt["outcomes"]:
                            nome = normalizar_nome(outcome["name"])
                            if nome == f1:
                                odds_f1_list.append(outcome["price"])
                            elif nome == f2:
                                odds_f2_list.append(outcome["price"])
            if odds_f1_list and odds_f2_list:
                rows.append({
                    "fighter_1": f1,
                    "fighter_2": f2,
                    "odds_1": round(float(np.median(odds_f1_list)), 3),
                    "odds_2": round(float(np.median(odds_f2_list)), 3),
                    "fetched_at": now,
                    "commence_time": c.get("commence_time", "")[:10],
                })
        if not rows:
            return

        new_df = pd.DataFrame(rows)
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/moneyline_odds_history.csv"
        r = requests.get(url, headers={"Authorization": f"token {GITHUB_TOKEN}"}, timeout=10)
        if r.status_code == 200:
            existing = pd.read_csv(io.StringIO(base64.b64decode(r.json()["content"]).decode("utf-8")))
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["fighter_1","fighter_2","fetched_at"], keep="first")
            sha = r.json()["sha"]
        else:
            combined = new_df
            sha = None

        csv_content = combined.to_csv(index=False)
        encoded = base64.b64encode(csv_content.encode()).decode()
        payload = {"message": f"Update moneyline odds {now}", "content": encoded}
        if sha:
            payload["sha"] = sha
        requests.put(url, headers={"Authorization": f"token {GITHUB_TOKEN}"},
                     json=payload, timeout=30)
    except Exception as e:
        print(f"Erro a guardar moneyline odds: {e}")

@st.cache_data(ttl=3600)
def gh_get_moneyline_history():
    """Carrega histórico de odds de moneyline do GitHub"""
    if not GITHUB_TOKEN:
        return pd.DataFrame()
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/moneyline_odds_history.csv"
        r = requests.get(url, headers={"Authorization": f"token {GITHUB_TOKEN}"}, timeout=10)
        if r.status_code == 200:
            content = base64.b64decode(r.json()["content"]).decode("utf-8")
            return pd.read_csv(io.StringIO(content))
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def calc_bookmaker_disagreement(f1_name, f2_name, combates_api):
    """Calcula disagreement entre bookmakers para um combate"""
    f1_lower = f1_name.lower()
    f2_lower = f2_name.lower()

    for c in combates_api:
        h = normalizar_nome(c["home_team"]).lower()
        a = normalizar_nome(c["away_team"]).lower()
        if not ((h == f1_lower and a == f2_lower) or (h == f2_lower and a == f1_lower)):
            continue

        odds_f1, odds_f2 = [], []
        for bm in c.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt["key"] == "h2h":
                    for o in mkt["outcomes"]:
                        nome = normalizar_nome(o["name"]).lower()
                        if nome == f1_lower:
                            odds_f1.append(o["price"])
                        elif nome == f2_lower:
                            odds_f2.append(o["price"])

        if len(odds_f1) < 3 or len(odds_f2) < 3:
            return None

        spread_f1 = max(odds_f1) - min(odds_f1)
        spread_f2 = max(odds_f2) - min(odds_f2)

        return {
            'f1_spread': round(spread_f1, 3),
            'f2_spread': round(spread_f2, 3),
            'f1_min': min(odds_f1), 'f1_max': max(odds_f1),
            'f2_min': min(odds_f2), 'f2_max': max(odds_f2),
            'n_bookmakers': len(odds_f1),
        }
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

    # rest_diff: dias de descanso (dias_inativo é proxy)
    r_rest = float(r.get("dias_inativo", 180) or 180)
    b_rest = float(b.get("dias_inativo", 180) or 180)
    feats["rest_diff"] = r_rest - b_rest

    # ufc_age_diff: anos de carreira na UFC
    r_ufc_debut = r.get("ufc_debut")
    b_ufc_debut = b.get("ufc_debut")
    try:
        if r_ufc_debut and b_ufc_debut:
            import datetime
            today = datetime.date.today()
            r_ufc_age = (today - pd.to_datetime(r_ufc_debut).date()).days / 365.25
            b_ufc_age = (today - pd.to_datetime(b_ufc_debut).date()).days / 365.25
            feats["ufc_age_diff"] = r_ufc_age - b_ufc_age
        else:
            feats["ufc_age_diff"] = 0.0
    except:
        feats["ufc_age_diff"] = 0.0

    # win_rate features
    feats["win_rate_l5_diff"] = sd("win_rate_l5")
    feats["win_rate_l3_diff"] = sd("win_rate_l3")

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

    # Injectar BFO movement features se disponíveis
    if bfo_features is not None:
        try:
            r_lower = f1_name.lower()
            b_lower = f2_name.lower()
            fk = '_'.join(sorted([r_lower, b_lower]))
            bfo_row = bfo_features[bfo_features['fight_key'] == fk]
            if not bfo_row.empty:
                bfo_row = bfo_row.iloc[0]
                # Alinhar com R/B
                if bfo_row.get('f1_lower', '') == r_lower:
                    X['r_clv']           = bfo_row.get('f1_clv', 0) or 0
                    X['b_clv']           = bfo_row.get('f2_clv', 0) or 0
                    X['r_movement_pct']  = bfo_row.get('f1_movement_pct', 0) or 0
                    X['b_movement_pct']  = bfo_row.get('f2_movement_pct', 0) or 0
                    X['r_late']          = bfo_row.get('f1_late', 0) or 0
                    X['b_late']          = bfo_row.get('f2_late', 0) or 0
                    X['r_steam']         = bfo_row.get('f1_steam', 0) or 0
                    X['b_steam']         = bfo_row.get('f2_steam', 0) or 0
                    X['r_vol']           = bfo_row.get('f1_vol', 0) or 0
                    X['b_vol']           = bfo_row.get('f2_vol', 0) or 0
                else:
                    X['r_clv']           = bfo_row.get('f2_clv', 0) or 0
                    X['b_clv']           = bfo_row.get('f1_clv', 0) or 0
                    X['r_movement_pct']  = bfo_row.get('f2_movement_pct', 0) or 0
                    X['b_movement_pct']  = bfo_row.get('f1_movement_pct', 0) or 0
                    X['r_late']          = bfo_row.get('f2_late', 0) or 0
                    X['b_late']          = bfo_row.get('f1_late', 0) or 0
                    X['r_steam']         = bfo_row.get('f2_steam', 0) or 0
                    X['b_steam']         = bfo_row.get('f1_steam', 0) or 0
                    X['r_vol']           = bfo_row.get('f2_vol', 0) or 0
                    X['b_vol']           = bfo_row.get('f1_vol', 0) or 0
                X['clv_diff_rb']        = X['r_clv'] - X['b_clv']
                X['movement_diff_rb']   = X['r_movement_pct'] - X['b_movement_pct']
                X['late_diff_rb']       = X['r_late'] - X['b_late']
                X['steam_diff_rb']      = X['r_steam'] - X['b_steam']
                X['vol_diff_rb']        = X['r_vol'] - X['b_vol']
        except:
            pass

    X = X.reindex(columns=features, fill_value=0).fillna(0)
    probs = np.array([m.predict_proba(X)[0][1] for m in models.values()])
    prob_raw = probs.mean()
    # Aplicar beta calibration se disponível
    if beta_cal is not None:
        try:
            prob_winner = float(beta_cal.predict(np.array([[prob_raw]]))[0])
        except:
            prob_winner = prob_raw
    else:
        prob_winner = prob_raw

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

    # Meta-modelo
    meta_score = None
    if meta_model is not None and meta_feats is not None:
        try:
            odds_prob = (decimal_to_prob(odds_f1) or 0.5) - (decimal_to_prob(odds_f2) or 0.5)
            confidence = abs(prob_winner - 0.5)
            ensemble_vs_odds = prob_winner - (odds_prob / 2 + 0.5)
            meta_X = pd.DataFrame([{
                'ensemble_prob': prob_winner,
                'confidence': confidence,
                'odds_prob': odds_prob,
                'ensemble_vs_odds': ensemble_vs_odds,
                'rest_diff': 0,
                'clv_diff': float(X.get('clv_diff_rb', 0) or 0),
                'odds_magnitude': abs(odds_prob),
            }])[meta_feats]
            meta_score = float(meta_model.predict_proba(meta_X)[0][1])
        except:
            pass

    # Ensemble disagreement
    ensemble_std = float(np.std(probs)) if len(probs) > 1 else None

    return prob_winner, 1 - prob_winner, r, b, prob_decision, prob_over25, meta_score, ensemble_std

def conviction_label(prob):
    if prob >= 0.75: return 0, "HIGH CONVICTION 83%", "#D4AF37", "high"
    if prob >= 0.70: return 1, "MODERATE CONVICTION 79%", "#C0A030", "moderate"
    if prob >= 0.60: return 2, "SLIGHT FAVOURITE 74%", "#888888", "slight"
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
@st.cache_data(ttl=86400)
def get_fighter_photo(name):
    """Busca foto do lutador na ufc.com"""
    try:
        slug = name.lower().replace(' ', '-')
        url = f"https://www.ufc.com/athlete/{slug}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        if r.status_code != 200:
            return None
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, 'html.parser')
        img = soup.find('img', class_='hero-profile__image')
        if not img:
            return None
        return img.get('src')
    except:
        return None



def sugerir_parlay(combates, threshold=0.25, max_legs=3):
    """Sugere a melhor parlay para um evento baseada em EV positivo"""
    candidatos = []
    for c in combates:
        if abs(c["prob_fav"] - 0.5) < threshold:
            continue
        # Fighter favorito e odds
        if c["p1"] >= c["p2"]:
            prob = c["p1"]
            odds = c.get("odds_f1")
            fighter = c["f1"]
        else:
            prob = c["p2"]
            odds = c.get("odds_f2")
            fighter = c["f2"]
        if not odds or odds <= 1.0:
            continue
        ev = prob * odds - 1
        if ev <= 0:
            continue
        candidatos.append({
            "fighter": fighter,
            "prob": prob,
            "odds": odds,
            "ev": ev,
            "conf": abs(c["prob_fav"] - 0.5),
        })

    if len(candidatos) < 2:
        return None

    # Ordenar por EV e seleccionar melhores legs
    candidatos.sort(key=lambda x: -x["ev"])
    legs_3 = candidatos[:min(3, max_legs)]
    legs_2 = candidatos[:2]

    def calc_parlay(legs):
        odds_total = 1.0
        prob_total = 1.0
        for l in legs:
            odds_total *= l["odds"]
            prob_total *= l["prob"]
        ev_total = prob_total * odds_total - 1
        return round(odds_total, 2), round(prob_total * 100, 1), round(ev_total * 100, 1)

    odds_3, prob_3, ev_3 = calc_parlay(legs_3)
    odds_2, prob_2, ev_2 = calc_parlay(legs_2)

    return {
        "legs_3": legs_3, "odds_3": odds_3, "prob_3": prob_3, "ev_3": ev_3,
        "legs_2": legs_2, "odds_2": odds_2, "prob_2": prob_2, "ev_2": ev_2,
    }

def render_parlay(parlay, aposta=10):
    """Renderiza a sugestão de parlay em HTML"""
    if not parlay:
        return ""

    def render_legs(legs, odds_total, prob_total, ev_total, aposta):
        retorno = round(aposta * (odds_total - 1), 2)
        legs_html = ""
        for l in legs:
            ev_color = "#22c55e" if l["ev"] > 0.05 else "#fbbf24"
            legs_html += (
                f'<div style="display:flex; justify-content:space-between; '
                f'padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.05);">'
                f'<span style="color:var(--text); font-weight:600;">{l["fighter"]}</span>'
                f'<span style="color:var(--muted); font-size:0.8rem;">'
                f'Odds {l["odds"]:.2f} · Prob {l["prob"]:.0%} · '
                f'<span style="color:{ev_color};">EV {l["ev"]*100:+.1f}%</span></span>'
                f'</div>'
            )
        prob_color = "#22c55e" if prob_total >= 50 else "#fbbf24"
        return (
            f'<div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); '
            f'border-radius:8px; padding:12px; margin-bottom:8px;">'
            f'<div style="display:flex; justify-content:space-between; margin-bottom:8px;">'
            f'<span style="font-weight:700; color:var(--text);">{len(legs)}-leg Parlay</span>'
            f'<span style="color:#a78bfa; font-weight:700;">Odds {odds_total:.2f}x</span>'
            f'</div>'
            f'{legs_html}'
            f'<div style="display:flex; justify-content:space-between; margin-top:8px; '
            f'padding-top:8px; border-top:1px solid rgba(255,255,255,0.08);">'
            f'<span style="color:var(--muted); font-size:0.8rem;">'
            f'Prob. ganhar: <span style="color:{prob_color}; font-weight:700;">{prob_total:.1f}%</span> · '
            f'EV: <span style="color:#22c55e; font-weight:700;">{ev_total:+.1f}%</span></span>'
            f'<span style="color:#22c55e; font-weight:700;">'
            f'€{aposta} → €{retorno + aposta:.2f}</span>'
            f'</div>'
            f'</div>'
        )

    html_3 = render_legs(parlay["legs_3"], parlay["odds_3"], parlay["prob_3"], parlay["ev_3"], aposta) if len(parlay["legs_3"]) >= 2 else ""
    html_2 = render_legs(parlay["legs_2"], parlay["odds_2"], parlay["prob_2"], parlay["ev_2"], aposta)

    return (
        f'<div style="margin-top:16px; padding:16px; '
        f'background:rgba(167,139,250,0.05); '
        f'border:1px solid rgba(167,139,250,0.2); border-radius:12px;">'
        f'<div style="font-weight:700; color:#a78bfa; margin-bottom:12px; font-size:1rem;">'
        f'🎯 Parlay Sugerida</div>'
        f'{html_3}'
        f'{"<div style=\'margin-top:8px; font-size:0.75rem; color:var(--muted);\'>Versão conservadora (2 legs):</div>" if html_3 else ""}'
        f'{html_2 if html_3 else html_2}'
        f'<div style="font-size:0.72rem; color:var(--muted); margin-top:8px;">'
        f'⚠️ Apenas para fins informativos. Aposte com responsabilidade.</div>'
        f'</div>'
    )

def render_fight_card(c, odds_history=None):
    fav    = c["f1"] if c["p1"] >= c["p2"] else c["f2"]
    em, lbl, color, level = conviction_label(c["prob_fav"])
    meta_score = c.get("meta_score")
    meta_html = ""
    # Consensus badge — baixo disagreement + alta confiança
    consensus_html = ""
    ensemble_std = c.get("ensemble_std")
    if ensemble_std is not None and ensemble_std <= 0.04 and c["prob_fav"] >= 0.65:
        consensus_html = (
            f'<span style="background:rgba(96,165,250,0.15); color:#60a5fa; '
            f'border:1px solid rgba(96,165,250,0.4); border-radius:4px; '
            f'padding:2px 8px; font-size:0.72rem; font-weight:700; '
            f'letter-spacing:0.05em;">'
            f'🤝 CONSENSUS {(1-ensemble_std/0.04)*100:.0f}%'
            f'</span>'
        )
    if meta_score is not None and meta_score >= 0.75:
        meta_html = (
            f'<span style="background:rgba(34,197,94,0.15); color:#22c55e; '
            f'border:1px solid rgba(34,197,94,0.4); border-radius:4px; '
            f'padding:2px 8px; font-size:0.72rem; font-weight:700; '
            f'letter-spacing:0.05em;">'
            f'✅ META-MODEL CONFIRMED {meta_score:.0%}'
            f'</span>'
        )
    elif meta_score is not None and meta_score >= 0.65:
        meta_html = (
            f'<span style="background:rgba(251,191,36,0.12); color:#fbbf24; '
            f'border:1px solid rgba(251,191,36,0.3); border-radius:4px; '
            f'padding:2px 8px; font-size:0.72rem; font-weight:700; '
            f'letter-spacing:0.05em;">'
            f'⚡ META-MODEL {meta_score:.0%}'
            f'</span>'
        )
    card_class = {"high": "fight-card-high", "moderate": "fight-card-med",
                  "slight": "fight-card-low", "close": "fight-card-draw"}[level]
    prob_decision = c.get("prob_decision")
    prob_over25   = c.get("prob_over25")

    p1_pct = c["p1"] * 100
    p2_pct = c["p2"] * 100

    # Pre-compute name classes to avoid quote conflicts in f-string
    f1_class = "fav" if c["p1"] >= c["p2"] else "plain-name"
    f2_class = "fav" if c["p2"] > c["p1"] else "plain-name"
    f1_name  = c["f1"]
    f2_name  = c["f2"]

    # EV calculation
    ev_f1_html = ""
    ev_f2_html = ""
    if c["odds_f1"] and c["p1"]:
        ev = calc_ev(c["odds_f1"], c["p1"])
        kelly = calc_kelly(c["odds_f1"], c["p1"])
        ev_parts = []
        if ev is not None:
            cls = "edge-pos" if ev > 0.05 else ("edge-neg" if ev < -0.05 else "edge-neu")
            ev_parts.append(f'<span class="{cls}" style="font-size:0.72rem;">EV {ev*100:+.1f}%</span>')
        if kelly is not None:
            cls = "edge-pos" if kelly >= 0.02 else "edge-neu"
            ev_parts.append(f'<span class="{cls}" style="font-size:0.72rem;">Kelly {kelly*100:.1f}%</span>')
        ev_f1_html = " ".join(ev_parts)
    if c["odds_f2"] and c["p2"]:
        ev = calc_ev(c["odds_f2"], c["p2"])
        kelly = calc_kelly(c["odds_f2"], c["p2"])
        ev_parts = []
        if ev is not None:
            cls = "edge-pos" if ev > 0.05 else ("edge-neg" if ev < -0.05 else "edge-neu")
            ev_parts.append(f'<span class="{cls}" style="font-size:0.72rem;">EV {ev*100:+.1f}%</span>')
        if kelly is not None:
            cls = "edge-pos" if kelly >= 0.02 else "edge-neu"
            ev_parts.append(f'<span class="{cls}" style="font-size:0.72rem;">Kelly {kelly*100:.1f}%</span>')
        ev_f2_html = " ".join(ev_parts)

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
    # Late movement warning — mercado a apostar contra o favorito
    prob_fav_c = c.get("prob_fav", 0.5)
    fav_is_red_c = prob_fav_c == c.get("p1", 0)
    r_late_c = c.get("r_late", 0) or 0
    b_late_c = c.get("b_late", 0) or 0
    fav_late_c = r_late_c if (c.get("p1", 0) >= c.get("p2", 0)) else -r_late_c
    if fav_late_c > 0.05:
        warnings.append(
            f'<span style="background:rgba(232,37,63,0.15); color:#e8253f; '
            f'border:1px solid rgba(232,37,63,0.4); border-radius:4px; '
            f'padding:2px 8px; font-size:0.72rem; font-weight:700; '
            f'letter-spacing:0.05em;">'
            f'⚠️ MARKET BETTING AGAINST FAVOURITE'
            f'</span>'
        )
    warnings_html = " ".join(warnings)
    if consensus_html:
        warnings_html = (consensus_html + " " + warnings_html).strip()

    # Bookmaker disagreement
    disagreement = c.get("disagreement")
    disagree_html = ""
    if disagreement:
        f1_sp = disagreement["f1_spread"]
        f2_sp = disagreement["f2_spread"]
        max_sp = max(f1_sp, f2_sp)
        if max_sp >= 0.30:
            bigger = c["f1"] if f1_sp >= f2_sp else c["f2"]
            sp = f1_sp if f1_sp >= f2_sp else f2_sp
            mn = disagreement["f1_min"] if f1_sp >= f2_sp else disagreement["f2_min"]
            mx = disagreement["f1_max"] if f1_sp >= f2_sp else disagreement["f2_max"]
            n_bm = disagreement["n_bookmakers"]
            disagree_html = (
                f'<span style="background:rgba(251,191,36,0.12); color:#fbbf24; '
                f'border:1px solid rgba(251,191,36,0.3); border-radius:4px; '
                f'padding:2px 8px; font-size:0.72rem; font-weight:700; '
                f'letter-spacing:0.05em;">'
                f'📊 MARKET SPLIT: {bigger} {mn:.2f}–{mx:.2f} across {n_bm} books (Δ{sp:.2f})'
                f'</span>'
            )
    if disagree_html:
        warnings_html = (disagree_html + " " + warnings_html).strip()

    # Polymarket odds
    pm_p1 = c.get("pm_p1")
    pm_p2 = c.get("pm_p2")
    pm_vol = c.get("pm_vol")
    pm_alert_html = ""
    if pm_p1 and pm_p2:
        # Verificar discrepância com bookmakers
        discrepancy_html = ""
        if c["odds_f1"] and c["odds_f2"]:
            bk_p1 = decimal_to_prob(c["odds_f1"])
            bk_p2 = decimal_to_prob(c["odds_f2"])
            diff1 = abs(pm_p1 - bk_p1)
            diff2 = abs(pm_p2 - bk_p2)
            max_diff = max(diff1, diff2)
            if max_diff >= 0.10:
                bigger_f = c["f1"] if diff1 >= diff2 else c["f2"]
                pm_prob  = pm_p1 if diff1 >= diff2 else pm_p2
                bk_prob  = bk_p1 if diff1 >= diff2 else bk_p2
                direction = "↑" if pm_prob > bk_prob else "↓"
                discrepancy_html = (
                    f' · <span style="color:#ffa500; font-weight:700;">'
                    f'⚠️ {bigger_f} {direction} Δ{max_diff:.0%}</span>'
                )
        pm_alert_html = (
            f'<span style="background:rgba(139,92,246,0.12); color:#a78bfa; '
            f'border:1px solid rgba(139,92,246,0.3); border-radius:4px; '
            f'padding:2px 8px; font-size:0.72rem; font-weight:700; '
            f'letter-spacing:0.05em;">'
            f'🔮 Polymarket: {c["f1"]} {pm_p1:.0%} · {c["f2"]} {pm_p2:.0%} '
            f'· Vol ${pm_vol:,.0f}{discrepancy_html}'
            f'</span>'
        )
    if pm_alert_html:
        warnings_html = (pm_alert_html + " " + warnings_html).strip()

    # Histórico de odds (sparkline)
    odds_sparkline_html = ""
    if odds_history is not None and not odds_history.empty:
        f1_lower = c["f1"].lower()
        f2_lower = c["f2"].lower()
        hist = odds_history[
            (odds_history["fighter_1"].str.lower() == f1_lower) &
            (odds_history["fighter_2"].str.lower() == f2_lower)
        ].sort_values("fetched_at")

        if not hist.empty and len(hist) >= 2:
            o1_vals = hist["odds_1"].tolist()
            o2_vals = hist["odds_2"].tolist()
            o1_open = o1_vals[0]
            o1_curr = o1_vals[-1]
            o2_open = o2_vals[0]
            o2_curr = o2_vals[-1]
            chg1 = round((o1_curr/o1_open - 1)*100, 1)
            chg2 = round((o2_curr/o2_open - 1)*100, 1)
            col1 = "#22c55e" if chg1 > 0 else "#e8253f"
            col2 = "#22c55e" if chg2 > 0 else "#e8253f"
            sign1 = "+" if chg1 > 0 else ""
            sign2 = "+" if chg2 > 0 else ""

            # Mini sparkline SVG para f1
            def make_sparkline(vals, color):
                if len(vals) < 2:
                    return ""
                mn, mx = min(vals), max(vals)
                rng = mx - mn if mx != mn else 1
                w, h = 60, 20
                pts = []
                for i, v in enumerate(vals):
                    x = int(i / (len(vals)-1) * w)
                    y = int(h - (v - mn) / rng * h)
                    pts.append(f"{x},{y}")
                polyline = " ".join(pts)
                return (f'<svg width="{w}" height="{h}" style="vertical-align:middle;">' 
                        f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="1.5"/>' 
                        f'</svg>')

            sp1 = make_sparkline(o1_vals, col1)
            sp2 = make_sparkline(o2_vals, col2)

            odds_sparkline_html = (
                f'<div style="display:flex; justify-content:space-between; align-items:center; '
                f'margin-top:6px; font-size:0.75rem; color:var(--muted);">'
                f'<div style="display:flex; align-items:center; gap:6px;">'
                f'<span style="color:var(--text);">{c["f1"]}</span>'
                f'<span style="color:var(--muted);">Open <strong style="color:var(--text);">{o1_open:.2f}</strong></span>'
                f'→ <strong style="color:var(--text);">{o1_curr:.2f}</strong>'
                f'<span style="color:{col1}; font-weight:700;">{sign1}{chg1}%</span>'
                f'{sp1}'
                f'</div>'
                f'<div style="display:flex; align-items:center; gap:6px;">'
                f'{sp2}'
                f'<span style="color:{col2}; font-weight:700;">{sign2}{chg2}%</span>'
                f'<strong style="color:var(--text);">{o2_curr:.2f}</strong>'
                f'→ <span style="color:var(--muted);">Open <strong style="color:var(--text);">{o2_open:.2f}</strong></span>'
                f'<span style="color:var(--text);">{c["f2"]}</span>'
                f'</div>'
                f'</div>'
            )

    # Buscar fotos
    photo_f1 = get_fighter_photo(f1_name)
    photo_f2 = get_fighter_photo(f2_name)

    # F1 (esquerda): sem espelhar | F2 (direita): sempre espelhado para o centro
    photo_f1_html = (
        f'<img src="{photo_f1}" style="height:140px; object-fit:contain; '
        f'filter:drop-shadow(0 4px 12px rgba(200,16,46,0.3));">'
        if photo_f1 else
        f'<div style="height:140px; width:90px; display:flex; align-items:center; '
        f'justify-content:center; font-size:2.5rem;">🥊</div>'
    )
    photo_f2_html = (
        f'<img src="{photo_f2}" style="height:140px; object-fit:contain; '
        f'transform:scaleX(-1); filter:drop-shadow(0 4px 12px rgba(26,108,255,0.3));">'
        if photo_f2 else
        f'<div style="height:140px; width:90px; display:flex; align-items:center; '
        f'justify-content:center; font-size:2.5rem;">🥊</div>'
    )

    combined_warnings = ' '.join(filter(None, [meta_html, warnings_html]))
    warnings_div = ('<div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap;">' + combined_warnings + '</div>') if combined_warnings else ''
    html = (
        f'<div class="fight-card {card_class}" style="overflow:hidden;">'
        f'  <div style="display:flex; align-items:stretch; gap:0;">'
        f'    <!-- Foto F1 -->'
        f'    <div style="width:100px; display:flex; align-items:flex-end; justify-content:center;'
        f'         padding-bottom:0; flex-shrink:0;">'
        f'      {photo_f1_html}'
        f'    </div>'
        f'    <!-- Conteúdo central -->'
        f'    <div style="flex:1; padding:14px 12px 12px;">'
        f'      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">'
        f'        <div class="fight-names">'
        f'          <span class="{f1_class}">{f1_name}</span>'
        f'          <span class="vs">VS</span>'
        f'          <span class="{f2_class}">{f2_name}</span>'
        f'        </div>'
        f'        <div class="conviction-badge conv-{level}">{em} {lbl}</div>'
        f'      </div>'
        f'      <div class="prob-bar-wrap">'
        f'        <div class="prob-bar-labels">'
        f'          <span><span class="name-left">{f1_name}</span>'
        f'                <span class="pct-left"> {p1_str}</span></span>'
        f'          <span><span class="pct-right">{p2_str} </span>'
        f'                <span class="name-right">{f2_name}</span></span>'
        f'        </div>'
        f'        <div class="prob-bar-outer">'
        f'          <div style="height:100%; display:flex;">'
        f'            <div style="width:{p1_w}; background:linear-gradient(90deg,#C8102E,#e8253f);'
        f'                 border-radius:999px 0 0 999px;"></div>'
        f'            <div style="width:{p2_w}; background:linear-gradient(90deg,#1a6cff,#60a5fa);'
        f'                 border-radius:0 999px 999px 0; margin-left:auto;"></div>'
        f'          </div>'
        f'        </div>'
        f'      </div>'
        f'      <div class="odds-row">'
        f'        <div>'
        f'          <span class="odds-chip">&#128202; Odds <span class="ov">{odds_f1_str}</span></span>'
        f'          <span class="odds-chip">Market <span class="ov">{mkt_f1_str}</span></span>'
        f'          {edge_f1_html}'
        f'          {ev_f1_html}'
        f'        </div>'
        f'        <div style="font-size:0.75rem; color:var(--muted);">Favourite:'
        f'          <strong style="color:var(--gold);">{fav}</strong>'
        f'        </div>'
        f'        <div>'
        f'          {edge_f2_html}'
        f'          <span class="odds-chip">Market <span class="ov">{mkt_f2_str}</span></span>'
        f'          <span class="odds-chip">&#128202; Odds <span class="ov">{odds_f2_str}</span></span>'
        f'        </div>'
        f'      </div>'
        f'      {warnings_div}'
        f'      {odds_sparkline_html}'
        f'    </div>'
        f'    <!-- Foto F2 -->'
        f'    <div style="width:100px; display:flex; align-items:flex-end; justify-content:center;'
        f'         padding-bottom:0; flex-shrink:0;">'
        f'      {photo_f2_html}'
        f'    </div>'
        f'  </div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

# ── INIT ──────────────────────────────────────────────────────────
st.set_page_config(page_title="UFC Fight Predictor", page_icon="🥊", layout="wide")
if not API_KEY:
    st.warning("ODDS_API_KEY is not set. Live odds features will be unavailable.")
inject_css()

# ── HEADER ────────────────────────────────────────────────────────
st.markdown("""
<div class="ufc-header">
  <div>
    <div class="ufc-logo-text">🥊 UFC FIGHT PREDICTOR</div>
    <div class="ufc-subtitle">Ensemble v6 · Powered by Machine Learning</div>
  </div>
</div>
<div style="margin: 10px 0 22px;">
  <span class="badge-stat">🎯 Accuracy <span>69.82%</span></span>
  <span class="badge-stat">🤖 Models <span>5 ensemble</span></span>
  <span class="badge-stat">⚡ High Conviction <span>83.5% acc</span></span>
  <span class="badge-stat">🎯 Moderate <span>79.3% acc</span></span>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📅  UPCOMING EVENTS", "🔍  PREDICT A FIGHT", "📋  HISTORY"])

# ── TAB 1 ─────────────────────────────────────────────────────────
with tab1:
    col_t, col_r = st.columns([5, 1])
    with col_t:
        st.markdown("""
        <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.6rem;
             font-weight:800; letter-spacing:0.04em; color:#ffffff; margin-bottom:4px;">
          UPCOMING FIGHTS
        </div>
        """, unsafe_allow_html=True)

    with st.expander("ℹ️ How to read EV (Expected Value)"):
        st.markdown("""
        **EV (Expected Value)** tells you whether a bet is mathematically profitable based on the model's probability vs the bookmaker's odds.

        **Formula:** `EV = odds × model_probability - 1`

        **How to interpret:**
        - 🟢 **EV > 0** → positive expected value. The model thinks this bet is worth taking.
        - 🔴 **EV < 0** → negative expected value. The bookmaker has an edge.
        - The higher the EV, the more attractive the bet.

        **Example:**
        > Bookmaker offers **2.50** odds on Fighter A. The model gives Fighter A a **50%** chance of winning.
        > EV = 2.50 × 0.50 − 1 = **+0.25** → For every €1 bet, you expect to profit €0.25 on average.

        > Same odds **2.50**, but model gives only **35%** chance.
        > EV = 2.50 × 0.35 − 1 = **−0.125** → You expect to lose €0.125 per €1 bet on average.

        ⚠️ EV is a long-run statistical concept. A positive EV bet can still lose — it means the bet is profitable *on average* over many bets.
        """)
    with col_r:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Loading card and odds..."):
        eventos_ufc  = get_card_ufcstats()
        combates_api = get_upcoming_odds()
        pm_markets   = get_polymarket_odds()
        odds_history = gh_get_moneyline_history()

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
        if GITHUB_TOKEN and eventos_ufc:
            if not st.session_state.get("history_toast_shown"):
                st.toast("✅ Predictions saved to history")
                st.session_state["history_toast_shown"] = True
            try:
                gh_update_resultados()
            except:
                pass
            # Recolher totals odds uma vez por dia
            last_totals = st.session_state.get("last_totals_fetch", None)
            today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
            if last_totals != today_str:
                try:
                    gh_save_totals_odds()
                    gh_save_moneyline_odds(combates_api)
                    st.session_state["last_totals_fetch"] = today_str
                except:
                    pass
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
                        p2, p1, _, _, _, _, _, ens_std = res
                    else:
                        sem_dados.append(f"{f1_ufc} vs {f2_ufc}")
                        continue
                else:
                    p1, p2, _, _, _, _, _, ens_std = res

                prob_fav = max(p1, p2)
                pm_p1, pm_p2, pm_vol = match_polymarket(f1_ufc, f2_ufc, pm_markets)
                disagreement = calc_bookmaker_disagreement(f1_ufc, f2_ufc, combates_api)
                combates_proc.append({
                    "f1": f1_ufc, "f2": f2_ufc,
                    "p1": p1, "p2": p2,
                    "prob_fav": prob_fav,
                    "odds_f1": odds_f1, "odds_f2": odds_f2,
                    "ordem": conviction_order(prob_fav),
                    "title_bout": False,
                    "rounds": 3,
                    "prob_decision": res[4] if len(res) > 4 else None,
                    "prob_over25":   res[5] if len(res) > 5 else None,
                    "pm_p1": pm_p1, "pm_p2": pm_p2, "pm_vol": pm_vol,
                    "disagreement": disagreement,
                    "meta_score": res[6] if len(res) > 6 else None,
                    "ensemble_std": res[7] if len(res) > 7 else None,
                })

            combates_proc.sort(key=lambda x: (x["ordem"], -x["prob_fav"]))
            parlay_sugerida = sugerir_parlay(combates_proc)

            # Guardar previsões só para eventos de hoje ou passados
            from datetime import date
            if combates_proc and GITHUB_TOKEN and evento["data"].date() <= date.today():
                try:
                    gh_save_previsoes(combates_proc, evento["nome"])
                except Exception as e:
                    pass

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
                render_fight_card(c, odds_history=odds_history)
                # Secondary markets expander
                prob_dec_c = c.get("prob_decision")
                prob_o25_c = c.get("prob_over25")
                if prob_dec_c is not None or prob_o25_c is not None:
                    with st.expander("📊 Secondary Markets"):
                        sec_c1, sec_c2 = st.columns(2)
                        if prob_dec_c is not None:
                            with sec_c1:
                                dec_lbl = "LIKELY DECISION" if prob_dec_c > 0.5 else "LIKELY FINISH"
                                dec_col = "#D4AF37" if prob_dec_c > 0.5 else "#e8253f"
                                st.markdown(f"""
                                <div style="background:var(--bg3); border:1px solid var(--border);
                                     border-radius:10px; padding:14px; text-align:center;">
                                  <div style="font-size:0.7rem; color:var(--muted);
                                       letter-spacing:2px; text-transform:uppercase;
                                       margin-bottom:4px;">Goes to Decision</div>
                                  <div style="font-family:'Barlow Condensed',sans-serif;
                                       font-size:2rem; font-weight:900;
                                       color:{dec_col};">{prob_dec_c*100:.0f}%</div>
                                  <div style="font-size:0.7rem; color:{dec_col};
                                       font-weight:700; margin-top:2px;">{dec_lbl}</div>
                                  <div style="font-size:0.65rem; color:var(--muted);
                                       margin-top:2px;">Model accuracy: 59.45%</div>
                                </div>
                                """, unsafe_allow_html=True)
                        if prob_o25_c is not None:
                            with sec_c2:
                                o25_lbl = "LIKELY OVER" if prob_o25_c > 0.5 else "LIKELY UNDER"
                                o25_col = "#22c55e" if prob_o25_c > 0.5 else "#60a5fa"
                                st.markdown(f"""
                                <div style="background:var(--bg3); border:1px solid var(--border);
                                     border-radius:10px; padding:14px; text-align:center;">
                                  <div style="font-size:0.7rem; color:var(--muted);
                                       letter-spacing:2px; text-transform:uppercase;
                                       margin-bottom:4px;">Over / Under 2.5 Rounds</div>
                                  <div style="font-family:'Barlow Condensed',sans-serif;
                                       font-size:2rem; font-weight:900;
                                       color:{o25_col};">{prob_o25_c*100:.0f}%</div>
                                  <div style="font-size:0.7rem; color:{o25_col};
                                       font-weight:700; margin-top:2px;">{o25_lbl}</div>
                                  <div style="font-size:0.65rem; color:var(--muted);
                                       margin-top:2px;">Model accuracy: 62.43%</div>
                                </div>
                                """, unsafe_allow_html=True)

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
         font-weight:800; letter-spacing:0.04em; color:#ffffff; margin-bottom:20px;">
      FIGHT ANALYSIS
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
            # Valores por defeito para inputs opcionais
            r_ko_odds_in  = locals().get('r_ko_odds_in', 1.0)
            r_sub_odds_in = locals().get('r_sub_odds_in', 1.0)
            r_dec_odds_in = locals().get('r_dec_odds_in', 1.0)
            b_ko_odds_in  = locals().get('b_ko_odds_in', 1.0)
            b_sub_odds_in = locals().get('b_sub_odds_in', 1.0)
            b_dec_odds_in = locals().get('b_dec_odds_in', 1.0)
            rounds_in     = locals().get('rounds_in', 3)
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
                prob_r, prob_b, red, blue, prob_decision, prob_over25, meta_score, ens_std = res
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
                  <div class="result-pct">Probability: {max(prob_r, prob_b):.1%}</div>
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
                  📊 Fighter Stats
                </div>
                """, unsafe_allow_html=True)

                stats_map = [
                    ("Wins",             "wins"),
                    ("Losses",             "losses"),
                    ("Win Streak",           "win_streak"),
                    ("KO Wins",              "ko_wins"),
                    ("Sub Wins",             "sub_wins"),
                    ("Height (cm)",          "height"),
                    ("Reach (cm)",           "reach"),
                    ("Age",                "age"),
                    ("SLpM",                 "SLpM"),
                    ("SApM",                 "SApM"),
                    ("Str. Accuracy",        "sig_str_acc"),
                    ("Str. Defence",         "str_def"),
                    ("TD avg",               "td_avg"),
                    ("TD Defence",           "td_def"),
                    ("Finish Rate",          "finish_rate"),
                    ("KO Rate",              "ko_rate"),
                    ("Sub Rate",             "sub_rate"),
                    ("Recent Win Rate",     "winrate_recente"),
                    ("KO losses (last 3)", "ko_sofrido_recente"),
                    ("Days inactive",         "dias_inactive"),
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

# ── TAB 3 ─────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.6rem;
         font-weight:800; letter-spacing:0.04em; color:#ffffff; margin-bottom:4px;">
      PREDICTION HISTORY
    </div>
    """, unsafe_allow_html=True)

    col_h1, col_h2 = st.columns([3,1])
    with col_h2:
        if st.button("🔄 Update Results", use_container_width=True):
            with st.spinner("Fetching latest results..."):
                gh_update_resultados()
            st.success("Results updated!")
            st.rerun()

    hist = get_historico()

    if hist.empty:
        st.info("No prediction history yet. Predictions are saved automatically when you load the Upcoming Events tab.")
    else:
        total = len(com_resultado)
        com_resultado = hist[hist["correct"].isin(["✅","❌"])]
        corretos = (com_resultado["correct"] == "✅").sum()
        acc = corretos / len(com_resultado) if len(com_resultado) > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Predictions", total)
        m2.metric("With Results", len(com_resultado))
        m3.metric("Correct", corretos)
        m4.metric("Accuracy", f"{acc:.1%}")

        st.markdown("---")

        # Mostrar só eventos com pelo menos um resultado
        eventos_com_resultado = hist[hist["correct"].isin(["✅","❌"])]["event_name"].unique().tolist()
        hist = hist[hist["event_name"].isin(eventos_com_resultado)]
        eventos = ["All"] + sorted(eventos_com_resultado, reverse=True)
        evento_sel = st.selectbox("Filter by Event", eventos)

        df_show = hist if evento_sel == "All" else hist[hist["event_name"] == evento_sel]
        df_show = df_show.sort_values("saved_at", ascending=False)

        for _, row in df_show.iterrows():
            correct_val = row["correct"]
            correct_val = "" if pd.isna(correct_val) else str(correct_val)
            correct_icon = correct_val if correct_val in ["✅","❌"] else "⏳"
            actual = row["actual_winner"]
            winner_display = "Pending" if (pd.isna(actual) or str(actual).strip() == "") else str(actual)
            pred_col = "#22c55e" if correct_val == "✅" else ("#e8253f" if correct_val == "❌" else "#D4AF37")

            st.markdown(f"""
            <div style="background:var(--bg2); border:1px solid var(--border);
                 border-radius:10px; padding:12px 16px; margin-bottom:8px;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                  <span style="font-family:'Barlow Condensed',sans-serif; font-size:1.1rem;
                       font-weight:700; color:#fff;">
                    {row['fighter_1']} <span style="color:var(--muted);">vs</span> {row['fighter_2']}
                  </span>
                  <span style="font-size:0.75rem; color:var(--muted); margin-left:10px;">
                    {row['event_name']}
                  </span>
                </div>
                <div style="font-size:1.4rem;">{correct_icon}</div>
              </div>
              <div style="display:flex; gap:20px; margin-top:6px; font-size:0.8rem;">
                <span>🎯 Predicted: <strong style="color:{pred_col};">{row['predicted_winner']}</strong>
                  ({max(row['prob_f1'], row['prob_f2'])*100:.0f}% · {row['conviction']})</span>
                <span>🏆 Result: <strong style="color:#fff;">{winner_display}</strong></span>
              </div>
            </div>
            """, unsafe_allow_html=True)
