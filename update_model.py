
import os
import subprocess
import sys
import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 60)
print("UFC PREDICTOR — UPDATE PIPELINE")
print(f"Data: {datetime.now().strftime('%d %b %Y %H:%M')}")
print("=" * 60)

# ── PASSO 1: DESCARREGAR DADOS KAGGLE ────────────────────────────
print("\n[1/4] A descarregar ufc-master.csv do Kaggle...")
try:
    result = subprocess.run(
        ["kaggle", "datasets", "download", "mdabbert/ultimate-ufc-dataset",
         "--unzip", "-p", "ultimate_ufc_dataset"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("  ✅ ufc-master.csv actualizado")
    else:
        print(f"  ❌ Erro Kaggle: {result.stderr[:100]}")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ {e}")
    sys.exit(1)

# ── PASSO 2: SCRAPER UFCStats ─────────────────────────────────────
print("\n[2/4] A correr scraper UFCStats...")
try:
    result = subprocess.run(
        ["python", "scrape_ufc_stats-main/scrape_ufc_stats_unparsed_data.py"],
        capture_output=True, text=True, timeout=1800  # 30 min max
    )
    if result.returncode == 0:
        print("  ✅ Dados UFCStats actualizados")
    else:
        print(f"  ⚠️  Scraper com erros: {result.stderr[:200]}")
        print("  A continuar com dados existentes...")
except subprocess.TimeoutExpired:
    print("  ⚠️  Scraper demorou demasiado — a continuar com dados existentes")
except Exception as e:
    print(f"  ⚠️  {e} — a continuar com dados existentes")

# ── PASSO 3: RECALCULAR ROLLING FEATURES ─────────────────────────
print("\n[3/4] A recalcular features rolling...")
try:
    import pickle
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    # Carregar dados
    df_stats_raw  = pd.read_csv("scrape_ufc_stats-main/ufc_fight_stats.csv")
    df_res_raw    = pd.read_csv("scrape_ufc_stats-main/ufc_fight_results.csv")
    df_events_raw = pd.read_csv("scrape_ufc_stats-main/ufc_event_details.csv")
    df_ult        = pd.read_csv("ultimate_ufc_dataset/ufc-master.csv")

    for df in [df_stats_raw, df_res_raw, df_events_raw]:
        df["EVENT"] = df["EVENT"].str.strip()

    print(f"  Combates no ufc-master: {len(df_ult):,}")
    print(f"  Eventos no UFCStats:    {len(df_events_raw):,}")

    # Funções de parse
    def parse_of(v):
        try: a, b = str(v).split(" of "); return int(a), int(b)
        except: return 0, 0

    def parse_pct(v):
        try: return float(str(v).replace("%","")) / 100
        except: return 0.0

    def ctrl_to_sec(v):
        try: m, s = str(v).split(":"); return int(m)*60 + int(s)
        except: return 0

    def time_to_total_sec(time_str, round_num):
        try: m, s = str(time_str).split(":"); return (int(round_num)-1)*5*60 + int(m)*60 + int(s)
        except: return 0

    df_res_raw["total_sec"] = df_res_raw.apply(
        lambda r: time_to_total_sec(r["TIME"], r["ROUND"]), axis=1)

    df_stats = df_stats_raw.merge(df_events_raw[["EVENT","DATE"]], on="EVENT", how="left")
    df_stats["sig_landed"], df_stats["sig_att"]   = zip(*df_stats["SIG.STR."].map(parse_of))
    df_stats["head_landed"], _                    = zip(*df_stats["HEAD"].map(parse_of))
    df_stats["body_landed"], _                    = zip(*df_stats["BODY"].map(parse_of))
    df_stats["leg_landed"],  _                    = zip(*df_stats["LEG"].map(parse_of))
    df_stats["distance_landed"], _                = zip(*df_stats["DISTANCE"].map(parse_of))
    df_stats["clinch_landed"],   _                = zip(*df_stats["CLINCH"].map(parse_of))
    df_stats["ground_landed"],   _                = zip(*df_stats["GROUND"].map(parse_of))
    df_stats["td_landed"], df_stats["td_att"]     = zip(*df_stats["TD"].map(parse_of))
    df_stats["sig_pct"]  = df_stats["SIG.STR. %"].map(parse_pct)
    df_stats["td_pct"]   = df_stats["TD %"].map(parse_pct)
    df_stats["ctrl_sec"] = df_stats["CTRL"].map(ctrl_to_sec)
    df_stats["sub_att"]  = pd.to_numeric(df_stats["SUB.ATT"], errors="coerce").fillna(0)
    df_stats["kd"]       = pd.to_numeric(df_stats["KD"], errors="coerce").fillna(0)
    df_stats["rev"]      = pd.to_numeric(df_stats["REV."], errors="coerce").fillna(0)

    per_bout = df_stats.groupby(["FIGHTER","BOUT","EVENT"]).agg(
        sig_landed=("sig_landed","sum"), sig_att=("sig_att","sum"),
        head_landed=("head_landed","sum"), body_landed=("body_landed","sum"),
        leg_landed=("leg_landed","sum"), distance_landed=("distance_landed","sum"),
        clinch_landed=("clinch_landed","sum"), ground_landed=("ground_landed","sum"),
        td_landed=("td_landed","sum"), td_att=("td_att","sum"),
        sub_att=("sub_att","sum"), kd=("kd","sum"), ctrl_sec=("ctrl_sec","sum"),
        sig_pct=("sig_pct","mean"), td_pct=("td_pct","mean"),
        rev=("rev","sum"), DATE=("DATE","first"),
    ).reset_index()

    per_bout = per_bout.merge(
        df_res_raw[["BOUT","total_sec"]].drop_duplicates("BOUT"), on="BOUT", how="left")
    per_bout["DATE"] = pd.to_datetime(per_bout["DATE"], format="%B %d, %Y", errors="coerce")
    per_bout = per_bout.sort_values(["FIGHTER","DATE"]).reset_index(drop=True)

    # Calcular rolling
    def calcular_rolling(per_bout):
        resultados = []
        for fighter, grupo in per_bout.groupby("FIGHTER"):
            grupo = grupo.sort_values("DATE").reset_index(drop=True)
            for i in range(len(grupo)):
                historico = grupo.iloc[:i]
                combate   = grupo.iloc[i]
                if len(historico) == 0:
                    row = {"FIGHTER": fighter, "BOUT": combate["BOUT"], "DATE": combate["DATE"]}
                    for c in ["SLpM","sig_str_acc","td_avg","td_acc","sub_avg","ctrl_avg",
                              "kd_per_fight","head_ratio","body_ratio","leg_ratio",
                              "distance_ratio","clinch_ratio","ground_ratio",
                              "sig_pct_fade","td_fade","ctrl_fade","rev_avg"]:
                        row[c] = np.nan
                    resultados.append(row)
                    continue
                total_min = max(historico["total_sec"].sum() / 60, 1)
                total_sig = max(historico["sig_landed"].sum(), 1)
                n = len(historico)
                primeiro = historico.iloc[:max(1, n // 3)]
                ultimo   = historico.iloc[max(1, n // 3 * 2):]
                resultados.append({
                    "FIGHTER": fighter, "BOUT": combate["BOUT"], "DATE": combate["DATE"],
                    "SLpM":           historico["sig_landed"].sum() / total_min,
                    "sig_str_acc":    historico["sig_pct"].mean(),
                    "td_avg":         historico["td_landed"].sum() / n,
                    "td_acc":         historico["td_landed"].sum() / max(historico["td_att"].sum(), 1),
                    "sub_avg":        historico["sub_att"].sum() / n,
                    "ctrl_avg":       historico["ctrl_sec"].sum() / n,
                    "kd_per_fight":   historico["kd"].sum() / n,
                    "rev_avg":        historico["rev"].sum() / n,
                    "head_ratio":     historico["head_landed"].sum() / total_sig,
                    "body_ratio":     historico["body_landed"].sum() / total_sig,
                    "leg_ratio":      historico["leg_landed"].sum() / total_sig,
                    "distance_ratio": historico["distance_landed"].sum() / total_sig,
                    "clinch_ratio":   historico["clinch_landed"].sum() / total_sig,
                    "ground_ratio":   historico["ground_landed"].sum() / total_sig,
                    "sig_pct_fade":   (ultimo["sig_pct"].mean() - primeiro["sig_pct"].mean()) if len(ultimo) > 0 else np.nan,
                    "td_fade":        (ultimo["td_landed"].mean() - primeiro["td_landed"].mean()) if len(ultimo) > 0 else np.nan,
                    "ctrl_fade":      (ultimo["ctrl_sec"].mean() - primeiro["ctrl_sec"].mean()) if len(ultimo) > 0 else np.nan,
                })
        return pd.DataFrame(resultados)

    print("  A calcular rolling features (3-4 min)...")
    df_rolling = calcular_rolling(per_bout)

    # Defensivas
    absorbed = []
    for bout, grupo in per_bout.groupby("BOUT"):
        if len(grupo) != 2: continue
        f1, f2 = grupo.iloc[0], grupo.iloc[1]
        for fighter, opp in [(f1, f2), (f2, f1)]:
            absorbed.append({
                "FIGHTER": fighter["FIGHTER"], "BOUT": bout, "DATE": fighter["DATE"],
                "sig_absorbed": opp["sig_landed"], "sig_att_opp": opp["sig_att"],
                "td_att_opp": opp["td_att"], "td_landed_opp": opp["td_landed"],
                "total_sec": fighter["total_sec"],
            })

    df_absorbed = pd.DataFrame(absorbed).sort_values(["FIGHTER","DATE"]).reset_index(drop=True)

    def calcular_defensivas(df_absorbed):
        resultados = []
        for fighter, grupo in df_absorbed.groupby("FIGHTER"):
            grupo = grupo.sort_values("DATE").reset_index(drop=True)
            for i in range(len(grupo)):
                historico = grupo.iloc[:i]
                combate   = grupo.iloc[i]
                if len(historico) == 0:
                    resultados.append({"FIGHTER": fighter, "BOUT": combate["BOUT"],
                                       "DATE": combate["DATE"],
                                       "SApM": np.nan, "str_def": np.nan, "td_def": np.nan})
                    continue
                total_min = max(historico["total_sec"].sum() / 60, 1)
                total_att = max(historico["sig_att_opp"].sum(), 1)
                total_td  = historico["td_att_opp"].sum()
                resultados.append({
                    "FIGHTER": fighter, "BOUT": combate["BOUT"], "DATE": combate["DATE"],
                    "SApM":    historico["sig_absorbed"].sum() / total_min,
                    "str_def": 1 - (historico["sig_absorbed"].sum() / total_att),
                    "td_def":  1 - (historico["td_landed_opp"].sum() / max(total_td, 1)),
                })
        return pd.DataFrame(resultados)

    df_defensivas = calcular_defensivas(df_absorbed)

    # Finish rates
    def get_fighter_results(df_res_dated):
        rows = []
        for _, row in df_res_dated.iterrows():
            bout    = str(row["BOUT"]).strip()
            outcome = str(row["OUTCOME"]).strip()
            method  = str(row["METHOD"]).strip()
            date    = row["DATE_dt"] if "DATE_dt" in row else row["DATE"]
            try:
                f1, f2 = bout.split(" vs. ")
                f1, f2 = f1.strip(), f2.strip()
            except:
                continue
            for fighter, ganhou in [(f1, outcome == "W/L"), (f2, outcome == "L/W")]:
                rows.append({
                    "FIGHTER": fighter, "BOUT": bout, "DATE": date,
                    "ganhou":     int(ganhou),
                    "metodo_ko":  int("KO" in method or "TKO" in method),
                    "metodo_sub": int("Submission" in method),
                })
        return pd.DataFrame(rows)

    df_res_dated = df_res_raw.copy()
    df_res_dated["DATE_dt"] = df_res_dated.merge(
        df_events_raw[["EVENT","DATE"]], on="EVENT", how="left")["DATE"]
    df_res_dated["DATE_dt"] = pd.to_datetime(df_res_dated["DATE_dt"],
                                              format="%B %d, %Y", errors="coerce")
    df_resultados = get_fighter_results(df_res_dated).sort_values(
        ["FIGHTER","DATE"]).reset_index(drop=True)

    def calcular_finish_rolling(df_resultados):
        resultados = []
        for fighter, grupo in df_resultados.groupby("FIGHTER"):
            grupo = grupo.sort_values("DATE").reset_index(drop=True)
            for i in range(len(grupo)):
                historico = grupo.iloc[:i]
                combate   = grupo.iloc[i]
                if len(historico) == 0:
                    resultados.append({"FIGHTER": fighter, "BOUT": combate["BOUT"],
                                       "DATE": combate["DATE"],
                                       "finish_rate": np.nan, "ko_rate": np.nan,
                                       "sub_rate": np.nan})
                    continue
                total = len(historico)
                resultados.append({
                    "FIGHTER": fighter, "BOUT": combate["BOUT"], "DATE": combate["DATE"],
                    "finish_rate": (historico["metodo_ko"].sum() + historico["metodo_sub"].sum()) / total,
                    "ko_rate":     historico["metodo_ko"].sum() / total,
                    "sub_rate":    historico["metodo_sub"].sum() / total,
                })
        return pd.DataFrame(resultados)

    df_finish = calcular_finish_rolling(df_resultados)

    print("  ✅ Rolling features calculadas")

    # ── PASSO 4: ACTUALIZAR LOOKUP ────────────────────────────────
    print("\n[4/4] A actualizar fighter lookup...")

    # Juntar todos os rolling
    all_rolling = df_rolling.merge(
        df_finish[["FIGHTER","BOUT","finish_rate","ko_rate","sub_rate"]],
        on=["FIGHTER","BOUT"], how="left"
    ).merge(
        df_defensivas[["FIGHTER","BOUT","SApM","str_def","td_def"]],
        on=["FIGHTER","BOUT"], how="left"
    )

    # Pegar o último valor de cada feature por lutador
    lookup_novo = all_rolling.sort_values("DATE").drop_duplicates(
        "FIGHTER", keep="last")[
        ["FIGHTER","SLpM","sig_str_acc","td_avg","td_acc","sub_avg","ctrl_avg",
         "kd_per_fight","head_ratio","body_ratio","leg_ratio",
         "distance_ratio","clinch_ratio","ground_ratio",
         "sig_pct_fade","td_fade","ctrl_fade","rev_avg",
         "finish_rate","ko_rate","sub_rate","SApM","str_def","td_def"]
    ].rename(columns={"FIGHTER": "name"})

    # Carregar lookup base do ufc-master
    df_ult["date"] = pd.to_datetime(df_ult["date"], errors="coerce")
    df_ult = df_ult[df_ult["Winner"].isin(["Red","Blue"])].copy()

    # Features base do ufc-master
    def build_base_lookup(df):
        rows = []
        for fighter in set(df["R_fighter"].tolist() + df["B_fighter"].tolist()):
            r_fights = df[df["R_fighter"] == fighter].sort_values("date")
            b_fights = df[df["B_fighter"] == fighter].sort_values("date")
            all_fights = pd.concat([
                r_fights.assign(corner="R"),
                b_fights.assign(corner="B")
            ]).sort_values("date")
            if len(all_fights) == 0:
                continue
            last = all_fights.iloc[-1]
            corner = last["corner"]
            p = corner + "_"
            row = {
                "name":               fighter,
                "wins":               last.get(f"{p}wins", np.nan),
                "losses":             last.get(f"{p}losses", np.nan),
                "win_streak":         last.get(f"{p}current_win_streak", np.nan),
                "lose_streak":        last.get(f"{p}current_lose_streak", np.nan),
                "longest_win_streak": last.get(f"{p}longest_win_streak", np.nan),
                "total_rounds":       last.get(f"{p}total_rounds_fought", np.nan),
                "title_bouts":        last.get(f"{p}total_title_bouts", np.nan),
                "ko_wins":            last.get(f"{p}win_by_KO/TKO", np.nan),
                "sub_wins":           last.get(f"{p}win_by_Submission", np.nan),
                "height":             last.get(f"{p}Height_cms", np.nan),
                "reach":              last.get(f"{p}Reach_cms", np.nan),
                "age":                last.get(f"{p}age", np.nan),
                "rank":               last.get(f"{p}match_weightclass_rank", np.nan),
                "dias_inativo":       (pd.Timestamp.now() - last["date"]).days
                                      if pd.notna(last["date"]) else np.nan,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    print("  A construir lookup base...")
    lookup_base = build_base_lookup(df_ult)
    lookup_final = lookup_base.merge(lookup_novo, on="name", how="left")

    # Guardar
    lookup_final.to_csv("fighter_lookup_final.csv", index=False)
    print(f"  ✅ Lookup actualizado: {len(lookup_final):,} lutadores")
    print(f"  Guardado em fighter_lookup_final.csv")

except Exception as e:
    import traceback
    print(f"  ❌ Erro no pipeline: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ UPDATE COMPLETO")
print("=" * 60)
