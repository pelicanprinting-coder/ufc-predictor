"""
Apex CSV Predictor V2 — enhanced ensemble with archetype features + weighted decay.

New in V2:
  - 6-archetype classifier (KO Artist, Volume Striker, Counter Striker, Wrestler,
    Sub Specialist, Point Fighter) — ported from Apex's own engine
  - Archetype matchup advantage (6x6 matrix)
  - Upset score for spot-checking underdog picks
  - Sample-weight decay: recent fights (2-yr half-life) weighted higher
  - Two holdouts reported: 2024-06 (24-mo) and 2026-01 (6-mo stress test)

Usage from Streamlit:
    from apex_predictor_v2 import render_apex_page_v2
    render_apex_page_v2()
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from archetype import classify_archetype, archetype_advantage, upset_score, confidence_multiplier, ARCHETYPES

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "ufc_ensemble_apex_v2.pkl"
FEATURES_PATH = ROOT / "features_ensemble_apex_v2.pkl"
LOOKUP_PATH = ROOT / "fighter_lookup_apex_v2.csv"
METRICS_PATH = ROOT / "apex_metrics_v2.json"

# Apex column -> internal feature-name map (same as V1)
APEX_TO_INTERNAL = {
    "age": "age", "height": "height_in", "reach": "reach_in",
    "is_orthodox": "is_orthodox", "is_southpaw": "is_southpaw", "is_switch": "is_switch",
    "ufc_fight_count": "ufc_fight_count", "ufc_win_count": "ufc_wins", "ufc_loss_count": "ufc_losses",
    "career_ko_wins": "career_ko_wins", "career_sub_wins": "career_sub_wins",
    "career_ko_losses": "career_ko_losses", "career_sub_losses": "career_sub_losses",
    "champ_experience": "champ_experience", "ufc_minutes_fought": "ufc_minutes_fought",
    "slpm": "slpm", "sapm": "sapm", "sig_strike_diff": "sig_strike_diff",
    "sig_strike_accuracy": "sig_strike_accuracy", "sig_strike_defense": "sig_strike_defense",
    "head_strike_landed_pct": "head_strike_landed_pct", "head_absorption_pct": "head_absorption_pct",
    "kd_rate": "kd_rate", "kd_absorbed_rate": "kd_absorbed_rate",
    "td_per_15": "td_per_15", "td_attempted_per_15": "td_attempted_per_15",
    "td_defense_pct": "td_defense_pct", "ctrl_time_per_15": "ctrl_time_per_15",
    "ctrl_absorbed_per_15": "ctrl_absorbed_per_15",
    "sub_win_pct": "sub_win_pct", "finish_rate": "finish_rate", "xr_pct": "xr_pct",
    "opp_ufc_win_pct": "opp_ufc_win_pct", "opp_xr_pct": "opp_xr_pct",
    "elo": "elo", "win_rate_l3": "win_rate_l3",
    "days_inactive": "days_inactive", "is_long_layoff": "is_long_layoff",
    "stance": "stance", "FIGHTER": "fighter_name", "OPPONENT": "opponent_name",
}
RATIO_COLS = ["reach_in", "height_in", "age", "elo"]

NUMERIC_CORE = [
    "age", "height_in", "reach_in", "is_orthodox", "is_southpaw", "is_switch",
    "ufc_fight_count", "ufc_wins", "ufc_losses", "career_ko_wins", "career_sub_wins",
    "career_ko_losses", "career_sub_losses", "champ_experience", "ufc_minutes_fought",
    "slpm", "sapm", "sig_strike_diff", "sig_strike_accuracy", "sig_strike_defense",
    "head_strike_landed_pct", "head_absorption_pct", "kd_rate", "kd_absorbed_rate",
    "td_per_15", "td_attempted_per_15", "td_defense_pct", "ctrl_time_per_15", "ctrl_absorbed_per_15",
    "sub_win_pct", "finish_rate", "xr_pct", "opp_ufc_win_pct", "opp_xr_pct",
    "elo", "win_rate_l3", "days_inactive", "is_long_layoff",
]


def _parse_apex_col(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return np.nan
    for suffix in [" years", "years", "in", " in", "yr", " yr", "min", " min"]:
        if s.lower().endswith(suffix):
            s = s[: -len(suffix)]
            break
    try:
        return float(s)
    except Exception:
        return np.nan


@st.cache_resource
def load_apex_model_v2():
    with open(MODEL_PATH, "rb") as f:
        models = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        features = pickle.load(f)
    lookup = pd.read_csv(LOOKUP_PATH)
    metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    return models, features, lookup, metrics


def parse_apex_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    rename_map = {src: dst for src, dst in APEX_TO_INTERNAL.items() if src in df.columns}
    df = df.rename(columns=rename_map)
    for col in df.columns:
        if col in ("fighter_name", "opponent_name", "stance"):
            continue
        df[col] = df[col].apply(_parse_apex_col)
    return df


def compute_archetype_inputs(row: pd.Series) -> dict:
    """Extract archetype-engine input dict from a fighter row."""
    finrate = float(row.get("finish_rate", 0) or 0)
    if finrate <= 1.5:
        finrate *= 100
    subpct = float(row.get("sub_win_pct", 0) or 0)
    if subpct <= 1.5:
        subpct *= 100
    kd = float(row.get("kd_rate", 0) or 0) * 15.0
    return {
        "kd": kd,
        "slpm": float(row.get("slpm", 0) or 0),
        "sapm": float(row.get("sapm", 0) or 0),
        "sdiff": float(row.get("sig_strike_diff", 0) or 0),
        "td15": float(row.get("td_per_15", 0) or 0),
        "ctrl": float(row.get("ctrl_time_per_15", 0) or 0),
        "subpct": subpct,
        "finrate": finrate,
        "age": float(row.get("age", 30) or 30),
        "fights": float(row.get("ufc_fight_count", 0) or 0),
    }


def build_matchup_row(a: pd.Series, b: pd.Series, features: list[str]) -> pd.DataFrame:
    row = {}

    # Numeric core + diffs + ratios
    for col in NUMERIC_CORE:
        av = a.get(col, np.nan)
        bv = b.get(col, np.nan)
        row[f"a_{col}"] = av
        row[f"b_{col}"] = bv
        row[f"diff_{col}"] = (av - bv) if (pd.notna(av) and pd.notna(bv)) else np.nan
    for col in RATIO_COLS:
        av = a.get(col, np.nan)
        bv = b.get(col, np.nan)
        if pd.notna(av) and pd.notna(bv) and bv != 0:
            row[f"ratio_{col}"] = av / bv
        else:
            row[f"ratio_{col}"] = np.nan
    row["same_stance"] = int(str(a.get("stance", "")).strip() == str(b.get("stance", "")).strip())
    row["scheduled_rounds"] = 3
    row["is_title_bout"] = 0
    row["is_main_event"] = 0

    # NEW: Archetype features
    a_inputs = compute_archetype_inputs(a)
    b_inputs = compute_archetype_inputs(b)
    a_class = classify_archetype(a_inputs)
    b_class = classify_archetype(b_inputs)

    # One-hot archetype dummies
    arch_safe_names = {arch: arch.replace(" ", "_") for arch in ARCHETYPES}
    for arch, safe in arch_safe_names.items():
        row[f"a_arch_{safe}"] = int(a_class["archetype"] == arch)
        row[f"b_arch_{safe}"] = int(b_class["archetype"] == arch)

    # Per-side max score + confidence multiplier (fights-based) + upset
    # (Matches add_archetype_features.py training-time definitions.)
    a_score_max = a_class["scores"].get(a_class["archetype"], 0.0) if a_class.get("scores") else 0.0
    b_score_max = b_class["scores"].get(b_class["archetype"], 0.0) if b_class.get("scores") else 0.0
    row["archetype_confidence_a"] = confidence_multiplier(a_inputs["fights"])
    row["archetype_confidence_b"] = confidence_multiplier(b_inputs["fights"])
    # Map to feature naming (a_ / b_) used by trainer
    row["a_archetype_confidence"] = row["archetype_confidence_a"]
    row["b_archetype_confidence"] = row["archetype_confidence_b"]
    row["a_archetype_score_max"] = a_score_max
    row["b_archetype_score_max"] = b_score_max
    us_a = upset_score(a_inputs, b_inputs)
    us_b = upset_score(b_inputs, a_inputs)
    row["a_upset_score"] = float(us_a["score"]) if isinstance(us_a, dict) else float(us_a)
    row["b_upset_score"] = float(us_b["score"]) if isinstance(us_b, dict) else float(us_b)

    # Diffs for continuous archetype features
    row["diff_upset_score"] = row["a_upset_score"] - row["b_upset_score"]
    row["diff_archetype_confidence"] = row["a_archetype_confidence"] - row["b_archetype_confidence"]
    row["diff_archetype_score_max"] = row["a_archetype_score_max"] - row["b_archetype_score_max"]

    # Global archetype_adv_diff (a's edge minus b's edge)
    adv_a = archetype_advantage(a_class["archetype"], b_class["archetype"])
    adv_b = archetype_advantage(b_class["archetype"], a_class["archetype"])
    row["archetype_adv_diff"] = adv_a - adv_b

    # Return DataFrame aligned to features
    return pd.DataFrame([{k: row.get(k, np.nan) for k in features}]), a_class, b_class


def predict_pair(models: dict, features: list[str], a: pd.Series, b: pd.Series) -> dict:
    X, a_class, b_class = build_matchup_row(a, b, features)
    Xr, _, _ = build_matchup_row(b, a, features)

    def one_direction(Xin):
        X_imp = pd.DataFrame(models["imputer"].transform(Xin), columns=Xin.columns)
        X_scaled = models["lr_scaler"].transform(X_imp)
        preds = {
            "xgb": float(models["xgb"].predict_proba(X_imp)[:, 1][0]),
            "lgb": float(models["lgb"].predict_proba(X_imp)[:, 1][0]),
            "rf":  float(models["rf"].predict_proba(X_imp)[:, 1][0]),
            "lr":  float(models["lr"].predict_proba(X_scaled)[:, 1][0]),
        }
        if models.get("cat") is not None:
            preds["cat"] = float(models["cat"].predict_proba(X_imp)[:, 1][0])
        preds["ensemble"] = float(np.mean(list(preds.values())))
        return preds

    p_a = one_direction(X)
    p_b = one_direction(Xr)
    p_a_final = (p_a["ensemble"] + (1 - p_b["ensemble"])) / 2
    p_b_final = 1 - p_a_final
    return {
        "prob_a": p_a_final,
        "prob_b": p_b_final,
        "per_model_a": p_a,
        "per_model_b": p_b,
        "a_archetype": a_class["archetype"],
        "b_archetype": b_class["archetype"],
        "a_confidence": 0.0,
        "b_confidence": 0.0,
    }


def render_apex_page_v2():
    st.header("Apex CSV Predictor V2")
    st.caption(
        "Enhanced ensemble with 6-archetype classifier, matchup-advantage matrix, "
        "upset scoring, and sample-weight decay. Upload an Apex-schema CSV to get predictions."
    )

    try:
        models, features, lookup, metrics = load_apex_model_v2()
    except Exception as e:
        st.error(f"Could not load V2 model: {e}")
        return

    with st.expander("Model performance & training details", expanded=False):
        m = metrics.get("metrics", {})
        h = m.get("holdout_ensemble", {})
        s = m.get("stress_ensemble", {})
        st.markdown(
            f"""
**Training set:** {metrics.get('train_rows', '?'):,} symmetrized rows (UFC fights up to {metrics.get('holdout_start', '?')})  
**Holdout (24-mo):** {metrics.get('test_rows', '?'):,} rows from {metrics.get('holdout_start', '?')}  
**Stress test (6-mo):** {metrics.get('stress_rows', '?'):,} rows from {metrics.get('stress_start', '?')}  
**Feature count:** {metrics.get('feature_count', '?')} (up from ~121 in V1; includes archetype + upset features)  
**Sample-weight half-life:** {metrics.get('half_life_days', '?')} days — recent fights weighted higher  
**Base rate:** 50% (symmetrized — no A/B bias)

### Ensemble metrics

| Split | Accuracy | Log-loss | ROC-AUC | Brier |
|---|---|---|---|---|
| Holdout (24-mo) | **{h.get('accuracy', 0):.1%}** | {h.get('log_loss', 0):.3f} | {h.get('roc_auc', 0):.3f} | {h.get('brier', 0):.3f} |
| Stress (6-mo, 2026 only) | **{s.get('accuracy', 0):.1%}** | {s.get('log_loss', 0):.3f} | {s.get('roc_auc', 0):.3f} | {s.get('brier', 0):.3f} |

### New in V2

- **6-archetype classifier** ported from CagePicks Apex engine: KO Artist, Volume Striker, Counter Striker, Wrestler, Sub Specialist, Point Fighter
- **6x6 matchup-advantage matrix** — wrestlers get an edge on volume strikers, etc.
- **Upset score** — flags likely underdog wins based on style + youth + finish rate
- **Bug fixes:** xr_pct now correctly = finishes / total_fights (was finishes / wins);
  td_defense_pct seeded from career profile (was defaulting to ~8%)
- **Sample-weight decay** — fights weighted by 2-year half-life so recent MMA meta is prioritized
"""
        )

    st.divider()

    up = st.file_uploader("Upload Apex CSV (fighter-per-row format)", type=["csv"], key="apex_csv_v2")

    if up is None:
        st.markdown(
            "**Expected columns** (Apex fighter-row format): `FIGHTER`, `OPPONENT`, `age`, `height`, `reach`, "
            "`stance`, `ufc_fight_count`, `ufc_win_count`, `ufc_loss_count`, `career_ko_wins`, `career_sub_wins`, "
            "`slpm`, `sapm`, `td_per_15`, `td_defense_pct`, `ctrl_time_per_15`, `sig_strike_accuracy`, "
            "`sig_strike_defense`, `finish_rate`, `xr_pct`, `opp_ufc_win_pct`, `opp_xr_pct`, `elo`, "
            "`win_rate_l3`, `days_inactive`, `kd_rate`, and a few more optional Apex columns."
        )
        return

    df = parse_apex_csv(up)
    st.success(f"Loaded {len(df)} rows from CSV. Detected columns: {len(df.columns)}")

    if "opponent_name" not in df.columns or "fighter_name" not in df.columns:
        st.error("Missing required columns: need FIGHTER and OPPONENT columns for matchup pairing.")
        st.dataframe(df.head())
        return

    df["_matchup_key"] = df.apply(
        lambda r: tuple(sorted([str(r["fighter_name"]).strip(), str(r["opponent_name"]).strip()])),
        axis=1,
    )

    results = []
    seen = set()
    for key, group in df.groupby("_matchup_key"):
        if key in seen or len(group) < 1:
            continue
        seen.add(key)
        rows = group.reset_index(drop=True)
        if len(rows) >= 2:
            a, b = rows.iloc[0], rows.iloc[1]
        else:
            a = rows.iloc[0]
            opp_name = a["opponent_name"]
            opp_row = df[df["fighter_name"] == opp_name]
            if len(opp_row) > 0:
                b = opp_row.iloc[0]
            else:
                lookup_row = lookup[lookup["fighter_name"].str.lower() == str(opp_name).lower()]
                if len(lookup_row) > 0:
                    b = lookup_row.iloc[0]
                else:
                    st.warning(f"Skipping {a['fighter_name']} vs {opp_name} — opponent stats missing")
                    continue

        pred = predict_pair(models, features, a, b)
        results.append({
            "Fighter A": a["fighter_name"],
            "A Archetype": pred["a_archetype"],
            "Fighter B": b["fighter_name"],
            "B Archetype": pred["b_archetype"],
            "P(A wins)": f"{pred['prob_a']:.1%}",
            "P(B wins)": f"{pred['prob_b']:.1%}",
            "Predicted Winner": a["fighter_name"] if pred["prob_a"] > 0.5 else b["fighter_name"],
            "Confidence": f"{abs(pred['prob_a'] - 0.5) * 2:.1%}",
            "_conf_num": abs(pred["prob_a"] - 0.5),
        })

    if not results:
        st.warning("No matchups could be resolved.")
        return

    result_df = pd.DataFrame(results).sort_values("_conf_num", ascending=False).drop(columns=["_conf_num"])
    st.subheader(f"Predictions ({len(result_df)} matchups)")
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    csv = result_df.to_csv(index=False).encode()
    st.download_button("Download predictions as CSV", csv, "apex_predictions_v2.csv", "text/csv")
