"""
Apex CSV Predictor — takes an Apex-schema CSV and returns fight predictions
using the retrained ensemble model.

Usage from Streamlit:
    from apex_predictor import render_apex_page
    render_apex_page()
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "ufc_ensemble_apex.pkl"
FEATURES_PATH = ROOT / "features_ensemble_apex.pkl"
LOOKUP_PATH = ROOT / "fighter_lookup_apex.csv"
METRICS_PATH = ROOT / "apex_metrics.json"

# ---------- Apex column -> internal feature-name map ----------
APEX_TO_INTERNAL = {
    "age": "age",
    "height": "height_in",
    "reach": "reach_in",
    "is_orthodox": "is_orthodox",
    "is_southpaw": "is_southpaw",
    "is_switch": "is_switch",
    "ufc_fight_count": "ufc_fight_count",
    "ufc_win_count": "ufc_wins",
    "ufc_loss_count": "ufc_losses",
    "career_ko_wins": "career_ko_wins",
    "career_sub_wins": "career_sub_wins",
    "career_ko_losses": "career_ko_losses",
    "career_sub_losses": "career_sub_losses",
    "champ_experience": "champ_experience",
    "ufc_minutes_fought": "ufc_minutes_fought",
    "slpm": "slpm",
    "sapm": "sapm",
    "sig_strike_diff": "sig_strike_diff",
    "sig_strike_accuracy": "sig_strike_accuracy",
    "sig_strike_defense": "sig_strike_defense",
    "head_strike_landed_pct": "head_strike_landed_pct",
    "head_absorption_pct": "head_absorption_pct",
    "kd_rate": "kd_rate",
    "kd_absorbed_rate": "kd_absorbed_rate",
    "td_per_15": "td_per_15",
    "td_attempted_per_15": "td_attempted_per_15",
    "td_defense_pct": "td_defense_pct",
    "ctrl_time_per_15": "ctrl_time_per_15",
    "ctrl_absorbed_per_15": "ctrl_absorbed_per_15",
    "sub_win_pct": "sub_win_pct",
    "finish_rate": "finish_rate",
    "xr_pct": "xr_pct",
    "opp_ufc_win_pct": "opp_ufc_win_pct",
    "opp_xr_pct": "opp_xr_pct",
    "elo": "elo",
    "win_rate_l3": "win_rate_l3",
    "days_inactive": "days_inactive",
    "is_long_layoff": "is_long_layoff",
    "stance": "stance",
    "FIGHTER": "fighter_name",
    "OPPONENT": "opponent_name",
}
RATIO_COLS = ["reach_in", "height_in", "age", "elo"]


def _parse_apex_col(x):
    """Parse Apex values like '38 years' -> 38, '74in' -> 74, '69%' -> 0.69, keep numbers as floats."""
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
    # Strip common units
    for suffix in [" years", "years", "in", " in", "yr", " yr", "min", " min"]:
        if s.lower().endswith(suffix):
            s = s[: -len(suffix)]
            break
    try:
        return float(s)
    except Exception:
        return np.nan


@st.cache_resource
def load_apex_model():
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
    """Load a raw Apex CSV, normalize column names, parse numeric strings."""
    df = pd.read_csv(uploaded_file)
    # Rename Apex columns to internal names
    rename_map = {}
    for src, dst in APEX_TO_INTERNAL.items():
        if src in df.columns:
            rename_map[src] = dst
    df = df.rename(columns=rename_map)
    # Parse messy strings
    for col in df.columns:
        if col in ("fighter_name", "opponent_name", "stance"):
            continue
        df[col] = df[col].apply(_parse_apex_col)
    return df


def build_matchup_row(a: pd.Series, b: pd.Series, features: list[str]) -> pd.DataFrame:
    """Given two rows (fighter A + fighter B) build one row of features aligned to `features`."""
    row = {}
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
    # Align to model feature order
    return pd.DataFrame([{k: row.get(k, np.nan) for k in features}])


def predict_pair(models: dict, features: list[str], a: pd.Series, b: pd.Series) -> dict:
    X = build_matchup_row(a, b, features)
    # Reverse pair to symmetrize
    Xr = build_matchup_row(b, a, features)

    def one_direction(X):
        X_imp = pd.DataFrame(models["imputer"].transform(X), columns=X.columns)
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

    p_a = one_direction(X)              # P(A beats B)
    p_b = one_direction(Xr)             # P(B beats A)  -> flip to get P(A) from B-perspective row
    p_a_final = (p_a["ensemble"] + (1 - p_b["ensemble"])) / 2
    p_b_final = 1 - p_a_final
    return {
        "prob_a": p_a_final,
        "prob_b": p_b_final,
        "per_model_a": p_a,
        "per_model_b": p_b,
    }


def render_apex_page():
    st.header("🏆 Apex CSV Predictor")
    st.caption(
        "Upload an Apex-schema CSV (rows = fighters with pre-computed features) and get "
        "matchup predictions from the retrained ensemble model."
    )

    with st.expander("Model performance & training details", expanded=False):
        try:
            models, features, lookup, metrics = load_apex_model()
            per = metrics.get("per_model", {})
            st.markdown(
                f"""
**Training set:** {metrics.get('train_rows', '?'):,} symmetrized rows from UFC fights 2021-01-16 → 2024-05-31  
**Holdout set:** {metrics.get('test_rows', '?'):,} rows from {metrics.get('holdout_start', '?')} onwards  
**Feature count:** {metrics.get('feature_count', '?')}  
**Base rate:** 50% (symmetrized — no A/B bias)

| Model | Holdout Acc | Log-loss | ROC-AUC | Brier |
|---|---|---|---|---|
| XGBoost   | {per.get('xgb', {}).get('accuracy', 0):.1%} | {per.get('xgb', {}).get('log_loss', 0):.3f} | {per.get('xgb', {}).get('roc_auc', 0):.3f} | {per.get('xgb', {}).get('brier', 0):.3f} |
| LightGBM  | {per.get('lgb', {}).get('accuracy', 0):.1%} | {per.get('lgb', {}).get('log_loss', 0):.3f} | {per.get('lgb', {}).get('roc_auc', 0):.3f} | {per.get('lgb', {}).get('brier', 0):.3f} |
| Random Forest | {per.get('rf', {}).get('accuracy', 0):.1%} | {per.get('rf', {}).get('log_loss', 0):.3f} | {per.get('rf', {}).get('roc_auc', 0):.3f} | {per.get('rf', {}).get('brier', 0):.3f} |
| Logistic Regression | {per.get('lr', {}).get('accuracy', 0):.1%} | {per.get('lr', {}).get('log_loss', 0):.3f} | {per.get('lr', {}).get('roc_auc', 0):.3f} | {per.get('lr', {}).get('brier', 0):.3f} |
| CatBoost  | {per.get('cat', {}).get('accuracy', 0):.1%} | {per.get('cat', {}).get('log_loss', 0):.3f} | {per.get('cat', {}).get('roc_auc', 0):.3f} | {per.get('cat', {}).get('brier', 0):.3f} |
| **Ensemble** | **{per.get('ensemble_test', {}).get('accuracy', 0):.1%}** | **{per.get('ensemble_test', {}).get('log_loss', 0):.3f}** | **{per.get('ensemble_test', {}).get('roc_auc', 0):.3f}** | **{per.get('ensemble_test', {}).get('brier', 0):.3f}** |
| Ensemble (unique fights) | {per.get('ensemble_orig_only_test', {}).get('accuracy', 0):.1%} | | | |
"""
            )
            st.info(
                "Ensemble was validated on 2 years of unseen holdout data. Note: current v5 model reports "
                "68.45% but uses additional odds-movement features (BFO line movement) that this model does not."
            )
        except Exception as e:
            st.error(f"Could not load model metrics: {e}")
            return

    st.divider()

    up = st.file_uploader("Upload Apex CSV (fighter-per-row format)", type=["csv"], key="apex_csv")

    if up is None:
        st.markdown(
            "**Expected columns** (Apex fighter-row format): `FIGHTER`, `OPPONENT`, `age`, `height`, `reach`, "
            "`stance`, `ufc_fight_count`, `ufc_win_count`, `ufc_loss_count`, `career_ko_wins`, `career_sub_wins`, "
            "`slpm`, `sapm`, `td_per_15`, `td_defense_pct`, `ctrl_time_per_15`, `sig_strike_accuracy`, "
            "`sig_strike_defense`, `head_strike_landed_pct`, `finish_rate`, `xr_pct`, `opp_ufc_win_pct`, "
            "`opp_xr_pct`, `elo`, `win_rate_l3`, `days_inactive`, and a few more optional Apex columns."
        )
        return

    df = parse_apex_csv(up)
    st.success(f"Loaded {len(df)} rows from CSV. Detected columns: {len(df.columns)}")

    # Try to pair fighters: expect OPPONENT column
    if "opponent_name" not in df.columns or "fighter_name" not in df.columns:
        st.error("Missing required columns: need FIGHTER and OPPONENT columns for matchup pairing.")
        st.dataframe(df.head())
        return

    # Build unique matchup keys
    df["_matchup_key"] = df.apply(
        lambda r: tuple(sorted([str(r["fighter_name"]).strip(), str(r["opponent_name"]).strip()])),
        axis=1,
    )

    models, features, lookup, metrics = load_apex_model()

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
            # Only one row for this matchup; look up opponent in the same DF or in lookup
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
            "Fighter B": b["fighter_name"],
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

    # Download
    csv = result_df.to_csv(index=False).encode()
    st.download_button("Download predictions as CSV", csv, "apex_predictions.csv", "text/csv")
