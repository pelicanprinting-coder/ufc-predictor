"""
2026 Backtest: run V2 model against every completed 2026 UFC fight and
break results down by weight class, favorite/underdog, main-card position,
short-notice, layoff, archetype matchup, and method (KO/Sub/Dec proxy via
recent_fight_1_result).

Output:
  - per_slice metrics printed to stdout
  - detailed per-fight CSV: backtest_2026_predictions.csv
  - findings JSON: backtest_2026_findings.json
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
LOOKUP = ROOT / "fighter_lookup_apex_v2.csv"
MODEL = ROOT / "ufc_ensemble_apex_v2.pkl"
FEATS = ROOT / "features_ensemble_apex_v2.pkl"
OUT_CSV = ROOT / "backtest_2026_predictions.csv"
OUT_JSON = ROOT / "backtest_2026_findings.json"

# --- load ---
with open(MODEL, "rb") as f:
    models = pickle.load(f)
with open(FEATS, "rb") as f:
    features = pickle.load(f)

df = pd.read_csv(LOOKUP)
df["event_date"] = pd.to_datetime(df["event_date"])

# Import build_matchup_row from the app module (needs streamlit; import lazily)
# We inline-copy the essentials to avoid streamlit dep issues in scripts.
import importlib.util
spec = importlib.util.spec_from_file_location("apex_v2", ROOT / "apex_predictor_v2.py")
# Streamlit is required for import; we already installed it earlier.
apex_v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apex_v2)
predict_pair = apex_v2.predict_pair

# Only 2026 completed fights
d2026 = df[(df["event_date"] >= "2026-01-01") & (df["event_date"] <= "2026-07-11")].copy()
print(f"Loaded {len(d2026)} fighter-fight rows across {d2026['fight_id'].nunique()} unique fights")

# Group into fight-pairs (each fight_id has exactly 2 rows)
groups = d2026.groupby("fight_id")

rows_out = []
skipped = 0

for fight_id, grp in groups:
    if len(grp) != 2:
        skipped += 1
        continue
    # Order deterministically: fighter with alphabetically smaller name = A
    grp = grp.sort_values("fighter_name").reset_index(drop=True)
    a = grp.iloc[0]
    b = grp.iloc[1]
    # Only include if both have is_winner labels
    if pd.isna(a.get("is_winner")) or pd.isna(b.get("is_winner")):
        skipped += 1
        continue
    # Ground truth: a_won = 1 if a is winner
    a_won = int(a["is_winner"] == 1)

    try:
        pred = predict_pair(models, features, a, b)
        p_a = float(pred["prob_a"])
    except Exception as e:
        skipped += 1
        continue

    # Model pick
    model_pick = "A" if p_a >= 0.5 else "B"
    correct = int((model_pick == "A" and a_won) or (model_pick == "B" and not a_won))

    # Favorite = higher probability side; upset detection needs opening odds
    # Absent odds here, we use model-favorite as a proxy — this is *not* a true
    # upset flag, so we report it as "model-favorite hit rate"
    fav_hit = int((p_a >= 0.5 and a_won) or (p_a < 0.5 and not a_won))

    # Confidence buckets
    p_max = max(p_a, 1 - p_a)
    if p_max >= 0.70:
        conf_bucket = "high (>=70%)"
    elif p_max >= 0.60:
        conf_bucket = "med (60-70%)"
    elif p_max >= 0.55:
        conf_bucket = "lean (55-60%)"
    else:
        conf_bucket = "tossup (50-55%)"

    rows_out.append({
        "fight_id": fight_id,
        "event_date": a["event_date"].strftime("%Y-%m-%d"),
        "weight_class": a.get("weight_class", ""),
        "is_title_bout": int(a.get("is_title_bout", 0) or 0),
        "is_main_event": int(a.get("is_main_event", 0) or 0),
        "a_name": a["fighter_name"],
        "b_name": b["fighter_name"],
        "a_archetype": pred["a_archetype"],
        "b_archetype": pred["b_archetype"],
        "a_short_notice": int(a.get("is_short_notice", 0) or 0),
        "b_short_notice": int(b.get("is_short_notice", 0) or 0),
        "a_long_layoff": int(a.get("is_long_layoff", 0) or 0),
        "b_long_layoff": int(b.get("is_long_layoff", 0) or 0),
        "a_age": float(a.get("age", 0) or 0),
        "b_age": float(b.get("age", 0) or 0),
        "a_ufc_fights": int(a.get("ufc_fight_count", 0) or 0),
        "b_ufc_fights": int(b.get("ufc_fight_count", 0) or 0),
        "a_win_rate_l3": float(a.get("win_rate_l3", 0) or 0),
        "b_win_rate_l3": float(b.get("win_rate_l3", 0) or 0),
        "a_finish_rate": float(a.get("finish_rate", 0) or 0),
        "b_finish_rate": float(b.get("finish_rate", 0) or 0),
        "a_recent_1": a.get("recent_fight_1_result", ""),
        "b_recent_1": b.get("recent_fight_1_result", ""),
        "prob_a": round(p_a, 4),
        "confidence": round(p_max, 4),
        "confidence_bucket": conf_bucket,
        "model_pick": model_pick,
        "a_won": a_won,
        "correct": correct,
        "fav_hit": fav_hit,
    })

print(f"Predicted {len(rows_out)} fights, skipped {skipped}")

if not rows_out:
    print("Nothing to backtest; exiting.")
    raise SystemExit(1)

back = pd.DataFrame(rows_out)
back.to_csv(OUT_CSV, index=False)

# ---------- Slice analysis ----------
def slice_stats(df_slice: pd.DataFrame, name: str) -> dict:
    n = len(df_slice)
    if n == 0:
        return {"slice": name, "n": 0, "acc": None, "avg_conf": None}
    acc = df_slice["correct"].mean()
    return {
        "slice": name,
        "n": int(n),
        "acc": round(float(acc), 4),
        "avg_conf": round(float(df_slice["confidence"].mean()), 4),
        "brier_proxy": round(float(((df_slice["prob_a"] - df_slice["a_won"]) ** 2).mean()), 4),
    }

findings = {"overall": slice_stats(back, "overall")}

# Confidence buckets
by_bucket = []
for b in ["high (>=70%)", "med (60-70%)", "lean (55-60%)", "tossup (50-55%)"]:
    by_bucket.append(slice_stats(back[back["confidence_bucket"] == b], b))
findings["by_confidence"] = by_bucket

# Weight class
by_wc = []
for wc in back["weight_class"].value_counts().index:
    if pd.isna(wc):
        continue
    by_wc.append(slice_stats(back[back["weight_class"] == wc], str(wc)))
by_wc = sorted(by_wc, key=lambda x: (x["acc"] or 0))
findings["by_weight_class"] = by_wc

# Title / main event
findings["title_bouts"] = slice_stats(back[back["is_title_bout"] == 1], "title bouts")
findings["main_events"] = slice_stats(back[back["is_main_event"] == 1], "main events")
findings["prelims"] = slice_stats(back[back["is_main_event"] == 0], "prelims+non-main")

# Short notice (either side)
sn_mask = (back["a_short_notice"] == 1) | (back["b_short_notice"] == 1)
findings["short_notice_involved"] = slice_stats(back[sn_mask], "short-notice involved")
findings["no_short_notice"] = slice_stats(back[~sn_mask], "no short-notice")

# Long layoff (either side)
ll_mask = (back["a_long_layoff"] == 1) | (back["b_long_layoff"] == 1)
findings["long_layoff_involved"] = slice_stats(back[ll_mask], "long-layoff involved")
findings["no_long_layoff"] = slice_stats(back[~ll_mask], "no long-layoff")

# Age gap buckets (|a_age - b_age|)
back["age_gap"] = (back["a_age"] - back["b_age"]).abs()
findings["age_gap_5plus"] = slice_stats(back[back["age_gap"] >= 5], "age gap >=5y")
findings["age_gap_lt5"] = slice_stats(back[back["age_gap"] < 5], "age gap <5y")

# Rookie fights: one fighter has <=3 UFC fights
rookie_mask = (back["a_ufc_fights"] <= 3) | (back["b_ufc_fights"] <= 3)
findings["rookie_involved"] = slice_stats(back[rookie_mask], "rookie (<=3 UFC fights)")
findings["experienced_only"] = slice_stats(back[~rookie_mask], "both experienced")

# Archetype-matchup performance
back["arch_pair"] = back.apply(
    lambda r: " vs ".join(sorted([str(r["a_archetype"]), str(r["b_archetype"])])), axis=1
)
by_arch = []
for pair, grp in back.groupby("arch_pair"):
    if len(grp) < 5:
        continue
    by_arch.append(slice_stats(grp, pair))
by_arch = sorted(by_arch, key=lambda x: (x["acc"] or 0))
findings["by_archetype_pair"] = by_arch

# Cold streak: fighter coming off a loss (recent_1 == 0 or "Loss")
def had_recent_loss(v):
    s = str(v).lower()
    return s in ("0", "0.0", "loss", "l", "false")

back["a_off_loss"] = back["a_recent_1"].apply(had_recent_loss)
back["b_off_loss"] = back["b_recent_1"].apply(had_recent_loss)
findings["fighter_off_loss_involved"] = slice_stats(
    back[back["a_off_loss"] | back["b_off_loss"]], "fighter off recent loss"
)

# Confidence calibration
cal_bins = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.0]
back["conf_bin"] = pd.cut(back["confidence"], bins=cal_bins, include_lowest=True)
calibration = []
for bin_lbl, grp in back.groupby("conf_bin", observed=True):
    if len(grp) == 0:
        continue
    calibration.append({
        "bin": str(bin_lbl),
        "n": int(len(grp)),
        "avg_predicted": round(float(grp["confidence"].mean()), 4),
        "empirical_acc": round(float(grp["correct"].mean()), 4),
        "gap": round(float(grp["confidence"].mean() - grp["correct"].mean()), 4),
    })
findings["calibration"] = calibration

# ------- write out -------
with open(OUT_JSON, "w") as f:
    json.dump(findings, f, indent=2, default=str)

# ------- pretty print -------
def _fmt(row):
    if row["n"] == 0:
        return "n=0"
    return f"n={row['n']:4d}  acc={row['acc']:.3f}  conf={row['avg_conf']:.3f}"

print("\n" + "=" * 70)
print("2026 BACKTEST — V2 MODEL")
print("=" * 70)
print(f"\nOverall:                 {_fmt(findings['overall'])}")

print("\nBy confidence:")
for s in findings["by_confidence"]:
    print(f"  {s['slice']:<20s}  {_fmt(s)}")

print("\nBy weight class (lowest acc first):")
for s in findings["by_weight_class"][:10]:
    print(f"  {s['slice']:<25s}  {_fmt(s)}")

print("\nContext slices:")
for k in ("title_bouts", "main_events", "prelims", "short_notice_involved", "no_short_notice",
         "long_layoff_involved", "no_long_layoff", "age_gap_5plus", "age_gap_lt5",
         "rookie_involved", "experienced_only", "fighter_off_loss_involved"):
    print(f"  {k:<30s}  {_fmt(findings[k])}")

print("\nArchetype pairs (>=5 fights, lowest acc first):")
for s in findings["by_archetype_pair"][:12]:
    print(f"  {s['slice']:<50s}  {_fmt(s)}")

print("\nCalibration (predicted vs empirical):")
for s in findings["calibration"]:
    print(f"  {s['bin']:<15s}  n={s['n']:4d}  predicted={s['avg_predicted']:.3f}  empirical={s['empirical_acc']:.3f}  gap={s['gap']:+.3f}")

print(f"\nSaved: {OUT_CSV}  and  {OUT_JSON}")
