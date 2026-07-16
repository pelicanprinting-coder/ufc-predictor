"""
FM-Consensus Backtest against the 2026 slice.

Since backtest_2026_predictions.csv has no historical market prices, we test
a simpler but still valuable filter: **V2 pick == FM Blend pick** (both models
agree on who wins). Hypothesis: agreement should raise accuracy above the
60% baseline; disagreement should tank it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fightmatrix_math import elo_win_prob

ROOT = Path(__file__).parent
PREDS = ROOT / "backtest_2026_predictions.csv"
FM_LOOKUP = ROOT / "fighter_lookup_apex_v2_with_fm.csv"
OUT_CSV = ROOT / "backtest_2026_fm_consensus.csv"
OUT_JSON = ROOT / "backtest_2026_fm_consensus.json"


def _blended(row) -> float | None:
    if row is None:
        return None
    g = row.get("fm_glicko1")
    w = row.get("fm_whr")
    try:
        g, w = float(g), float(w)
    except (TypeError, ValueError):
        return None
    if pd.isna(g) or pd.isna(w):
        return None
    return 0.45 * g + 0.55 * w


def _row_for(fm: pd.DataFrame, name: str):
    if not name:
        return None
    m = fm[fm["fighter_name"].str.lower() == str(name).strip().lower()]
    if m.empty:
        return None
    m = m.sort_values("fm_blended", na_position="last")
    return m.iloc[0]


def main():
    preds = pd.read_csv(PREDS)
    fm = pd.read_csv(FM_LOOKUP, low_memory=False)

    fm_blend_a, fm_blend_b, fm_prob_a = [], [], []
    for _, r in preds.iterrows():
        ra = _row_for(fm, r["a_name"])
        rb = _row_for(fm, r["b_name"])
        ba, bb = _blended(ra), _blended(rb)
        if ba is None or bb is None:
            fm_blend_a.append(np.nan); fm_blend_b.append(np.nan); fm_prob_a.append(np.nan)
            continue
        fm_blend_a.append(ba); fm_blend_b.append(bb); fm_prob_a.append(elo_win_prob(ba, bb))

    preds["fm_blend_a"] = fm_blend_a
    preds["fm_blend_b"] = fm_blend_b
    preds["fm_prob_a"] = fm_prob_a
    preds["fm_pick"] = np.where(preds["fm_prob_a"] > 0.5, "A", np.where(preds["fm_prob_a"] < 0.5, "B", "TIE"))
    preds["fm_correct"] = np.where(
        preds["fm_prob_a"].isna(),
        np.nan,
        ((preds["fm_prob_a"] > 0.5) == preds["a_won"].astype(bool)).astype(int),
    )

    # Agreement filter
    preds["v2_fm_agree"] = (preds["model_pick"] == preds["fm_pick"]) & preds["fm_prob_a"].notna()
    # Ensemble avg (equal weight)
    preds["ensemble_prob_a"] = (preds["prob_a"] + preds["fm_prob_a"]) / 2
    preds["ensemble_pick"] = np.where(preds["ensemble_prob_a"] > 0.5, "A", "B")
    preds["ensemble_correct"] = np.where(
        preds["fm_prob_a"].isna(),
        np.nan,
        ((preds["ensemble_prob_a"] > 0.5) == preds["a_won"].astype(bool)).astype(int),
    )

    total = len(preds)
    with_fm = preds["fm_prob_a"].notna().sum()
    baseline = preds["correct"].mean() * 100

    print(f"Total 2026 fights: {total}")
    print(f"With FM ratings:   {with_fm} ({with_fm/total*100:.1f}%)")
    print()
    print(f"BASELINE V2:         {baseline:5.2f}%  (n={total})")

    # FM-only accuracy
    fm_avail = preds[preds["fm_prob_a"].notna()]
    if len(fm_avail):
        fm_acc = fm_avail["fm_correct"].mean() * 100
        v2_on_fm = fm_avail["correct"].mean() * 100
        ens_acc = fm_avail["ensemble_correct"].mean() * 100
        print(f"V2 on FM-avail:      {v2_on_fm:5.2f}%  (n={len(fm_avail)})")
        print(f"FM Blend standalone: {fm_acc:5.2f}%  (n={len(fm_avail)})")
        print(f"Ensemble avg (V2+FM):{ens_acc:5.2f}%  (n={len(fm_avail)})")

    # Agreement filter
    agree = preds[preds["v2_fm_agree"] & preds["fm_prob_a"].notna()]
    disagree = preds[~preds["v2_fm_agree"] & preds["fm_prob_a"].notna()]
    if len(agree):
        print(f"\n>>> V2 and FM AGREE:       V2 acc={agree['correct'].mean()*100:5.2f}%  (n={len(agree)})")
        print(f"    (same rows, FM acc:     {agree['fm_correct'].mean()*100:5.2f}%)")
        print(f"    (same rows, Ensemble:   {agree['ensemble_correct'].mean()*100:5.2f}%)")
    if len(disagree):
        print(f">>> V2 and FM DISAGREE:    V2 acc={disagree['correct'].mean()*100:5.2f}%  (n={len(disagree)})")
        print(f"    (same rows, FM acc:     {disagree['fm_correct'].mean()*100:5.2f}%)")

    # By V2 confidence bucket, filtered on agreement
    print("\nBy V2 confidence bucket (all vs agreement-filtered):")
    for bucket in sorted(preds["confidence_bucket"].dropna().unique()):
        sub_all = preds[preds["confidence_bucket"] == bucket]
        sub_agree = agree[agree["confidence_bucket"] == bucket]
        base = sub_all["correct"].mean() * 100
        n_all = len(sub_all)
        if len(sub_agree):
            ag = sub_agree["correct"].mean() * 100
            n_ag = len(sub_agree)
            print(f"  {bucket:18s}  all: {base:5.2f}% (n={n_all:3d})   agreeing: {ag:5.2f}% (n={n_ag:3d})")
        else:
            print(f"  {bucket:18s}  all: {base:5.2f}% (n={n_all:3d})   agreeing: (none)")

    # Weight-class red flags with agreement filter
    print("\nBy weight class (all vs agreement-filtered, top 8):")
    for wc in preds["weight_class"].value_counts().head(8).index:
        sub_all = preds[preds["weight_class"] == wc]
        sub_agree = agree[agree["weight_class"] == wc]
        base = sub_all["correct"].mean() * 100
        n_all = len(sub_all)
        if len(sub_agree) >= 3:
            ag = sub_agree["correct"].mean() * 100
            n_ag = len(sub_agree)
            delta = ag - base
            print(f"  {wc:20s}  all: {base:5.2f}% (n={n_all:3d})   agreeing: {ag:5.2f}% (n={n_ag:3d})  Δ={delta:+.1f}pp")

    preds.to_csv(OUT_CSV, index=False)
    findings = {
        "total": int(total),
        "with_fm_ratings": int(with_fm),
        "baseline_v2_pct": round(float(baseline), 2),
        "v2_on_fm_avail_pct": round(float(preds[preds["fm_prob_a"].notna()]["correct"].mean() * 100), 2) if with_fm else None,
        "fm_standalone_pct": round(float(preds[preds["fm_prob_a"].notna()]["fm_correct"].mean() * 100), 2) if with_fm else None,
        "ensemble_avg_pct": round(float(preds[preds["fm_prob_a"].notna()]["ensemble_correct"].mean() * 100), 2) if with_fm else None,
        "v2_fm_agree_n": int(len(agree)),
        "v2_fm_agree_pct": round(float(agree["correct"].mean() * 100), 2) if len(agree) else None,
        "v2_fm_disagree_n": int(len(disagree)),
        "v2_fm_disagree_pct": round(float(disagree["correct"].mean() * 100), 2) if len(disagree) else None,
    }
    OUT_JSON.write_text(json.dumps(findings, indent=2))
    print(f"\nSaved: {OUT_CSV.name}, {OUT_JSON.name}")


if __name__ == "__main__":
    main()
