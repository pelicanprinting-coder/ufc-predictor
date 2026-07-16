"""
Look-ahead-SAFE backtest of the blended ELO (0.45*Glicko-1 + 0.55*WHR).

The user provided historical fight-time ELO in fightmatrix_history_2026.csv:
each row has ELO = as-of value on the day of the fight. This gives us a
clean out-of-sample test with no data leakage.

We test:
1. FM Blended ELO standalone (higher ELO -> pick to win)
2. Consensus: ELO pick vs OPEN market pick (agree = higher-confidence bucket)
3. ELO/IMPLIED disagreement filter (chase where ELO says value)

Baseline reference: V2 model = 60.0% on 2026 slice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
HIST = ROOT / "fightmatrix_history_2026.csv"
OUT_JSON = ROOT / "backtest_fm_historical.json"


def _american_to_implied(american) -> float:
    try:
        a = float(american)
    except (TypeError, ValueError):
        return float("nan")
    if a > 0:
        return 100.0 / (a + 100.0)
    if a < 0:
        return -a / (-a + 100.0)
    return float("nan")


def _pair_up(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the long-format (2 rows per fight) CSV into wide pair rows.

    Each event is a block; consecutive pairs of rows are the two sides of
    one fight (that's how the user's spreadsheet is laid out).
    """
    pairs = []
    for event, grp in df.groupby("Event", sort=False):
        rows = grp.reset_index(drop=True)
        # Each pair is (2i, 2i+1). Drop trailing odd row if any.
        n_pairs = len(rows) // 2
        for i in range(n_pairs):
            a = rows.iloc[2 * i]
            b = rows.iloc[2 * i + 1]
            pairs.append({
                "event": event,
                "fighter_a": a["FIGHTER"],
                "fighter_b": b["FIGHTER"],
                "open_a": a["OPEN"],
                "open_b": b["OPEN"],
                "current_a": a["CURRENT"],
                "current_b": b["CURRENT"],
                "implied_a": a["IMPLIED"],
                "implied_b": b["IMPLIED"],
                "elo_a": a["ELO"],
                "elo_b": b["ELO"],
                "result_a": a["RESULTS"],
                "result_b": b["RESULTS"],
                "method": a.get("METHOD"),
                "round": a.get("ROUND"),
                "elo_over_implied_a": a.get("ELO/IMPLIED"),
                "elo_over_implied_b": b.get("ELO/IMPLIED"),
            })
    return pd.DataFrame(pairs)


def _clean_result(r) -> str | None:
    if pd.isna(r):
        return None
    r = str(r).strip().upper()
    if r in ("W", "L"):
        return r
    return None  # NC / MD / TBD


def _pair_accuracy(pairs: pd.DataFrame, pick_a_col: str) -> dict:
    """Given a boolean pick_a_col ('a is winner picked'), compute accuracy."""
    valid = pairs.dropna(subset=[pick_a_col, "result_a"]).copy()
    valid["correct"] = (
        (valid[pick_a_col].astype(bool)) & (valid["result_a"] == "W")
    ) | (
        (~valid[pick_a_col].astype(bool)) & (valid["result_a"] == "L")
    )
    n = len(valid)
    return {"n": int(n), "acc": float(valid["correct"].mean() * 100) if n else float("nan")}


def main():
    df = pd.read_csv(HIST)
    # Restrict to W/L completed fights
    df = df[df["RESULTS"].isin(["W", "L"])].copy()

    pairs = _pair_up(df)
    # Filter to pairs with both sides in W/L (drop mismatches)
    pairs = pairs[pairs["result_a"].isin(["W", "L"]) & pairs["result_b"].isin(["W", "L"])].copy()
    pairs = pairs[pairs["result_a"] != pairs["result_b"]].copy()  # sanity
    print(f"Pair rows: {len(pairs)}")

    # ELO pick: whichever side has higher ELO wins
    pairs["elo_pick_a"] = pairs["elo_a"] > pairs["elo_b"]

    # Market pick (open price): more-negative American number is the favorite
    def _open_pick(row):
        try:
            oa = float(row["open_a"]); ob = float(row["open_b"])
        except (TypeError, ValueError):
            return np.nan
        # Convert to implied probability for a numeric comparison
        pa = _american_to_implied(oa); pb = _american_to_implied(ob)
        if pa != pa or pb != pb:
            return np.nan
        return pa > pb
    pairs["market_pick_a"] = pairs.apply(_open_pick, axis=1)

    # Winner flag
    pairs["winner_a"] = pairs["result_a"] == "W"

    # === Metric 1: ELO standalone ===
    elo_res = _pair_accuracy(pairs, "elo_pick_a")
    print(f"\n1) FM Blended ELO standalone: {elo_res['acc']:.2f}% (n={elo_res['n']})")

    # === Metric 2: Market opening line standalone ===
    market_res = _pair_accuracy(pairs.dropna(subset=["market_pick_a"]), "market_pick_a")
    print(f"2) Market OPEN line standalone: {market_res['acc']:.2f}% (n={market_res['n']})")

    # === Metric 3: Agreement filter (ELO and market both agree) ===
    agree_mask = (pairs["elo_pick_a"] == pairs["market_pick_a"]) & pairs["market_pick_a"].notna()
    agree = pairs[agree_mask]
    disagree = pairs[~agree_mask & pairs["market_pick_a"].notna()]
    agree_res = _pair_accuracy(agree, "elo_pick_a")
    disagree_res_elo = _pair_accuracy(disagree, "elo_pick_a")
    disagree_res_market = _pair_accuracy(disagree, "market_pick_a")
    print(f"\n3a) ELO + Market AGREE  -> pick: {agree_res['acc']:.2f}% (n={agree_res['n']})")
    print(f"3b) ELO + Market DISAGREE -> ELO pick: {disagree_res_elo['acc']:.2f}% (n={disagree_res_elo['n']})")
    print(f"3c) ELO + Market DISAGREE -> Market pick: {disagree_res_market['acc']:.2f}% (n={disagree_res_market['n']})")

    # === Metric 4: ELO/IMPLIED disagreement bucket (large edge only) ===
    # user's spreadsheet already computes ELO/IMPLIED per fighter — the higher-ELO
    # side with the largest ratio is the "value" pick. Take rows where ELO
    # disagrees with market and elo edge >= X%.
    def elo_prob_a(row):
        # Simple normalization: elo_a / (elo_a + elo_b) as a rough prob
        try:
            ea, eb = float(row["elo_a"]), float(row["elo_b"])
        except (TypeError, ValueError):
            return np.nan
        # These ELO values are already model-derived win% (0-100), not ratings
        # e.g. Gaethje 40.99 vs Pimblett 59.01 (sum = 100) -> already normalized
        # So this IS the P(A win)% directly
        return ea / (ea + eb) * 100
    pairs["elo_prob_a"] = pairs.apply(elo_prob_a, axis=1)
    pairs["elo_edge_a"] = pairs["elo_prob_a"] - pairs["implied_a"]

    for threshold in [3, 5, 8, 12]:
        big_edge = pairs[abs(pairs["elo_edge_a"]) >= threshold].copy()
        big_edge["pick_a"] = big_edge["elo_edge_a"] > 0  # pick the side with positive elo edge
        res = _pair_accuracy(big_edge, "pick_a")
        print(f"4) ELO/Market edge >= {threshold:2d}%: {res['acc']:.2f}% (n={res['n']})")

    # === Metric 5: Same as #4 but bet UNDERDOG only ===
    for threshold in [3, 5, 8]:
        big_edge = pairs[abs(pairs["elo_edge_a"]) >= threshold].copy()
        # Underdog side (implied < 50%)
        underdog = big_edge[
            ((big_edge["elo_edge_a"] > 0) & (big_edge["implied_a"] < 50)) |
            ((big_edge["elo_edge_a"] < 0) & (big_edge["implied_a"] > 50))
        ].copy()
        underdog["pick_a"] = underdog["elo_edge_a"] > 0
        res = _pair_accuracy(underdog, "pick_a")
        # ROI at avg opening price
        if len(underdog):
            underdog["pick_open"] = np.where(underdog["pick_a"], underdog["open_a"], underdog["open_b"])
            wins = underdog[underdog["pick_a"] == underdog["winner_a"]]
            payout_per_bet = wins["pick_open"].apply(lambda x: (x / 100.0) if x > 0 else (100.0 / abs(x))).sum()
            total_wagered = len(underdog) * 1.0
            roi = (payout_per_bet - (len(underdog) - len(wins))) / total_wagered * 100
            print(f"5) Underdog + edge>={threshold}%: {res['acc']:.2f}% (n={res['n']}), ROI={roi:+.1f}%")

    # By event
    print("\nBy event (ELO standalone):")
    for ev, grp in pairs.groupby("event"):
        r = _pair_accuracy(grp, "elo_pick_a")
        print(f"  {ev:15s}  {r['acc']:5.1f}% (n={r['n']})")

    findings = {
        "total_pairs": int(len(pairs)),
        "elo_standalone_pct": round(elo_res["acc"], 2),
        "market_open_standalone_pct": round(market_res["acc"], 2),
        "elo_market_agree_pct": round(agree_res["acc"], 2),
        "elo_market_agree_n": agree_res["n"],
        "elo_market_disagree_elo_pct": round(disagree_res_elo["acc"], 2),
        "elo_market_disagree_market_pct": round(disagree_res_market["acc"], 2),
        "elo_market_disagree_n": disagree_res_elo["n"],
    }
    OUT_JSON.write_text(json.dumps(findings, indent=2))
    print(f"\nSaved: {OUT_JSON.name}")


if __name__ == "__main__":
    main()
