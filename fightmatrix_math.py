"""
Fight Matrix Elo/Glicko/WHR helpers.

Standard Elo win-probability formula:
    P(A beats B) = 1 / (1 + 10 ** ((rating_B - rating_A) / 400))

Fight Matrix scale is Elo-style so this transfers directly. On the Program
page they publish a per-fight win % which we can verify against.

For the user's requested blend:
    blended_rating = 0.45 * glicko1 + 0.55 * whr
    blended_win_prob = elo_win_prob(blended_a, blended_b)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


def elo_win_prob(rating_a: float, rating_b: float) -> float:
    """Standard Elo formula. Returns P(A beats B)."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


@dataclass
class FMComparison:
    fighter_a: str
    fighter_b: str
    system_probs: Dict[str, tuple]  # system -> (prob_a, prob_b)
    ratings_a: Dict[str, Optional[float]]
    ratings_b: Dict[str, Optional[float]]
    blended_prob_a: float
    blended_prob_b: float
    blended_rating_a: Optional[float]
    blended_rating_b: Optional[float]
    consensus_verdict: str  # e.g. "3/4 systems favor A"
    largest_disagreement: float  # spread in prob across systems
    fm_profile_a: str
    fm_profile_b: str


def _get_rating(row: pd.Series, col: str) -> Optional[float]:
    if row is None:
        return None
    v = row.get(col)
    if v is None or (isinstance(v, str) and not v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_fm_comparison(
    fighter_a: str, fighter_b: str,
    row_a: Optional[pd.Series], row_b: Optional[pd.Series],
    glicko_weight: float = 0.45, whr_weight: float = 0.55,
) -> Optional[FMComparison]:
    """Compute per-system win probs and user-blended win prob.

    Requires either full or partial FM row for each fighter. Returns None if
    both fighters have no rating data at all.
    """
    if row_a is None and row_b is None:
        return None

    systems = ["glicko1", "whr", "k170"]
    system_probs: Dict[str, tuple] = {}
    ratings_a = {s: _get_rating(row_a, f"fm_{s}") for s in systems}
    ratings_b = {s: _get_rating(row_b, f"fm_{s}") for s in systems}

    for s in systems:
        ra, rb = ratings_a[s], ratings_b[s]
        if ra is None or rb is None:
            continue
        pa = elo_win_prob(ra, rb)
        system_probs[s] = (pa, 1.0 - pa)

    # Blended (Glicko-1 + WHR)
    ba_g = ratings_a["glicko1"]
    bb_g = ratings_b["glicko1"]
    ba_w = ratings_a["whr"]
    bb_w = ratings_b["whr"]
    blended_a = None
    blended_b = None
    if all(v is not None for v in (ba_g, bb_g, ba_w, bb_w)):
        blended_a = glicko_weight * ba_g + whr_weight * ba_w
        blended_b = glicko_weight * bb_g + whr_weight * bb_w
    blended_prob_a = (
        elo_win_prob(blended_a, blended_b) if blended_a is not None and blended_b is not None else float("nan")
    )
    blended_prob_b = 1.0 - blended_prob_a if blended_prob_a == blended_prob_a else float("nan")

    # Consensus verdict
    if system_probs:
        favor_a = sum(1 for (pa, _) in system_probs.values() if pa > 0.5)
        total = len(system_probs)
        favor = fighter_a if favor_a > total / 2 else (fighter_b if favor_a < total / 2 else "split")
        if favor == "split":
            verdict = f"{total}/{total} systems split evenly"
        else:
            n = favor_a if favor == fighter_a else total - favor_a
            verdict = f"{n}/{total} systems favor {favor}"
        probs = [p[0] for p in system_probs.values()]
        largest = max(probs) - min(probs)
    else:
        verdict = "insufficient data"
        largest = 0.0

    return FMComparison(
        fighter_a=fighter_a,
        fighter_b=fighter_b,
        system_probs=system_probs,
        ratings_a=ratings_a,
        ratings_b=ratings_b,
        blended_prob_a=blended_prob_a,
        blended_prob_b=blended_prob_b,
        blended_rating_a=blended_a,
        blended_rating_b=blended_b,
        consensus_verdict=verdict,
        largest_disagreement=largest,
        fm_profile_a=str(row_a.get("fm_profile_url", "")) if row_a is not None else "",
        fm_profile_b=str(row_b.get("fm_profile_url", "")) if row_b is not None else "",
    )


def three_way_consensus(
    model_prob_a: float, blended_prob_a: float, market_prob_a: float,
    threshold: float = 0.03,
) -> str:
    """Return a consensus label based on model + FM blend + market.

    All three probs are P(A wins).
    """
    if any(p != p for p in (model_prob_a, blended_prob_a, market_prob_a)):
        return "INCOMPLETE"
    model_side = "A" if model_prob_a > 0.5 else "B"
    blend_side = "A" if blended_prob_a > 0.5 else "B"
    market_side = "A" if market_prob_a > 0.5 else "B"

    model_edge = model_prob_a - market_prob_a
    blend_edge = blended_prob_a - market_prob_a

    # Strong: both non-market signals agree with each other AND disagree with market by >= threshold
    if (
        model_side == blend_side
        and model_side != market_side
        and abs(model_edge) >= threshold
        and abs(blend_edge) >= threshold
    ):
        return "STRONG_CONSENSUS"
    if model_side == blend_side and model_side != market_side:
        return "WEAK_CONSENSUS"
    if model_side != blend_side:
        return "SPLIT"
    if model_side == market_side:
        return "MARKET_ALIGNED"
    return "NEUTRAL"
