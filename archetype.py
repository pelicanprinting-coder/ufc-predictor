"""
CagePicks Apex Archetype Engine — Python port of client/src/engine/archetype.ts
Ports classifyArchetype (pm), ARCHETYPE_MATRIX (ww), archetypeAdvantage (hm),
upsetScore (bw), and confidenceMultiplier (Sw) verbatim.

INPUTS EXPECTED per fighter (dict or pandas Series):
  kd          -- knockdown rate per 100 sig strikes (Apex format, e.g. 2.9 for 2.9%)
  slpm        -- sig strikes landed per minute
  sapm        -- sig strikes absorbed per minute
  sdiff       -- sig strike diff (slpm - sapm)
  td15        -- takedowns per 15 min
  ctrl        -- control time per 15 min (in minutes)
  subpct      -- submission win % (Apex format, e.g. 4.0 for 4%; percentage 0-100)
  finrate     -- finish rate (Apex format, e.g. 71.0 for 71%; percentage 0-100)
  ml          -- moneyline (American, optional; only used for upsetScore)
  age         -- years (int/float)
  fights      -- total UFC fights (int)
  l3f         -- list of {"result": "W"/"L"/"D"/"NC", "method": ...}, most-recent first
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import math

ARCHETYPES = [
    "KO Artist",
    "Volume Striker",
    "Counter Striker",
    "Wrestler",
    "Sub Specialist",
    "Point Fighter",
]

# 6x6 matchup matrix (attacker win % vs defender). Verbatim from `ww` in archetype.ts.
ARCHETYPE_MATRIX: Dict[str, Dict[str, float]] = {
    "KO Artist": {
        "KO Artist": 50.0,
        "Volume Striker": 52.7,
        "Counter Striker": 48.6,
        "Wrestler": 53.7,
        "Sub Specialist": 41.3,
        "Point Fighter": 57.7,
    },
    "Volume Striker": {
        "KO Artist": 47.3,
        "Volume Striker": 50.0,
        "Counter Striker": 38.7,
        "Wrestler": 44.5,
        "Sub Specialist": 45.9,
        "Point Fighter": 50.9,
    },
    "Counter Striker": {
        "KO Artist": 51.4,
        "Volume Striker": 61.3,
        "Counter Striker": 50.0,
        "Wrestler": 52.1,
        "Sub Specialist": 45.6,
        "Point Fighter": 60.7,
    },
    "Wrestler": {
        "KO Artist": 46.3,
        "Volume Striker": 55.5,
        "Counter Striker": 47.9,
        "Wrestler": 50.0,
        "Sub Specialist": 39.4,
        "Point Fighter": 56.7,
    },
    "Sub Specialist": {
        "KO Artist": 58.7,
        "Volume Striker": 54.1,
        "Counter Striker": 54.4,
        "Wrestler": 60.6,
        "Sub Specialist": 50.0,
        "Point Fighter": 58.0,
    },
    "Point Fighter": {
        "KO Artist": 42.3,
        "Volume Striker": 49.1,
        "Counter Striker": 39.3,
        "Wrestler": 43.3,
        "Sub Specialist": 42.0,
        "Point Fighter": 50.0,
    },
}


def _r(b: float, N: float) -> float:
    """Bounded ratio [0, 1]. Direct port of TypeScript `r`."""
    if N == 0:
        return 0.0
    return max(0.0, min(1.0, b / N))


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def classify_archetype(n: Dict[str, Any]) -> Dict[str, Any]:
    """
    Port of `classifyArchetype` (pm) from archetype.ts.
    Returns {"archetype": str, "scores": dict, "reasons": list[str]}.
    """
    kd = float(n.get("kd", 0) or 0)
    slpm = float(n.get("slpm", 0) or 0)
    sapm = float(n.get("sapm", 0) or 0)
    sdiff = float(n.get("sdiff", 0) or 0)
    td15 = float(n.get("td15", 0) or 0)
    ctrl = float(n.get("ctrl", 0) or 0)
    subpct = float(n.get("subpct", 0) or 0)
    finrate = float(n.get("finrate", 0) or 0)

    s = _r(kd, 3) * 0.45 + _r(finrate, 80) * 0.30 + _r(max(0, sdiff), 3) * 0.25
    l = _r(slpm, 7.5) * 0.55 + _r(sapm, 5.4) * 0.20 + (1 - _r(kd, 3)) * 0.25
    u = _r(max(0, sdiff), 3) * 0.50 + (1 - _r(slpm, 7.5)) * 0.20 + (1 - _r(sapm, 5.4)) * 0.30
    c = _r(td15, 5) * 0.60 + _r(ctrl, 5) * 0.30 + (1 - _r(subpct, 50)) * 0.10
    f = _r(subpct, 50) * 0.55 + _r(td15, 5) * 0.25 + _r(ctrl, 5) * 0.20
    h = td15 >= 2.5 or ctrl >= 2.5
    m = (1 - _r(slpm, 7.5)) * 0.45 + (0 if h else 0.30) + (1 - _r(finrate, 80)) * 0.25

    scores = {
        "KO Artist": s,
        "Volume Striker": l,
        "Counter Striker": u,
        "Wrestler": c,
        "Sub Specialist": f,
        "Point Fighter": m,
    }

    g = max(scores.items(), key=lambda kv: kv[1])[0]

    reasons: List[str] = []
    if g == "KO Artist":
        reasons.append(f"{kd:.1f}% KD rate")
        reasons.append(f"{finrate:.0f}% finish rate")
    elif g == "Volume Striker":
        reasons.append(f"{slpm:.1f} SLpM")
        if sdiff > 0:
            reasons.append(f"+{sdiff:.1f} strike diff")
        else:
            reasons.append(f"{kd:.1f}% KD rate")
    elif g == "Counter Striker":
        reasons.append(f"+{sdiff:.1f} strike diff")
        reasons.append(f"{sapm:.1f} absorbed/min")
    elif g == "Wrestler":
        reasons.append(f"{td15:.1f} TD/15")
        reasons.append(f"{ctrl:.1f} ctrl/15")
    elif g == "Sub Specialist":
        reasons.append(f"{subpct:.0f}% sub-win rate")
        if td15 > 0:
            reasons.append(f"{td15:.1f} TD/15")
    else:  # Point Fighter
        reasons.append(f"{slpm:.1f} SLpM")
        reasons.append(f"{finrate:.0f}% finish rate")

    return {"archetype": g, "scores": scores, "reasons": reasons}


def archetype_of(n: Dict[str, Any]) -> str:
    """Port of `archetypeOf` (qn). Returns just the archetype string."""
    return classify_archetype(n)["archetype"]


def archetype_advantage(a: str, b: str) -> float:
    """Port of `archetypeAdvantage` (hm). Returns attacker win prob (0-1) for arch a vs b."""
    return ARCHETYPE_MATRIX.get(a, {}).get(b, 50.0) / 100.0


def upset_score(n: Dict[str, Any], r: Dict[str, Any]) -> float:
    """
    Port of `upsetScore` (bw). Computes upset likelihood.
    n = potential upset fighter, r = opponent.
    Returns -1 to 1 (positive = upset likely).
    """
    s = archetype_of(n)
    l = archetype_of(r)
    c = (archetype_advantage(s, l) - 0.5) * 4
    c = _clamp(c, -1, 1)

    n_ml = n.get("ml")
    r_ml = r.get("ml")
    if n_ml is not None and r_ml is not None and n_ml > r_ml:
        # Underdog-specific boosters
        if float(n.get("fights", 0) or 0) > 12:
            c += 0.08
        l3f_n = n.get("l3f") or []
        if len(l3f_n) >= 3 and all(entry.get("result") == "W" for entry in l3f_n[:3]):
            c += 0.07
        try:
            age_diff = float(r.get("age", 0) or 0) - float(n.get("age", 0) or 0)
            if age_diff >= 5:
                c += 0.06
        except (TypeError, ValueError):
            pass
        if float(r.get("age", 0) or 0) >= 35:
            c += 0.06
        l3f_r = r.get("l3f") or []
        if l3f_r and l3f_r[0].get("result") == "L":
            c += 0.05

    return _clamp(c, -1, 1)


def confidence_multiplier(n_fights: float) -> float:
    """Port of `confidenceMultiplier` (Sw). Discount for small-sample fighters."""
    if n_fights >= 10:
        return 1.0
    if n_fights >= 5:
        return 0.85
    if n_fights >= 3:
        return 0.75
    if n_fights >= 1:
        return 0.65
    return 0.5


# ------------------------------------------------------------------
# CLI test
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Sanity check against Apex UFC 329 CSV (McGregor / Holloway)
    mcgregor = {
        "kd": 2.9, "slpm": 5.32, "sapm": 4.66, "sdiff": 0.66,
        "td15": 1.11, "ctrl": 1.85, "subpct": 4.0, "finrate": 71.0,
        "age": 38, "fights": 14, "ml": 190,
        "l3f": [{"result": "L", "method": "TKO"}, {"result": "L", "method": "TKO"}, {"result": "W", "method": "TKO"}],
    }
    holloway = {
        "kd": 0.4, "slpm": 6.91, "sapm": 4.61, "sdiff": 2.30,
        "td15": 0.27, "ctrl": 0.82, "subpct": 5.0, "finrate": 37.0,
        "age": 34.6, "fights": 32, "ml": -220,
        "l3f": [{"result": "W", "method": "TKO"}, {"result": "W", "method": "UD"}, {"result": "L", "method": "UD"}],
    }
    m = classify_archetype(mcgregor)
    h = classify_archetype(holloway)
    print(f"McGregor: {m['archetype']}")
    print(f"  scores: {m['scores']}")
    print(f"  reasons: {m['reasons']}")
    print(f"Holloway: {h['archetype']}")
    print(f"  scores: {h['scores']}")
    print(f"  reasons: {h['reasons']}")
    print(f"Archetype advantage (McG vs Hol): {archetype_advantage(m['archetype'], h['archetype']):.3f}")
    print(f"Upset score (McG as underdog): {upset_score(mcgregor, holloway):.3f}")
    print(f"Confidence mult (McG, 14 fights): {confidence_multiplier(14)}")
