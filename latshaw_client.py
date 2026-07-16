"""
Latshaw data client.
Reads the four bundled CSV snapshots from data/latshaw/ (extracted from
https://natelatshaw.shinyapps.io/ufc_fight_night_statistical_companion/)
and provides per-fighter lookups mapped to the APEX-ENGINE schema.

Source of truth for column definitions is Scott's APEX-ENGINE.csv.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "latshaw")


def _pct(x) -> Optional[float]:
    """Strip trailing % and return float. Returns None if unparseable."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).replace("%", "").replace(",", "").strip()
    if s in ("", "-", "NA", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _num(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).replace(",", "").strip()
    if s in ("", "-", "NA", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


@lru_cache(maxsize=1)
def _load_all():
    welcome = pd.read_csv(os.path.join(_DATA_DIR, "latshaw_welcome.csv"))
    fo = pd.read_csv(os.path.join(_DATA_DIR, "latshaw_fight_outcome.csv"))
    sos = pd.read_csv(os.path.join(_DATA_DIR, "latshaw_strength_schedule.csv"))
    strk = pd.read_csv(os.path.join(_DATA_DIR, "latshaw_striking.csv"))
    wr = pd.read_csv(os.path.join(_DATA_DIR, "latshaw_wrestling.csv"))
    # Normalize Name column for case-insensitive lookup
    for df in (welcome, fo, sos, strk, wr):
        df["_name_key"] = df["Name"].astype(str).str.lower().str.strip()
    return welcome, fo, sos, strk, wr


def event_names() -> list[str]:
    """Return list of distinct event names in the Latshaw data (usually the
    next 1-2 upcoming UFC events)."""
    welcome, _, _, _, _ = _load_all()
    return welcome["Event"].dropna().unique().tolist()


def available_fighters(event: str | None = None) -> list[str]:
    welcome, _, _, _, _ = _load_all()
    if event:
        welcome = welcome[welcome["Event"] == event]
    return sorted(welcome["Name"].dropna().unique().tolist())


def event_name() -> str:
    """Return the primary (first) event name."""
    return event_names()[0]


def bout_list(event: str | None = None) -> list[tuple[str, str, int]]:
    """Return list of (fighter_a, fighter_b, bout_order) tuples in card order,
    filtered to a specific event (defaults to the first event in the data).
    Bout order 1 is the main event."""
    welcome, _, _, _, _ = _load_all()
    if event is None:
        event = event_names()[0]
    df = welcome[welcome["Event"] == event]
    bouts: dict[int, dict] = {}
    for _, r in df.iterrows():
        try:
            bo = int(r["Bout Order"])
        except Exception:
            continue
        side = str(r["Side"]).strip().upper()
        bouts.setdefault(bo, {})[side] = str(r["Name"])
    return [(bouts[bo].get("A", ""), bouts[bo].get("B", ""), bo) for bo in sorted(bouts.keys())]


def get_welcome_row(name: str) -> Optional[dict]:
    """Return welcome-table fields for a fighter: age, reach, height, stance,
    pro_record, weight_class, event, date. Values are Latshaw-formatted strings
    (e.g. '32.5 years', '76in', '6ft 1in')."""
    welcome, _, _, _, _ = _load_all()
    key = name.lower().strip()
    m = welcome[welcome["_name_key"] == key]
    if m.empty:
        return None
    r = m.iloc[0]
    return {
        "age": str(r["Age"]),
        "reach": str(r["Reach"]),
        "height": str(r["Height"]),
        "stance": str(r["Stance"]),
        "pro_record": str(r["Pro Record"]),
        "weight_class": str(r["Weight Class"]),
        "event": str(r["Event"]),
        "date": str(r["Date"]),
        "bout_order": str(r["Bout Order"]),
    }


def get_row(name: str) -> Optional[dict]:
    """
    Return a dict of Latshaw-derived fields mapped to APEX-schema keys.
    Keys returned: age, reach, height, stance, pro_record, ufc_record,
    weight_class, ufc_fight_count, ufc_minutes_fought, xr_pct, finish_rate_ufc,
    opp_ufc_win_pct, opp_xr_pct, slpm, sapm, sig_strike_diff,
    head_strike_landed_pct, head_absorption_pct, kd_rate, kd_absorbed_rate,
    td_defense_pct, td_per_15, ctrl_time_per_15, sub_win_pct, ctrl_absorbed_per_15.

    Missing values are None. Returns None if fighter not on the card.
    """
    welcome, fo, sos, strk, wr = _load_all()
    key = name.lower().strip()

    fo_m = fo[fo["_name_key"] == key]
    if fo_m.empty:
        return None
    fo_r = fo_m.iloc[0]

    # Merge welcome-table fields first (age, reach, height, stance, weight_class)
    wel = get_welcome_row(name) or {}

    sos_r = sos[sos["_name_key"] == key].iloc[0] if not sos[sos["_name_key"] == key].empty else None
    strk_r = strk[strk["_name_key"] == key].iloc[0] if not strk[strk["_name_key"] == key].empty else None
    wr_r = wr[wr["_name_key"] == key].iloc[0] if not wr[wr["_name_key"] == key].empty else None

    row: dict = dict(wel)  # start with welcome fields

    # ------- Fight Outcome table -------
    ufc_wins_losses_draws = str(fo_r["UFC Record"]).strip()
    row["ufc_record"] = ufc_wins_losses_draws
    row["pro_record"] = str(fo_r["MMA Record"]).strip()
    row["ufc_minutes_fought"] = _num(fo_r["Fight Minutes"])
    row["xr_pct"] = _pct(fo_r["ADVANCED_xR%"])
    # UFC-only finish rate (FIN% among wins) — different from APEX career-based finish_rate
    row["finish_rate_ufc"] = _pct(fo_r["AmongWins_FIN%"])
    # ufc_fight_count = wins + losses + draws
    try:
        w, l, d = [int(p) for p in ufc_wins_losses_draws.split("-")]
        row["ufc_fight_count"] = w + l + d
        row["ufc_wins"] = w
        row["ufc_losses"] = l
        row["ufc_draws"] = d
    except Exception:
        row["ufc_fight_count"] = None
        row["ufc_wins"] = None
        row["ufc_losses"] = None
        row["ufc_draws"] = None

    # ------- Strength of Schedule table -------
    if sos_r is not None:
        row["opp_ufc_win_pct"] = _pct(sos_r["AVERAGE OPPONENT UFC CAREER STATISTICS_Avg Win%"])
        row["opp_xr_pct"] = _pct(sos_r["AVERAGE OPPONENT UFC CAREER STATISTICS_Avg xR%"])

    # ------- Striking table -------
    if strk_r is not None:
        dist_time_pct = _pct(strk_r["DISTANCE STRIKING PREVALENCE_% of Time"])
        dist_slpm = _num(strk_r["DISTANCE SIGNIFICANT STRIKES (PER MINUTE AT DISTANCE)_Landed"])
        dist_sapm = _num(strk_r["DISTANCE SIGNIFICANT STRIKES (PER MINUTE AT DISTANCE)_Absorbed"])
        row["head_strike_landed_pct"] = _pct(strk_r["HEAD SIGNIFICANT STRIKES_% of Strikes Landed"])
        row["head_absorption_pct"] = _pct(strk_r["HEAD SIGNIFICANT STRIKES_% of Strikes Absorbed"])
        row["kd_rate"] = _pct(strk_r["KNOCKDOWNS_Land Rate"])
        row["kd_absorbed_rate"] = _pct(strk_r["KNOCKDOWNS_Concede Rate"])

        # Reconstruct overall SLpM/SApM/Diff by combining distance + clinch/ground shares
        if wr_r is not None:
            cg_time_pct = _pct(wr_r["CLINCH + GROUND STRIKING PREVALENCE_% of Time"])
            cg_slpm = _num(wr_r["CLINCH + GROUND STRIKING PACE (PER MINUTE IN CLINCH/GROUND)_Landed"])
            cg_sapm = _num(wr_r["CLINCH + GROUND STRIKING PACE (PER MINUTE IN CLINCH/GROUND)_Absorbed"])
            if None not in (dist_slpm, dist_sapm, dist_time_pct, cg_slpm, cg_sapm, cg_time_pct):
                row["slpm"] = round(dist_slpm * dist_time_pct / 100 + cg_slpm * cg_time_pct / 100, 2)
                row["sapm"] = round(dist_sapm * dist_time_pct / 100 + cg_sapm * cg_time_pct / 100, 2)
                row["sig_strike_diff"] = round(row["slpm"] - row["sapm"], 2)

    # ------- Wrestling/Grappling table -------
    if wr_r is not None:
        row["td_defense_pct"] = _pct(wr_r["TAKEDOWN EFFICIENCY_Defense"])
        # APEX td_per_15 uses the distance-time denominator (TDs are attempted
        # from distance), matching UFCStats' Td.Avg style: landed / (mins × dist%) × 15.
        # Verified vs APEX-ENGINE.csv: DP 3.16 vs 3.18, Cannonier 0.70 vs 0.69,
        # Usman 5.36 vs 5.34 (rounding within ±0.02).
        td_landed = _num(wr_r["TAKEDOWN EFFICIENCY_Landed"])
        mins = _num(wr_r["WRESTLING/GRAPPLING PREVALENCE & EFFICIENCY_Fight Minutes"])
        dist_pct_wr = _pct(wr_r["WRESTLING/GRAPPLING PREVALENCE & EFFICIENCY_Distance %"])
        if td_landed is not None and mins and mins > 0 and dist_pct_wr:
            row["td_per_15"] = round(td_landed / (mins * dist_pct_wr / 100) * 15, 2)
        # Control time per 15 = control% × 15 (share of fight time is spent controlling)
        ctrl_pct = _pct(wr_r["WRESTLING/GRAPPLING PREVALENCE & EFFICIENCY_Control %"])
        if ctrl_pct is not None:
            row["ctrl_time_per_15"] = round(ctrl_pct / 100 * 15, 2)
        controlled_pct = _pct(wr_r["WRESTLING/GRAPPLING PREVALENCE & EFFICIENCY_Controlled %"])
        if controlled_pct is not None:
            row["ctrl_absorbed_per_15"] = round(controlled_pct / 100 * 15, 2)
        # NOTE: APEX sub_win_pct = career sub wins / total fights (career metric).
        # Latshaw's SUBMISSION ATTEMPTS Success Rate is a per-attempt UFC metric,
        # which is a different definition. Store as sub_attempt_success_rate
        # so it's available but doesn't get mislabeled into APEX sub_win_pct.
        sub_success = _pct(wr_r["SUBMISSION ATTEMPTS_Success Rate"])
        if sub_success is not None:
            row["sub_attempt_success_rate"] = sub_success / 100

    return row



