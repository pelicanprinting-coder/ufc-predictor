"""
Card Prep tab: pull the upcoming UFC card straight from Nate Latshaw's
Statistical Companion snapshot, layer in FM Blended ELO, auto-fill every
APEX column we can, and highlight the still-missing cells.

APEX schema is the 78-column layout in Scott's APEX-ENGINE.csv (source of truth).

Data sources:
  - data/latshaw/latshaw_welcome.csv           (age, reach, height, stance, records)
  - data/latshaw/latshaw_fight_outcome.csv     (UFC record, minutes, xR, xR%)
  - data/latshaw/latshaw_strength_schedule.csv (opp UFC win%, opp xR%)
  - data/latshaw/latshaw_striking.csv          (per-min strikes, head/body/leg breakdowns)
  - data/latshaw/latshaw_wrestling.csv         (TD, control, sub attempts)
  - fightmatrix_ratings.csv                     (Glicko-1 + WHR → blended ELO)
"""

from __future__ import annotations

import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import latshaw_client as L
from opticodds_client import _load_snapshot


APEX_COLUMNS = [
    "Event", "fight_pair", "FIGHTER", "opponent_name", "weight_class",
    "scheduled_rounds", "short_notice", "dk_salary", "age", "reach", "height",
    "stance", "pro_record", "ufc_record", "ufc_fight_count", "career_ko_wins",
    "career_sub_wins", "career_ko_losses", "career_sub_losses", "slpm",
    "sig_strike_diff", "kd_rate", "sapm", "kd_absorbed_rate",
    "head_absorption_pct", "td_per_15", "ctrl_time_per_15", "sub_win_pct",
    "ufc_minutes_fought", "td_defense_pct", "ctrl_absorbed_per_15",
    "recent_fight_1_result", "recent_fight_1_method", "recent_fight_2_result",
    "recent_fight_2_method", "recent_fight_3_result", "recent_fight_3_method",
    "opp_ufc_win_pct", "opp_xr_pct", "xr_pct", "finish_rate", "elo", "OPEN",
    "CURRENT", "ml_win_by_ko", "ml_win_by_sub", "ml_win_by_dec", "avg_dk_pts",
    "dk_win_avg", "dk_loss_avg", "champ_experience", "Total Fights",
    "head_strike_landed_pct", "head_strike_absorbed_pct", "Missed_weight",
    "Long_layoff", "Elo_pct_secondary", "GPT", "SIM", "MMA-AI", "CAPPERS",
    "GROK", "CLAUDE", "GEMINI", "CHAT GPT", "DEEP SEEK", "AI-AVERAGE",
    "MODEL 75", "NEW MODEL", "SCRAP", "CAGE SCORE", "CAGE PICKS",
    "ELO/IMPLIED", "CI LOW", "CI HIGH", "RESULTS", "sig_strike_accuracy_pct",
    "sig_strike_defense_pct",
]


# Columns explicitly marked "Manually Loaded" in docs/apex_column_mapping.csv —
# these are the ONLY columns that should be highlighted yellow in the preview.
MANUAL_ONLY_COLUMNS = {
    "Event", "fight_pair", "short_notice", "dk_salary", "champ_experience",
    "Missed_weight", "Long_layoff", "Elo_pct_secondary",
    "GPT", "SIM", "MMA-AI", "CAPPERS", "GROK", "CLAUDE", "GEMINI",
    "CHAT GPT", "DEEP SEEK", "AI-AVERAGE", "MODEL 75", "NEW MODEL",
    "SCRAP", "CAGE SCORE", "CAGE PICKS", "ELO/IMPLIED", "CI LOW", "CI HIGH",
}


# --- Tapology / UFCStats / OpticOdds data loaders --------------------

@st.cache_data(show_spinner=False)
def _load_tapology() -> pd.DataFrame:
    """Career totals (KO/sub wins & losses, pro record, total fights)."""
    path = os.path.join(os.path.dirname(__file__),
                        "data", "tapology", "career_totals.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _load_recent_fights() -> pd.DataFrame:
    """Last 3 UFCStats results + methods."""
    path = os.path.join(os.path.dirname(__file__),
                        "data", "ufcstats", "recent_fights.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _load_card_odds() -> pd.DataFrame:
    """OpticOdds open + current + method-of-victory prices."""
    path = os.path.join(os.path.dirname(__file__),
                        "data", "opticodds", "card_odds.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _norm(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return "".join(c for c in name.lower() if c.isalnum())


def _lookup(df: pd.DataFrame, name: str, col: str = "name") -> dict | None:
    if df is None or df.empty or not name:
        return None
    target = _norm(name)
    if col not in df.columns:
        return None
    norm_series = df[col].astype(str).map(_norm)
    m = df[norm_series == target]
    if not m.empty:
        return m.iloc[0].to_dict()
    # Last-name / suffix fallback
    last = target[-6:] if len(target) > 6 else target
    m = df[norm_series.str.endswith(last)]
    if not m.empty:
        return m.iloc[0].to_dict()
    return None


# --- FM ELO lookup ----------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_fm_ratings() -> pd.DataFrame:
    """Load Fight Matrix ratings (Elo, Glicko-1, WHR + blended)."""
    path = os.path.join(os.path.dirname(__file__), "fightmatrix_ratings.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _fm_lookup(fm_df: pd.DataFrame, name: str) -> dict | None:
    if fm_df.empty or not name:
        return None
    n = name.strip().lower()
    # Try exact match on common name columns
    for col in ("fighter_name", "name", "Fighter"):
        if col in fm_df.columns:
            m = fm_df[fm_df[col].astype(str).str.lower() == n]
            if not m.empty:
                r = m.iloc[0]
                return {
                    "fm_glicko1": r.get("fm_glicko1", r.get("glicko1")),
                    "fm_whr": r.get("fm_whr", r.get("whr")),
                    "fm_blended": r.get("fm_blended"),
                }
    # Last-name fallback
    last = n.split()[-1]
    for col in ("fighter_name", "name", "Fighter"):
        if col in fm_df.columns:
            m = fm_df[fm_df[col].astype(str).str.lower().str.endswith(" " + last)]
            if not m.empty:
                r = m.iloc[0]
                return {
                    "fm_glicko1": r.get("fm_glicko1", r.get("glicko1")),
                    "fm_whr": r.get("fm_whr", r.get("whr")),
                    "fm_blended": r.get("fm_blended"),
                }
    return None


def _blended_elo(fm: dict | None) -> int | None:
    if not fm:
        return None
    g = fm.get("fm_glicko1")
    w = fm.get("fm_whr")
    if g is not None and w is not None and pd.notna(g) and pd.notna(w):
        return int(round(0.45 * float(g) + 0.55 * float(w)))
    # Fallback to precomputed blended
    b = fm.get("fm_blended")
    if b is not None and pd.notna(b):
        return int(round(float(b)))
    return None


# --- Formatting helpers ----------------------------------------------

def _pct_str(v, decimals: int = 0, already_pct: bool = True) -> str | None:
    """Format a numeric value into an APEX-style percent string.

    Latshaw returns most percent-like values as already-scaled percents
    (e.g. kd_rate=0.8 means 0.8%, not 80%; xr_pct=56 means 56%). Set
    already_pct=False for fraction inputs (0..1).
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not already_pct:
        f *= 100
    return f"{f:.{decimals}f}%" if decimals > 0 else f"{int(round(f))}%"


def _american(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        i = int(v)
        return f"+{i}" if i > 0 else str(i)
    except Exception:
        return None


# --- Row builder ------------------------------------------------------

def build_row(
    event_name: str,
    fight_pair_idx: int,
    lat: dict | None,
    lat_opp: dict | None,
    fm: dict | None,
    fighter_name: str,
    opponent_name: str,
    scheduled_rounds: int,
    open_odds: int | None = None,
    current_odds: int | None = None,
    tap: dict | None = None,
    recent: dict | None = None,
    odds_row: dict | None = None,
) -> dict:
    """
    Assemble one APEX row for a fighter using Latshaw + FM data.

    lat and lat_opp are dicts returned by latshaw_client.get_row() (already
    APEX-schema-aligned key names).
    """
    def g(key, default=None):
        if lat is None:
            return default
        v = lat.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return v

    # Blended ELO (FM Glicko-1 × 0.45 + WHR × 0.55)
    elo = _blended_elo(fm)

    # Finish rate — APEX definition = (career_ko_wins + career_sub_wins) / Total Fights.
    # Latshaw only gives UFC AmongWins FIN%, so this is manual (yellow).
    # ufc_record, pro_record, weight_class come straight from Latshaw
    ufc_record = g("ufc_record")

    # Age formatting: Latshaw already gives "32.5 years" — keep as-is
    age_fmt = g("age")
    # Reach: "76in" already formatted
    reach_fmt = g("reach")
    # Height: "6ft 1in" already formatted
    height_fmt = g("height")

    # Format each column to match APEX-ENGINE's storage type:
    #   xr_pct               → integer string like "56"
    #   opp_ufc_win_pct      → float like 73.0
    #   opp_xr_pct           → float like 64.0
    #   *_pct with % suffix  → string like "75%" (head strikes, td defense)
    #   kd_rate/kd_abs_rate  → string with 1 decimal like "0.8%"
    def _int_str(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        try:
            return str(int(round(float(v))))
        except Exception:
            return None

    def _float_val(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        try:
            return float(v)
        except Exception:
            return None

    xr_pct = _int_str(g("xr_pct"))
    opp_ufc_win_pct = _float_val(g("opp_ufc_win_pct"))
    opp_xr_pct = _float_val(g("opp_xr_pct"))
    head_landed_fmt = _pct_str(g("head_strike_landed_pct"))
    head_absorbed_fmt = _pct_str(g("head_absorption_pct"))
    td_def_fmt = _pct_str(g("td_defense_pct"))
    kd_rate_fmt = _pct_str(g("kd_rate"), decimals=1)
    kd_abs_fmt = _pct_str(g("kd_absorbed_rate"), decimals=1)

    # --- Tapology-derived career totals ------------------------------
    def t(key, default=None):
        if not tap:
            return default
        v = tap.get(key)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        return v

    career_ko_wins = t("ko_wins")
    career_sub_wins = t("sub_wins")
    career_ko_losses = t("ko_losses")
    career_sub_losses = t("sub_losses")
    total_fights = t("total_fights")
    pro_record = t("pro_record") or g("pro_record")

    # sub_win_pct = career_sub_wins / Total Fights  (per Blank-13 mapping)
    sub_win_pct_val = None
    if career_sub_wins is not None and total_fights not in (None, 0):
        try:
            sub_win_pct_val = round(float(career_sub_wins) / float(total_fights), 4)
        except Exception:
            sub_win_pct_val = None

    # finish_rate = (career_ko_wins + career_sub_wins) / Total Fights
    finish_rate_val = t("finish_rate")
    if finish_rate_val is None and total_fights not in (None, 0):
        ko = career_ko_wins or 0
        sub = career_sub_wins or 0
        try:
            finish_rate_val = round((float(ko) + float(sub)) / float(total_fights), 4)
        except Exception:
            finish_rate_val = None

    # --- UFCStats recent fights --------------------------------------
    def r(key):
        if not recent:
            return None
        v = recent.get(key)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return v

    # --- OpticOdds row -----------------------------------------------
    def o(key):
        if not odds_row:
            return None
        v = odds_row.get(key)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return v

    open_ml = o("ml_open_dk")
    if open_ml is None:
        open_ml = open_odds
    current_ml = o("ml_current")
    if current_ml is None:
        current_ml = current_odds
    # ml_win_by_* are stored as implied probabilities in card_odds.csv.
    # APEX-ENGINE.csv holds them as American odds — convert so the column
    # is drop-in comparable with the existing Google Sheet.
    ml_ko = o("ml_win_by_ko")
    ml_sub = o("ml_win_by_sub")
    ml_dec = o("ml_win_by_dec")

    def _prob_to_american(p):
        if p is None or (isinstance(p, float) and pd.isna(p)):
            return None
        try:
            p = float(p)
            if p <= 0 or p >= 1:
                return None
            if p >= 0.5:
                return f"-{int(round(p / (1 - p) * 100))}"
            return f"+{int(round((1 - p) / p * 100))}"
        except Exception:
            return None

    row = {
        "Event": event_name,
        "fight_pair": fight_pair_idx,
        "FIGHTER": fighter_name,
        "opponent_name": opponent_name,
        "weight_class": g("weight_class"),
        "scheduled_rounds": scheduled_rounds,
        "short_notice": 0,  # default; user can override
        "dk_salary": None,  # external
        "age": age_fmt,
        "reach": reach_fmt,
        "height": height_fmt,
        "stance": g("stance"),
        "pro_record": pro_record,
        "ufc_record": ufc_record,
        "ufc_fight_count": g("ufc_fight_count"),
        # Career finish counts — from Tapology
        "career_ko_wins": career_ko_wins,
        "career_sub_wins": career_sub_wins,
        "career_ko_losses": career_ko_losses,
        "career_sub_losses": career_sub_losses,
        "slpm": g("slpm"),
        "sig_strike_diff": g("sig_strike_diff"),
        "kd_rate": kd_rate_fmt,
        "sapm": g("sapm"),
        "kd_absorbed_rate": kd_abs_fmt,
        "head_absorption_pct": head_absorbed_fmt,
        "td_per_15": g("td_per_15"),
        "ctrl_time_per_15": g("ctrl_time_per_15"),
        "sub_win_pct": sub_win_pct_val,
        "ufc_minutes_fought": g("ufc_minutes_fought"),
        "td_defense_pct": td_def_fmt,
        "ctrl_absorbed_per_15": g("ctrl_absorbed_per_15"),
        # Recent fight results — from UFCStats
        "recent_fight_1_result": r("recent_fight_1_result"),
        "recent_fight_1_method": r("recent_fight_1_method"),
        "recent_fight_2_result": r("recent_fight_2_result"),
        "recent_fight_2_method": r("recent_fight_2_method"),
        "recent_fight_3_result": r("recent_fight_3_result"),
        "recent_fight_3_method": r("recent_fight_3_method"),
        "opp_ufc_win_pct": opp_ufc_win_pct,
        "opp_xr_pct": opp_xr_pct,
        "xr_pct": xr_pct,
        "finish_rate": finish_rate_val,
        "elo": elo,
        "OPEN": _american(open_ml),
        "CURRENT": _american(current_ml),
        "ml_win_by_ko": _prob_to_american(ml_ko),
        "ml_win_by_sub": _prob_to_american(ml_sub),
        "ml_win_by_dec": _prob_to_american(ml_dec),
        "avg_dk_pts": g("avg_dk_pts"),
        "dk_win_avg": g("dk_win_avg"),
        "dk_loss_avg": g("dk_loss_avg"),
        "champ_experience": 0,  # default
        "Total Fights": total_fights,
        "head_strike_landed_pct": head_landed_fmt,
        "head_strike_absorbed_pct": head_absorbed_fmt,
        "Missed_weight": 0,
        "Long_layoff": 0,
        "Elo_pct_secondary": None,
        "GPT": None, "SIM": None, "MMA-AI": None, "CAPPERS": None,
        "GROK": None, "CLAUDE": None, "GEMINI": None,
        "CHAT GPT": None, "DEEP SEEK": None, "AI-AVERAGE": None,
        "MODEL 75": None, "NEW MODEL": None, "SCRAP": None,
        "CAGE SCORE": None, "CAGE PICKS": None,
        "ELO/IMPLIED": None, "CI LOW": None, "CI HIGH": None,
        "RESULTS": "PENDING",
        # Per Scott's column mapping, sig_strike_accuracy_pct / defense_pct come
        # from Latshaw's Significant Strike Accuracy / Defense (distance-only).
        "sig_strike_accuracy_pct": _pct_str(g("sig_strike_accuracy_pct")),
        "sig_strike_defense_pct": _pct_str(g("sig_strike_defense_pct")),
    }
    return row


# --- Odds lookup ------------------------------------------------------

def _open_current_from_snapshot(snap, fighter_a: str, fighter_b: str):
    """Best-effort look up of current moneyline in the OpticOdds snapshot.
    Returns (open_a, current_a, open_b, current_b)."""
    if not snap:
        return None, None, None, None
    odds_by_fixture = snap.get("odds_by_fixture", {})
    for entry in odds_by_fixture.values():
        f = entry["fixture"]
        names = {f["home_name"].lower(), f["away_name"].lower()}
        if fighter_a.lower() in names and fighter_b.lower() in names:
            odds = entry.get("odds", [])
            best_a, best_b = None, None
            for o in odds:
                if o.get("market") not in ("Moneyline", "moneyline"):
                    continue
                nm = (o.get("name") or "").lower()
                price = o.get("price")
                if nm == fighter_a.lower():
                    if best_a is None or (price is not None and price < best_a):
                        best_a = price
                elif nm == fighter_b.lower():
                    if best_b is None or (price is not None and price < best_b):
                        best_b = price
            return None, best_a, None, best_b
    return None, None, None, None


# --- Streamlit tab ----------------------------------------------------

def render_card_prep_page():
    st.header("Card Prep — Auto-fill APEX rows")
    st.caption(
        "Pulls fighter data from four sources: Nate Latshaw's Statistical "
        "Companion (fighter stats), Tapology (career finish totals), UFCStats "
        "(recent-fight history), and OpticOdds (open/current ML + method-of-"
        "victory prices). Yellow cells are the columns you still fill in "
        "manually (DK salary, model outputs, judgement flags)."
    )

    fm_df = _load_fm_ratings()
    snap = _load_snapshot()
    tap_df = _load_tapology()
    recent_df = _load_recent_fights()
    card_odds_df = _load_card_odds()

    # --- Event selector -----------------------------------------------
    try:
        events = L.event_names()
    except FileNotFoundError:
        st.error(
            "Latshaw data files not found in `data/latshaw/`. "
            "Run `python scripts/refresh_latshaw.py` (or extract from "
            "https://natelatshaw.shinyapps.io/ufc_fight_night_statistical_companion/)."
        )
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_event = st.selectbox(
            "Select event", options=events, index=0,
            help="Latshaw usually shows the next 1-2 upcoming UFC events."
        )
    with col2:
        st.metric("Bouts on card", len(L.bout_list(selected_event)))

    st.markdown("#### Fights on this card (from Latshaw)")
    st.caption(
        "Edit rounds (5 for main-event/title bouts, 3 otherwise). Fighter A "
        "and B come straight from Latshaw's Welcome page bout list."
    )

    # Build fights_df from Latshaw bout_list
    bouts = L.bout_list(selected_event)
    default_rows = []
    for a, b, bo in bouts:
        # Main event (bout 1) = 5 rounds by default
        rounds = 5 if bo == 1 else 3
        default_rows.append({"Fighter A": a, "Fighter B": b, "Rounds": rounds})
    fights_df = pd.DataFrame(default_rows)

    edited = st.data_editor(
        fights_df,
        use_container_width=True,
        num_rows="dynamic",
        hide_index=True,
        key="card_prep_fights",
    )

    if st.button("🏗️  Build APEX rows", type="primary"):
        rows = []
        missing_by_fighter = {}
        for i, r in edited.iterrows():
            a = str(r["Fighter A"]).strip()
            b = str(r["Fighter B"]).strip()
            rounds = int(r["Rounds"]) if pd.notna(r["Rounds"]) else 3
            if not a or not b or a.lower() == "nan":
                continue
            lat_a = L.get_row(a)
            lat_b = L.get_row(b)
            fm_a = _fm_lookup(fm_df, a)
            fm_b = _fm_lookup(fm_df, b)
            tap_a = _lookup(tap_df, a)
            tap_b = _lookup(tap_df, b)
            recent_a = _lookup(recent_df, a)
            recent_b = _lookup(recent_df, b)
            odds_a = _lookup(card_odds_df, a)
            odds_b = _lookup(card_odds_df, b)

            _, current_a, _, current_b = _open_current_from_snapshot(snap, a, b)

            row_a = build_row(selected_event, i + 1, lat_a, lat_b, fm_a,
                              a, b, rounds, open_odds=None, current_odds=current_a,
                              tap=tap_a, recent=recent_a, odds_row=odds_a)
            row_b = build_row(selected_event, i + 1, lat_b, lat_a, fm_b,
                              b, a, rounds, open_odds=None, current_odds=current_b,
                              tap=tap_b, recent=recent_b, odds_row=odds_b)
            rows.append(row_a)
            rows.append(row_b)
            missing_by_fighter[a] = lat_a is None
            missing_by_fighter[b] = lat_b is None

        if not rows:
            st.warning("No fights entered.")
            return

        df = pd.DataFrame(rows, columns=APEX_COLUMNS)

        # Coverage stats
        total_cells = df.size
        filled = df.notna().sum().sum()
        by_col_missing = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total rows", len(df))
        c2.metric("Filled cells", f"{filled:,} / {total_cells:,}",
                  f"{filled/total_cells*100:.1f}%")
        not_found = [n for n, missing in missing_by_fighter.items() if missing]
        c3.metric("Not in Latshaw", len(not_found),
                  help=", ".join(not_found) if not_found else "All matched")

        # Yellow-highlight ONLY the columns explicitly marked "Manually Loaded"
        # in docs/apex_column_mapping.csv — not every missing value.
        def _style_manual_col(col):
            if col.name in MANUAL_ONLY_COLUMNS:
                return ["background-color: #fff9b0; color: #333;"] * len(col)
            return [""] * len(col)

        styled = df.style.apply(_style_manual_col, axis=0)
        st.markdown("### 📋 Preview (yellow = needs manual fill)")
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Column coverage summary
        st.markdown("#### Columns needing manual entry")
        all_missing = by_col_missing[by_col_missing > 90]
        partial_missing = by_col_missing[(by_col_missing > 0) & (by_col_missing <= 90)]
        c_left, c_right = st.columns(2)
        with c_left:
            st.caption("**Always manual** (no Latshaw source)")
            st.write(list(all_missing.index))
        with c_right:
            st.caption("**Sometimes missing** (fighter-specific gaps)")
            st.dataframe(
                partial_missing.round(1).reset_index().rename(
                    columns={"index": "Column", 0: "% missing"}
                ),
                use_container_width=True, hide_index=True,
            )

        # --- Export -----------------------------------------------------
        st.markdown("### 📤 Export")
        col_tsv, col_csv = st.columns(2)

        tsv_buf = io.StringIO()
        df.to_csv(tsv_buf, sep="\t", index=False, na_rep="")
        tsv_str = tsv_buf.getvalue()

        with col_tsv:
            st.download_button(
                "📥 Download TSV (Google Sheets paste)",
                data=tsv_str,
                file_name=f"apex_card_prep_{datetime.now().strftime('%Y%m%d_%H%M')}.tsv",
                mime="text/tab-separated-values",
                use_container_width=True,
            )

        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False, na_rep="")
        with col_csv:
            st.download_button(
                "📥 Download CSV",
                data=csv_buf.getvalue(),
                file_name=f"apex_card_prep_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with st.expander("📋 Copy TSV directly (select all, ⌘+C, paste to Sheet)", expanded=False):
            st.text_area(
                "Select all and copy",
                value=tsv_str,
                height=300,
                help="Google Sheets accepts tab-separated pastes into a single starting cell.",
                key="tsv_copy_area",
            )

        st.info(
            "🔎 Pro tip: paste into a fresh sheet starting at A1. Google Sheets "
            "auto-splits into columns since values are tab-separated."
        )
