"""
Card Prep tab: pick the upcoming UFC card, auto-fill every APEX column
we can from the V2 roster + FM ratings, and highlight missing cells in
yellow. Output is copy/paste-ready for Google Sheets (TSV) or CSV
download.

APEX target schema (78 columns): see /uploaded_attachments/APEX-UFC-329.
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from fightmatrix_math import elo_win_prob
from opticodds_client import DEFAULT_SPORTSBOOKS, _load_snapshot


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


# --- Data loaders -----------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_v2_lookup() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__), "fighter_lookup_apex_v2_with_fm.csv")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), "fighter_lookup_apex_v2.csv")
    return pd.read_csv(path, low_memory=False)


def _lookup_fighter(df: pd.DataFrame, name: str):
    """Best-effort match by fighter_name."""
    if not name:
        return None
    n = name.strip().lower()
    m = df[df["fighter_name"].str.lower() == n]
    if m.empty:
        # last-name fallback
        last = n.split()[-1] if n else ""
        m = df[df["fighter_name"].str.lower().str.endswith(" " + last)]
    if m.empty:
        return None
    # Prefer the row with FM data populated when duplicates exist
    if "fm_blended" in m.columns:
        m = m.sort_values("fm_blended", na_position="last")
    return m.iloc[0]


# --- Row builder ------------------------------------------------------

def _fmt(v, decimals: int | None = None, pct: bool = False):
    """Format a numeric value for the APEX schema."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if pct:
        if abs(f) <= 1.0:  # fraction -> %
            return round(f * 100, decimals) if decimals is not None else round(f * 100, 2)
        return round(f, decimals) if decimals is not None else f
    if decimals is not None:
        return round(f, decimals)
    return f


def _american(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        i = int(v)
        return f"+{i}" if i > 0 else str(i)
    except Exception:
        return v


def build_row(
    event_name: str,
    fight_pair_idx: int,
    fighter_row,
    opponent_row,
    fighter_name: str,
    opponent_name: str,
    scheduled_rounds: int,
    open_odds: int | None = None,
    current_odds: int | None = None,
    market_implied: float | None = None,
) -> dict:
    """Assemble one APEX row for a single fighter.

    Cells that can't be resolved return None so the UI can highlight them.
    """
    def g(key, default=None):
        if fighter_row is None:
            return default
        v = fighter_row.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return v

    def opp(key, default=None):
        if opponent_row is None:
            return default
        v = opponent_row.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return v

    # Compute FM blended ELO for the elo column: 0.45*Glicko-1 + 0.55*WHR
    fm_g = g("fm_glicko1")
    fm_w = g("fm_whr")
    blended_elo = None
    if fm_g is not None and fm_w is not None:
        try:
            blended_elo = round(0.45 * float(fm_g) + 0.55 * float(fm_w), 0)
        except (TypeError, ValueError):
            blended_elo = g("elo")  # fallback to V2 elo
    else:
        blended_elo = g("elo")

    # Height / reach formatting to match APEX style
    height_in = g("height_in")
    if height_in:
        try:
            hi = int(height_in)
            ft = hi // 12
            inch = hi % 12
            height_fmt = f"{ft}ft {inch}in"
        except Exception:
            height_fmt = None
    else:
        height_fmt = None
    reach_in = g("reach_in")
    reach_fmt = f"{int(reach_in)}in" if reach_in else None

    # Age
    age = g("age")
    age_fmt = f"{age} years" if age is not None else None

    # Pro/UFC record — approximated from ufc_wins/losses/draws
    ufc_w = g("ufc_wins")
    ufc_l = g("ufc_losses")
    ufc_d = g("ufc_draws", 0) or 0
    ufc_record = f"{int(ufc_w)}-{int(ufc_l)}-{int(ufc_d)}" if ufc_w is not None and ufc_l is not None else None

    # Head strike absorbed pct — approximated as 1 - head_absorption defense
    # Actually head_absorption_pct in V2 IS the share of head strikes absorbed.
    # For "head_strike_absorbed_pct" (APEX naming), we mirror head_absorption_pct.
    head_strike_absorbed = _fmt(g("head_absorption_pct"), decimals=0, pct=True)
    head_strike_absorbed_fmt = f"{int(head_strike_absorbed)}%" if head_strike_absorbed is not None else None

    # Long layoff / short notice booleans
    long_layoff = int(bool(g("is_long_layoff", 0))) if g("is_long_layoff") is not None else 0
    short_notice = int(bool(g("is_short_notice", 0))) if g("is_short_notice") is not None else 0

    # sig_strike_accuracy/defense to % strings ("48%")
    sig_acc = g("sig_strike_accuracy")
    sig_def = g("sig_strike_defense")
    sig_acc_fmt = f"{int(round(float(sig_acc) * 100))}%" if sig_acc is not None else None
    sig_def_fmt = f"{int(round(float(sig_def) * 100))}%" if sig_def is not None else None

    # head_strike_landed_pct → format as %
    head_landed = g("head_strike_landed_pct")
    head_landed_fmt = f"{int(round(float(head_landed) * 100))}%" if head_landed is not None else None

    # td_defense_pct → format as %
    td_def = g("td_defense_pct")
    td_def_fmt = f"{int(round(float(td_def) * 100))}%" if td_def is not None else None

    # opp_* percentages → integer % values
    def _int_pct(v):
        if v is None:
            return None
        try:
            return int(round(float(v) * 100)) if abs(float(v)) <= 1 else int(round(float(v)))
        except Exception:
            return None

    # xr_pct / opp_xr_pct / opp_ufc_win_pct scaling
    opp_ufc_win_pct = _int_pct(g("opp_ufc_win_pct")) if g("opp_ufc_win_pct") is not None else None
    opp_xr = _int_pct(g("opp_xr_pct")) if g("opp_xr_pct") is not None else None
    xr = _int_pct(g("xr_pct")) if g("xr_pct") is not None else None

    total_fights = None
    if ufc_w is not None and ufc_l is not None:
        total_fights = int(ufc_w) + int(ufc_l) + int(ufc_d or 0)

    row = {
        "Event": event_name,
        "fight_pair": fight_pair_idx,
        "FIGHTER": fighter_name,
        "opponent_name": opponent_name,
        "weight_class": g("weight_class"),
        "scheduled_rounds": scheduled_rounds,
        "short_notice": short_notice,
        "dk_salary": None,  # external
        "age": age_fmt,
        "reach": reach_fmt,
        "height": height_fmt,
        "stance": g("stance"),
        "pro_record": None,  # external
        "ufc_record": ufc_record,
        "ufc_fight_count": g("ufc_fight_count"),
        "career_ko_wins": g("career_ko_wins"),
        "career_sub_wins": g("career_sub_wins"),
        "career_ko_losses": g("career_ko_losses"),
        "career_sub_losses": g("career_sub_losses"),
        "slpm": _fmt(g("slpm"), 2),
        "sig_strike_diff": _fmt(g("sig_strike_diff"), 2),
        "kd_rate": f"{_fmt(g('kd_rate'), 1, pct=True)}%" if g("kd_rate") is not None else None,
        "sapm": _fmt(g("sapm"), 2),
        "kd_absorbed_rate": f"{_fmt(g('kd_absorbed_rate'), 1, pct=True)}%" if g("kd_absorbed_rate") is not None else None,
        "head_absorption_pct": f"{int(round(float(g('head_absorption_pct'))*100))}%" if g("head_absorption_pct") is not None else None,
        "td_per_15": _fmt(g("td_per_15"), 2),
        "ctrl_time_per_15": _fmt(g("ctrl_time_per_15"), 2),
        "sub_win_pct": _fmt(g("sub_win_pct"), 2),
        "ufc_minutes_fought": _fmt(g("ufc_minutes_fought"), 2),
        "td_defense_pct": td_def_fmt,
        "ctrl_absorbed_per_15": _fmt(g("ctrl_absorbed_per_15"), 2),
        "recent_fight_1_result": g("recent_fight_1_result"),
        "recent_fight_1_method": g("recent_fight_1_method"),
        "recent_fight_2_result": g("recent_fight_2_result"),
        "recent_fight_2_method": g("recent_fight_2_method"),
        "recent_fight_3_result": g("recent_fight_3_result"),
        "recent_fight_3_method": g("recent_fight_3_method"),
        "opp_ufc_win_pct": opp_ufc_win_pct,
        "opp_xr_pct": opp_xr,
        "xr_pct": xr,
        "finish_rate": _fmt(g("finish_rate"), 2),
        "elo": int(blended_elo) if blended_elo is not None else None,
        "OPEN": _american(open_odds),
        "CURRENT": _american(current_odds),
        "ml_win_by_ko": None,
        "ml_win_by_sub": None,
        "ml_win_by_dec": None,
        "avg_dk_pts": None,
        "dk_win_avg": None,
        "dk_loss_avg": None,
        "champ_experience": g("champ_experience", 0),
        "Total Fights": total_fights,
        "head_strike_landed_pct": head_landed_fmt,
        "head_strike_absorbed_pct": head_strike_absorbed_fmt,
        "Missed_weight": 0,  # default; user can override
        "Long_layoff": long_layoff,
        "Elo_pct_secondary": None,  # requires v2 model runtime prob
        "GPT": None, "SIM": None, "MMA-AI": None, "CAPPERS": None,
        "GROK": None, "CLAUDE": None, "GEMINI": None,
        "CHAT GPT": None, "DEEP SEEK": None, "AI-AVERAGE": None,
        "MODEL 75": None, "NEW MODEL": None, "SCRAP": None,
        "CAGE SCORE": None, "CAGE PICKS": None,
        "ELO/IMPLIED": None, "CI LOW": None, "CI HIGH": None,
        "RESULTS": "PENDING",
        "sig_strike_accuracy_pct": sig_acc_fmt,
        "sig_strike_defense_pct": sig_def_fmt,
    }
    return row


# --- Odds lookup ------------------------------------------------------

def _open_current_from_snapshot(snap, fighter_a: str, fighter_b: str):
    """Best-effort look up of the current DraftKings / consensus moneyline
    in the OpticOdds snapshot. Returns (open_a, current_a, open_b, current_b).
    """
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
        "Pick the upcoming UFC card and paste the resulting table into your Google Sheet. "
        "Cells that couldn't be filled from the V2 roster are highlighted in yellow so you "
        "know exactly what still needs manual data collection."
    )

    v2 = _load_v2_lookup()
    snap = _load_snapshot()

    # --- Event selector -----------------------------------------------
    default_event = "UFC Fight Night: Du Plessis vs Usman"
    default_card = [
        ("Dricus Du Plessis", "Kamaru Usman", 5),
        ("Jared Cannonier", "Christian Leroy Duncan", 3),
        ("Chase Hooper", "Mitch Ramirez", 3),
        ("Tabatha Ricci", "Fatima Kline", 3),
        ("Tommy McMillen", "Alberto Montes", 3),
        ("Jose Delgado", "Austin Bashi", 3),
        ("Seok Hyeon Ko", "Jean-Paul Lebosnoyani", 3),
        ("Felipe Franco", "Levi Rodrigues Jr.", 3),
        ("Alden Coria", "Stewart Nicoll", 3),
        ("Alvin Hines", "RJ Harris", 3),
        ("Dione Barbosa", "Anna Melisano", 3),
        ("Damien Anderson", "Ezra Elliott", 3),
    ]

    col1, col2 = st.columns([2, 1])
    with col1:
        event_name = st.text_input("Event name", value=default_event)
    with col2:
        st.metric("Fighters in roster", f"{len(v2):,}")

    st.markdown("#### Fights on this card")
    st.caption("Edit the fighter names or scheduled rounds below. Use 5 for main event / title bouts, 3 otherwise.")

    # Editable dataframe of fights
    fights_df = pd.DataFrame(default_card, columns=["Fighter A", "Fighter B", "Rounds"])
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
            ra = _lookup_fighter(v2, a)
            rb = _lookup_fighter(v2, b)

            open_a = current_a = open_b = current_b = None
            _, current_a, _, current_b = _open_current_from_snapshot(snap, a, b)

            row_a = build_row(event_name, i + 1, ra, rb, a, b, rounds,
                              open_odds=None, current_odds=current_a)
            row_b = build_row(event_name, i + 1, rb, ra, b, a, rounds,
                              open_odds=None, current_odds=current_b)
            rows.append(row_a)
            rows.append(row_b)
            # Track missing
            missing_by_fighter[a] = ra is None
            missing_by_fighter[b] = rb is None

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
        c3.metric("Fighters not in roster", len(not_found),
                  help=", ".join(not_found) if not_found else "All found")

        # Highlight missing cells in yellow
        def _style_missing(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "background-color: #fff9b0; color: #333;"
            return ""
        styled = df.style.map(_style_missing)
        st.markdown("### 📋 Preview (yellow = needs manual fill)")
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Which columns are all/mostly missing — hint to user what data pipeline gap looks like
        st.markdown("#### Columns needing manual entry")
        all_missing = by_col_missing[by_col_missing > 90]
        partial_missing = by_col_missing[(by_col_missing > 0) & (by_col_missing <= 90)]
        c_left, c_right = st.columns(2)
        with c_left:
            st.caption("**Always manual** (no V2 source)")
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

        # TSV for Google Sheets paste
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

        # CSV
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

        # Copy-to-clipboard textarea (Sheets-compatible TSV)
        with st.expander("📋 Copy TSV directly (select all, ⌘+C, then paste into a Google Sheet)", expanded=False):
            st.text_area(
                "Select all and copy",
                value=tsv_str,
                height=300,
                help="Google Sheets accepts tab-separated pastes into a single starting cell.",
                key="tsv_copy_area",
            )

        st.info(
            "🔎 Pro tip: paste into a fresh sheet starting at cell A1. Google Sheets will "
            "auto-split into columns since values are tab-separated."
        )
