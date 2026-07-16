"""
Live Odds & Edge tab for CagePicks Apex V2.

Pulls current UFC odds via OpticOdds (or a committed snapshot for the
public Streamlit deploy) and cross-references them against the V2 model
to surface no-vig fair lines, best available prices, and model edges.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from opticodds_client import (
    DEFAULT_SPORTSBOOKS,
    SNAPSHOT_PATH,
    fetch_active_ufc_fixtures,
    fetch_fixture_odds,
    snapshot_age_hours,
    _connector_available,
    _load_snapshot,
)
from edge_math import (
    american_to_implied,
    build_moneyline_snapshot,
    compute_edge,
    format_american,
    no_vig_two_way,
)
from fightmatrix_math import build_fm_comparison, three_way_consensus


def _fmt_prob(p: float) -> str:
    return f"{p * 100:.1f}%"


def _fmt_ts(ts: float) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _fmt_event_date(iso: str) -> str:
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%a %b %d, %Y")
    except Exception:
        return iso


def _lookup_fighter_row(lookup: pd.DataFrame, name: str) -> pd.Series | None:
    """Best-effort lookup by fighter_name."""
    if lookup is None or lookup.empty:
        return None
    name_norm = name.strip().lower()
    matches = lookup[lookup["fighter_name"].str.lower() == name_norm]
    if matches.empty:
        # Try last-name fallback
        last = name_norm.split()[-1] if name_norm else ""
        matches = lookup[lookup["fighter_name"].str.lower().str.endswith(" " + last)]
    if matches.empty:
        return None
    return matches.iloc[0]


@st.cache_data(show_spinner=False)
def _load_fm_lookup() -> pd.DataFrame | None:
    """Load fighter_lookup_apex_v2_with_fm.csv if present."""
    import os
    path = os.path.join(os.path.dirname(__file__), "fighter_lookup_apex_v2_with_fm.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, low_memory=False)


def _lookup_fm_row(fm_lookup: pd.DataFrame, name: str) -> pd.Series | None:
    if fm_lookup is None or fm_lookup.empty:
        return None
    # Take the most recent entry for that fighter (has FM columns)
    name_norm = name.strip().lower()
    matches = fm_lookup[fm_lookup["fighter_name"].str.lower() == name_norm]
    if matches.empty:
        last = name_norm.split()[-1] if name_norm else ""
        matches = fm_lookup[fm_lookup["fighter_name"].str.lower().str.endswith(" " + last)]
    if matches.empty:
        return None
    # Prefer rows that have FM data populated
    matches = matches.sort_values("fm_blended", na_position="last")
    return matches.iloc[0]


def _render_market_row(book: str, price: int, is_best: bool = False) -> str:
    css = "background: #10331c; color: #7ce29c;" if is_best else ""
    return f"<td style='padding:6px 12px; {css}'>{format_american(price)}</td>"


def render_live_odds_page():
    st.header("Live Odds & Edge (V2)")
    st.caption(
        "Real market prices vs. V2 model probabilities. Green rows = PLAY signals (edge ≥ 4%), "
        "amber = LEAN (edge ≥ 2%). Odds via OpticOdds."
    )

    # Load snapshot (metadata + fixture list)
    snap = _load_snapshot()
    if snap is None:
        st.warning(
            "No odds snapshot found yet. If you are running locally with the OpticOdds bridge, "
            "click **Refresh from OpticOdds** below."
        )

    connector_live = _connector_available()

    # Header row with refresh button and freshness
    top_l, top_m, top_r = st.columns([2, 2, 1])
    with top_l:
        age = snapshot_age_hours()
        if age is not None:
            fetched = datetime.fromtimestamp(snap.get("fetched_at", 0), tz=timezone.utc)
            st.metric("Snapshot age", f"{age:.1f}h", help=f"Fetched {fetched:%Y-%m-%d %H:%M UTC}")
        else:
            st.metric("Snapshot age", "—")
    with top_m:
        book_count = len(snap.get("sportsbooks", DEFAULT_SPORTSBOOKS)) if snap else len(DEFAULT_SPORTSBOOKS)
        st.metric("Sportsbooks tracked", book_count)
    with top_r:
        if connector_live:
            if st.button("🔄 Refresh from OpticOdds"):
                with st.spinner("Refreshing odds…"):
                    from opticodds_client import refresh_snapshot
                    result = refresh_snapshot()
                    st.success(f"Refreshed {result['with_odds']}/{result['fixtures']} fixtures")
                    st.rerun()
        else:
            st.caption("Bridge unavailable — snapshot mode")

    # Fixture selector: only ones with odds
    if not snap:
        return
    odds_by_fixture = snap.get("odds_by_fixture", {})
    fixtures_with_odds = [
        entry["fixture"] for entry in odds_by_fixture.values()
        if entry.get("odds")
    ]
    # Filter to upcoming (start date >= now - 6h so still-live shows briefly)
    now = datetime.now(timezone.utc)
    upcoming = []
    for f in fixtures_with_odds:
        try:
            dt = datetime.fromisoformat(f["start_date"].replace("Z", "+00:00"))
            if (dt - now).total_seconds() > -6 * 3600:
                upcoming.append((f, dt))
        except Exception:
            upcoming.append((f, None))
    # Sort by start date
    upcoming.sort(key=lambda x: x[1] or datetime.max.replace(tzinfo=timezone.utc))

    if not upcoming:
        st.info("No upcoming fixtures with live odds in the snapshot. Refresh to pull the next slate.")
        return

    st.divider()

    # Fixture picker
    labels = []
    for f, dt in upcoming:
        date_str = _fmt_event_date(f["start_date"])
        labels.append(f"{f['home_name']} vs {f['away_name']}  ·  {date_str}  ·  {f.get('event_name', 'UFC')}")

    idx = st.selectbox(
        "Select fight",
        options=range(len(upcoming)),
        format_func=lambda i: labels[i],
    )
    fixture_dict, fixture_dt = upcoming[idx]
    fixture_id = fixture_dict["fixture_id"]
    fighter_a = fixture_dict["home_name"]
    fighter_b = fixture_dict["away_name"]

    # Get odds for the selected fixture
    entry = odds_by_fixture[fixture_id]
    odds_rows_raw = entry.get("odds", [])

    # Convert to OddsRow-like objects (duck typed)
    class _Row:
        __slots__ = ("sportsbook", "market", "market_id", "name", "selection",
                     "price", "points", "grouping_key", "deep_link", "timestamp")

        def __init__(self, d):
            for k in self.__slots__:
                setattr(self, k, d.get(k))

    odds_rows = [_Row(o) for o in odds_rows_raw]

    st.subheader(f"{fighter_a}  vs  {fighter_b}")
    st.caption(
        f"{fixture_dict.get('event_name', '')} · {fixture_dict.get('venue', '')} · "
        f"Odds updated {_fmt_ts(entry.get('fetched_at', 0))}"
    )

    # --- Moneyline market -------------------------------------------------
    ml_snap = build_moneyline_snapshot(odds_rows, fighter_a, fighter_b)
    if ml_snap is None:
        st.info("No moneyline odds available for this fight yet.")
    else:
        st.markdown("### Moneyline")

        # Build side-by-side table
        books_a = {q.sportsbook: q for q in ml_snap.quotes_a}
        books_b = {q.sportsbook: q for q in ml_snap.quotes_b}
        all_books = [b for b in DEFAULT_SPORTSBOOKS if b in books_a or b in books_b]

        rows = []
        for book in all_books:
            qa = books_a.get(book)
            qb = books_b.get(book)
            rows.append({
                "Sportsbook": book,
                f"{fighter_a}": format_american(qa.price) if qa else "—",
                f"{fighter_b}": format_american(qb.price) if qb else "—",
            })
        df_ml = pd.DataFrame(rows)
        st.dataframe(df_ml, hide_index=True, use_container_width=True)

        # No-vig + best line summary
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{fighter_a} best", format_american(ml_snap.best_a.price),
                  help=f"at {ml_snap.best_a.sportsbook}")
        c2.metric(f"{fighter_b} best", format_american(ml_snap.best_b.price),
                  help=f"at {ml_snap.best_b.sportsbook}")
        c3.metric(
            "No-vig fair",
            f"{_fmt_prob(ml_snap.consensus_a)} / {_fmt_prob(ml_snap.consensus_b)}",
            help="Vig stripped from consensus implied probabilities across all books.",
        )

        # --- Model comparison + edge --------------------------------------
        st.markdown("### V2 Model vs. Market")

        # Try to load model + generate a prediction
        try:
            from apex_predictor_v2 import load_apex_model_v2, predict_pair
            models, features, lookup, metrics = load_apex_model_v2()
            row_a = _lookup_fighter_row(lookup, fighter_a)
            row_b = _lookup_fighter_row(lookup, fighter_b)
        except Exception as e:
            st.error(f"Could not load V2 model: {e}")
            row_a = row_b = None
            models = features = lookup = None

        if row_a is None or row_b is None:
            missing = []
            if row_a is None:
                missing.append(fighter_a)
            if row_b is None:
                missing.append(fighter_b)
            st.warning(
                "V2 fighter data unavailable for: " + ", ".join(missing) +
                ". Model edge not computed — showing market only."
            )
        else:
            try:
                pred = predict_pair(models, features, row_a, row_b)
                model_p_a = float(pred["prob_a"])
                model_p_b = float(pred["prob_b"])

                edge_a = compute_edge(fighter_a, model_p_a, ml_snap.consensus_a, ml_snap.best_a)
                edge_b = compute_edge(fighter_b, model_p_b, ml_snap.consensus_b, ml_snap.best_b)

                # Choose the side with a larger positive edge as the pick if any
                pick = None
                if edge_a.edge_pct > edge_b.edge_pct and edge_a.edge_pct > 0:
                    pick = edge_a
                elif edge_b.edge_pct > 0:
                    pick = edge_b

                # Verdict banners
                for report in (edge_a, edge_b):
                    color, icon = {
                        "PLAY": ("#10331c", "✅"),
                        "LEAN": ("#3b2f10", "🟡"),
                        "PASS": ("#2b2b2b", "⚪"),
                    }[report.verdict]
                    st.markdown(
                        f"<div style='background:{color}; padding:14px 18px; border-radius:8px; margin-bottom:8px;'>"
                        f"<span style='font-size:1.15em; font-weight:600;'>{icon} {report.fighter}  ·  {report.verdict}</span>"
                        f"<br/><span style='color:#bbb;'>Model {_fmt_prob(report.model_prob)}  ·  "
                        f"Market {_fmt_prob(report.market_prob)}  ·  "
                        f"Edge <b>{report.edge_pct:+.1f}%</b>  ·  "
                        f"Best {format_american(report.best_price)} at {report.best_book}  ·  "
                        f"EV ${report.ev_per_dollar:+.3f}/$1  ·  "
                        f"¼-Kelly {report.kelly_quarter * 100:.1f}%</span></div>",
                        unsafe_allow_html=True,
                    )

                # Model archetype context
                st.caption(
                    f"V2 archetypes: {fighter_a} → **{pred.get('a_archetype', '?')}**  ·  "
                    f"{fighter_b} → **{pred.get('b_archetype', '?')}**"
                )

                # --- Fight Matrix consensus panel -------------------------
                fm_lookup = _load_fm_lookup()
                if fm_lookup is not None:
                    fm_row_a = _lookup_fm_row(fm_lookup, fighter_a)
                    fm_row_b = _lookup_fm_row(fm_lookup, fighter_b)
                    comp = build_fm_comparison(fighter_a, fighter_b, fm_row_a, fm_row_b)
                    if comp is not None:
                        with st.expander("Fight Matrix Consensus (2nd opinion)", expanded=True):
                            _render_fm_panel(comp, model_p_a, ml_snap.consensus_a, fighter_a, fighter_b)

                # Deep links
                if pick and ml_snap:
                    best = ml_snap.best_a if pick.fighter == fighter_a else ml_snap.best_b
                    if best and best.deep_link:
                        st.markdown(
                            f"[Open bet slip at {best.sportsbook}]({best.deep_link})"
                        )

            except Exception as e:
                st.error(f"Edge computation failed: {e}")
                st.exception(e)

    # --- Method of Victory + Total Rounds (informational) -----------------
    _continue_odds_page(odds_rows, fighter_a, fighter_b)


def _render_fm_panel(comp, model_prob_a: float, market_prob_a: float,
                     fighter_a: str, fighter_b: str) -> None:
    """Render the Fight Matrix consensus expander block."""
    # Header rating table
    st.caption(
        f"Ratings from [fightmatrix.com]({comp.fm_profile_a or 'https://www.fightmatrix.com/'}) — "
        f"user blend: **0.45·Glicko-1 + 0.55·WHR**"
    )
    rows = []
    for label, key in [("Glicko-1", "glicko1"), ("WHR", "whr"), ("Elo K170", "k170")]:
        ra = comp.ratings_a.get(key)
        rb = comp.ratings_b.get(key)
        if ra is None or rb is None:
            continue
        pa, pb = comp.system_probs.get(key, (None, None))
        rows.append({
            "System": label,
            f"{fighter_a}": f"{ra:.0f}",
            f"{fighter_b}": f"{rb:.0f}",
            "Diff": f"{ra - rb:+.0f}",
            f"P({fighter_a})": f"{pa*100:.1f}%" if pa is not None else "—",
        })
    if comp.blended_rating_a is not None:
        rows.append({
            "System": "Blended (0.45G+0.55W)",
            f"{fighter_a}": f"{comp.blended_rating_a:.1f}",
            f"{fighter_b}": f"{comp.blended_rating_b:.1f}",
            "Diff": f"{comp.blended_rating_a - comp.blended_rating_b:+.1f}",
            f"P({fighter_a})": f"{comp.blended_prob_a*100:.1f}%",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)

    # Three-way comparison metric row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Market (no-vig)", f"{market_prob_a*100:.1f}% / {(1-market_prob_a)*100:.1f}%",
              help=f"Market implied for {fighter_a} / {fighter_b}")
    m2.metric("V2 Model", f"{model_prob_a*100:.1f}% / {(1-model_prob_a)*100:.1f}%",
              help="V2 ensemble prediction")
    if comp.blended_prob_a == comp.blended_prob_a:  # not NaN
        m3.metric("FM Blend", f"{comp.blended_prob_a*100:.1f}% / {(1-comp.blended_prob_a)*100:.1f}%",
                  help="0.45·Glicko-1 + 0.55·WHR blend")
    m4.metric("Systems verdict", comp.consensus_verdict,
              help=f"Largest disagreement across FM systems: {comp.largest_disagreement*100:.1f}%")

    # Three-way consensus banner
    if comp.blended_prob_a == comp.blended_prob_a:
        label = three_way_consensus(model_prob_a, comp.blended_prob_a, market_prob_a)
        banner_cfg = {
            "STRONG_CONSENSUS": ("#0e4d29", "🎯", f"STRONG CONSENSUS — V2 + FM Blend agree against the market"),
            "WEAK_CONSENSUS": ("#3b2f10", "🟡", f"WEAK CONSENSUS — V2 + FM Blend agree against the market, small edge"),
            "SPLIT": ("#3b1010", "⚠️", "SPLIT — V2 Model and FM Blend disagree on the pick"),
            "MARKET_ALIGNED": ("#1e2a3a", "➖", "Market-aligned — model and FM blend agree with the market"),
            "NEUTRAL": ("#2b2b2b", "⚪", "Neutral"),
            "INCOMPLETE": ("#2b2b2b", "⚪", "Insufficient data"),
        }[label]
        color, icon, msg = banner_cfg
        st.markdown(
            f"<div style='background:{color}; padding:12px 16px; border-radius:6px; margin-top:8px;'>"
            f"<span style='font-size:1.05em; font-weight:600;'>{icon} {msg}</span></div>",
            unsafe_allow_html=True,
        )


def _continue_odds_page(odds_rows, fighter_a: str, fighter_b: str):
    """Method of Victory + Total Rounds informational panels."""
    method_rows = [r for r in odds_rows if r.market == "Method of Victory"]
    if method_rows:
        st.markdown("### Method of Victory")
        rows = []
        for r in method_rows:
            rows.append({
                "Sportsbook": r.sportsbook,
                "Outcome": r.name,
                "Price": format_american(r.price),
                "Implied": _fmt_prob(american_to_implied(r.price)),
            })
        df_m = pd.DataFrame(rows).sort_values(["Outcome", "Sportsbook"])
        st.dataframe(df_m, hide_index=True, use_container_width=True)

    # --- Total Rounds -----------------------------------------------------
    total_rows = [r for r in odds_rows if r.market == "Total Rounds"]
    if total_rows:
        st.markdown("### Total Rounds")
        rows = []
        for r in total_rows:
            rows.append({
                "Sportsbook": r.sportsbook,
                "Line": r.points,
                "Side": r.selection or ("Over" if "over" in (r.grouping_key or "").lower() else "Under"),
                "Name": r.name,
                "Price": format_american(r.price),
                "Implied": _fmt_prob(american_to_implied(r.price)),
            })
        df_t = pd.DataFrame(rows).sort_values(["Line", "Sportsbook"])
        st.dataframe(df_t, hide_index=True, use_container_width=True)

    st.divider()
    st.caption(
        "Disclaimer: model probabilities are point estimates from an ensemble. "
        "Edges below 2% are within model noise. Bet responsibly."
    )
