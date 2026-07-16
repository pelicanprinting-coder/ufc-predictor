"""
Tapology career-totals client.

Parses Tapology fighter pages for the career-finish breakdown that Latshaw
doesn't expose (career KO wins/losses, sub wins/losses, and derived
finish_rate for APEX-ENGINE).

Tapology is Cloudflare-gated so we can't hit it from plain Python.  The
pipeline is:

  1. `wide_browse` pulls the rendered page HTML (or the parsed stats)
     for each fighter, using the search endpoint to resolve the slug.
  2. Output is saved to data/tapology/career_totals.csv
  3. `get_career_totals(name)` reads the cached CSV for the tab.

For fighters not yet cached, `card_prep_tab` falls back to yellow.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Optional

import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "tapology")
_CACHE_FILE = os.path.join(_DATA_DIR, "career_totals.csv")


# --- HTML parser (used by the scraper subagent AND for offline testing) ---

_METHOD_ORDER = ("KO/TKO", "Submission", "Decision")


def parse_career_totals(html: str) -> Optional[dict]:
    """
    Parse a fighter's Tapology page HTML for PRO MMA STATISTICS.

    Returns a dict with:
      - pro_record: str like "23-3-0"
      - ko_wins, sub_wins, dec_wins: int
      - ko_losses, sub_losses, dec_losses: int
      - total_wins, total_losses: int
      - finish_rate: float 0..1 = (ko_wins+sub_wins) / total_fights
    Returns None if the stats section isn't found.
    """
    if not html or "KO/TKO" not in html:
        return None

    stats: dict = {}

    # Pro record (e.g. "PRO MMA RECORD 23-3-0" or "Record: 23-3-0")
    m = re.search(r"(?:PRO\s*MMA\s*RECORD|Record:)\s*(\d+)-(\d+)-(\d+)",
                  html, flags=re.IGNORECASE)
    if m:
        stats["pro_record"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        stats["total_wins"] = int(m.group(1))
        stats["total_losses"] = int(m.group(2))
        stats["total_draws"] = int(m.group(3))

    # For each method label, find the "N wins" and "N loss(es)" that follow.
    # Tapology renders each method as a chart block; the label is followed
    # (within ~1500 chars) by the win/loss counts.
    for label in _METHOD_ORDER:
        pat = re.compile(
            rf"<div class='primary [^']*'>{re.escape(label)}</div>(.*?)"
            rf"(?=<div class='primary [^']*'>(?:KO/TKO|Submission|Decision)"
            rf"</div>|</section>|<section)",
            flags=re.DOTALL,
        )
        m = pat.search(html)
        if not m:
            continue
        block = m.group(1)
        wm = re.search(r"(\d+)\s+wins?", block)
        lm = re.search(r"(\d+)\s+loss", block)
        wins = int(wm.group(1)) if wm else 0
        losses = int(lm.group(1)) if lm else 0
        key = label.lower().replace("/tko", "").replace("submission", "sub").replace("decision", "dec")
        stats[f"{key}_wins"] = wins
        stats[f"{key}_losses"] = losses

    # Derive finish rate = (KO wins + sub wins) / total fights.
    tw = stats.get("total_wins")
    tl = stats.get("total_losses")
    td = stats.get("total_draws", 0)
    total_fights = (tw or 0) + (tl or 0) + (td or 0)
    ko = stats.get("ko_wins", 0)
    sub = stats.get("sub_wins", 0)
    if total_fights > 0:
        stats["finish_rate"] = round((ko + sub) / total_fights, 3)
        stats["total_fights"] = total_fights
    else:
        stats["finish_rate"] = None

    return stats


# --- Cache lookup ---------------------------------------------------

@lru_cache(maxsize=1)
def _load_cache() -> pd.DataFrame:
    if not os.path.exists(_CACHE_FILE):
        return pd.DataFrame()
    df = pd.read_csv(_CACHE_FILE)
    df["_name_key"] = df["name"].astype(str).str.lower().str.strip()
    return df


def get_career_totals(name: str) -> Optional[dict]:
    """Look up cached Tapology career totals for a fighter name."""
    df = _load_cache()
    if df.empty:
        return None
    key = name.lower().strip()
    m = df[df["_name_key"] == key]
    if m.empty:
        # Last-name fallback
        last = key.split()[-1]
        m = df[df["_name_key"].str.endswith(" " + last)]
        if m.empty:
            return None
    r = m.iloc[0]
    return {
        "pro_record": r.get("pro_record"),
        "ko_wins": _int_or_none(r.get("ko_wins")),
        "sub_wins": _int_or_none(r.get("sub_wins")),
        "dec_wins": _int_or_none(r.get("dec_wins")),
        "ko_losses": _int_or_none(r.get("ko_losses")),
        "sub_losses": _int_or_none(r.get("sub_losses")),
        "dec_losses": _int_or_none(r.get("dec_losses")),
        "total_fights": _int_or_none(r.get("total_fights")),
        "finish_rate": _float_or_none(r.get("finish_rate")),
    }


def _int_or_none(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _float_or_none(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def cached_fighters() -> list[str]:
    df = _load_cache()
    if df.empty:
        return []
    return df["name"].tolist()
