"""
Join Fight Matrix ratings to the Apex V2 fighter lookup.

Reads:
    fightmatrix_ratings.csv (from fightmatrix_client.py)
    fighter_lookup_apex_v2.csv (V2 model roster)

Produces:
    fighter_lookup_apex_v2_with_fm.csv (roster + fm_* columns)
    fightmatrix_join_report.txt (match statistics + unmatched samples)

Matching strategy:
    1. Normalize names: lowercase, strip diacritics, collapse whitespace.
    2. Exact match on normalized name.
    3. Fallback: fuzzy match with rapidfuzz, threshold 88, requires
       last-name token overlap to avoid false positives.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from rapidfuzz import fuzz, process

    _HAS_FUZZ = True
except ImportError:
    _HAS_FUZZ = False


ROOT = Path(__file__).parent
FM_CSV = ROOT / "fightmatrix_ratings.csv"
APEX_CSV = ROOT / "fighter_lookup_apex_v2.csv"
OUT_CSV = ROOT / "fighter_lookup_apex_v2_with_fm.csv"
REPORT_TXT = ROOT / "fightmatrix_join_report.txt"


def normalize_name(name: str) -> str:
    """Lowercase, strip diacritics, collapse whitespace."""
    if not isinstance(name, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_only = nfkd.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^\w\s]", "", ascii_only).lower()
    return re.sub(r"\s+", " ", cleaned).strip()


def last_name(norm: str) -> str:
    parts = norm.split()
    return parts[-1] if parts else ""


def build_fm_index(fm: pd.DataFrame) -> Dict[str, pd.Series]:
    """Map normalized fighter name -> best row (highest total ratings)."""
    fm = fm.copy()
    fm["norm"] = fm["fighter_name"].map(normalize_name)
    # Score each row by number of non-null ratings so we prefer complete rows
    fm["completeness"] = (
        fm["glicko1"].notna().astype(int)
        + fm["whr"].notna().astype(int)
        + fm["k170"].notna().astype(int)
    )
    fm = fm.sort_values(["completeness", "glicko1"], ascending=False)
    return {row["norm"]: row for _, row in fm.drop_duplicates("norm").iterrows()}


def match_fighter(
    query_norm: str,
    fm_index: Dict[str, pd.Series],
    all_norms: List[str],
    threshold: int = 88,
) -> Tuple[Optional[pd.Series], str]:
    """Return (matched_row, method) or (None, 'no_match')."""
    if not query_norm:
        return None, "empty"
    # 1. Exact
    if query_norm in fm_index:
        return fm_index[query_norm], "exact"
    if not _HAS_FUZZ:
        return None, "no_fuzz_lib"
    # 2. Fuzzy with last-name constraint
    q_last = last_name(query_norm)
    if not q_last:
        return None, "no_last"
    candidates = [n for n in all_norms if last_name(n) == q_last]
    if not candidates:
        return None, "no_candidate"
    best = process.extractOne(query_norm, candidates, scorer=fuzz.WRatio)
    if best and best[1] >= threshold:
        return fm_index[best[0]], f"fuzzy_{best[1]}"
    return None, "below_threshold"


def main() -> None:
    fm = pd.read_csv(FM_CSV)
    apex = pd.read_csv(APEX_CSV)
    print(f"FM ratings: {len(fm):,} rows")
    print(f"Apex roster: {len(apex):,} rows")

    fm_index = build_fm_index(fm)
    all_norms = list(fm_index.keys())
    print(f"FM unique normalized names: {len(all_norms):,}")

    apex["norm"] = apex["fighter_name"].map(normalize_name)

    fm_cols = {
        "fm_fighter_id": [],
        "fm_matched_name": [],
        "fm_match_method": [],
        "fm_division": [],
        "fm_glicko1": [],
        "fm_whr": [],
        "fm_k170": [],
        "fm_glicko1_rank": [],
        "fm_whr_rank": [],
        "fm_k170_rank": [],
        "fm_blended": [],
        "fm_profile_url": [],
    }
    method_counts: Dict[str, int] = {}
    unmatched_samples: List[str] = []

    for _, row in apex.iterrows():
        match, method = match_fighter(row["norm"], fm_index, all_norms)
        method_counts[method] = method_counts.get(method, 0) + 1
        if match is None:
            for k in fm_cols:
                fm_cols[k].append("")
            fm_cols["fm_match_method"][-1] = method
            if len(unmatched_samples) < 30:
                unmatched_samples.append(row["fighter_name"])
        else:
            fm_cols["fm_fighter_id"].append(str(match["fighter_id"]))
            fm_cols["fm_matched_name"].append(match["fighter_name"])
            fm_cols["fm_match_method"].append(method)
            fm_cols["fm_division"].append(match["division"])
            fm_cols["fm_glicko1"].append(match.get("glicko1", ""))
            fm_cols["fm_whr"].append(match.get("whr", ""))
            fm_cols["fm_k170"].append(match.get("k170", ""))
            fm_cols["fm_glicko1_rank"].append(match.get("glicko1_rank", ""))
            fm_cols["fm_whr_rank"].append(match.get("whr_rank", ""))
            fm_cols["fm_k170_rank"].append(match.get("k170_rank", ""))
            fm_cols["fm_blended"].append(match.get("blended", ""))
            fm_cols["fm_profile_url"].append(match.get("profile_url", ""))

    for k, v in fm_cols.items():
        apex[k] = v
    apex = apex.drop(columns=["norm"])
    apex.to_csv(OUT_CSV, index=False)

    # Report
    report_lines = []
    report_lines.append(f"Apex roster: {len(apex):,}")
    report_lines.append(f"FM ratings pool: {len(fm):,} rows / {len(all_norms):,} unique names")
    report_lines.append(f"Output: {OUT_CSV.name}")
    report_lines.append("")
    report_lines.append("Match methods:")
    for m, c in sorted(method_counts.items(), key=lambda x: -x[1]):
        pct = c / len(apex) * 100
        report_lines.append(f"  {m:20s} {c:5d}  ({pct:5.1f}%)")
    matched = sum(c for m, c in method_counts.items() if m in ("exact",) or m.startswith("fuzzy_"))
    report_lines.append(f"\nMatched:   {matched:,} ({matched/len(apex)*100:.1f}%)")
    report_lines.append(f"Unmatched: {len(apex)-matched:,}")
    report_lines.append("\nSample unmatched Apex fighters:")
    for n in unmatched_samples[:30]:
        report_lines.append(f"  - {n}")

    # Verify the coverage on fighters with blended values
    blended_ok = apex[apex["fm_blended"] != ""].shape[0]
    report_lines.append(f"\nRoster fighters with FM blended score: {blended_ok:,} ({blended_ok/len(apex)*100:.1f}%)")

    REPORT_TXT.write_text("\n".join(report_lines))
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
