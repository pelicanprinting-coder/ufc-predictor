"""
Fight Matrix ratings collector.

Pulls per-fighter ratings from fightmatrix.com public ranking pages for
three rating systems (Glicko-1, WHR, Elo K170) across all MMA divisions
supported by the site. Server-side rendered HTML — no browser needed.

Output: fightmatrix_ratings.csv with columns:
    fighter_id, fighter_name, division, country, age, record,
    glicko1, whr, k170,
    glicko1_rank, whr_rank, k170_rank,
    blended (= 0.55 * whr + 0.45 * glicko1)
    profile_url, captured_at

Usage:
    python fightmatrix_client.py            # full walk (all divisions x 3 systems)
    python fightmatrix_client.py middleweight   # single division
"""

from __future__ import annotations

import csv
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

BASE = "https://www.fightmatrix.com"
UA = "Mozilla/5.0 (compatible; CagePicks/1.0)"
REQUEST_DELAY = 0.15
MAX_WORKERS = 6

# Division slugs on fightmatrix.com (verified 2026-07-16)
DIVISIONS: List[str] = [
    "flyweight",
    "bantamweight",
    "featherweight",
    "lightweight",
    "welterweight",
    "middleweight",
    "light-heavyweight-185-205-lbs",
    "heavyweight-265-lbs",
    "strawweight",  # men's strawweight (exists but tiny)
    "womens-strawweight",
    "womens-flyweight",
    "womens-bantamweight",
    "womens-featheweight",  # note: FM's actual URL has the typo
    "womens-atomweight",
]

# Human-friendly division names for the output CSV
DIV_NAMES: Dict[str, str] = {
    "flyweight": "Flyweight",
    "bantamweight": "Bantamweight",
    "featherweight": "Featherweight",
    "lightweight": "Lightweight",
    "welterweight": "Welterweight",
    "middleweight": "Middleweight",
    "light-heavyweight-185-205-lbs": "Light Heavyweight",
    "heavyweight-265-lbs": "Heavyweight",
    "strawweight": "Men's Strawweight",
    "womens-strawweight": "Women's Strawweight",
    "womens-flyweight": "Women's Flyweight",
    "womens-bantamweight": "Women's Bantamweight",
    "womens-featheweight": "Women's Featherweight",
    "womens-atomweight": "Women's Atomweight",
}

SYSTEMS: Dict[str, str] = {
    "glicko1": "R_GLICKO1",
    "whr": "R_WHR",
    "k170": "R_ELOK170",
}


@dataclass
class FighterRow:
    fighter_id: str = ""
    fighter_name: str = ""
    division: str = ""
    country: str = ""
    age: Optional[int] = None
    record: str = ""
    profile_url: str = ""
    ratings: Dict[str, float] = field(default_factory=dict)  # system -> points
    ranks: Dict[str, int] = field(default_factory=dict)      # system -> formula rank


def _fetch_html(session: requests.Session, url: str) -> str:
    r = session.get(url, timeout=20)
    r.raise_for_status()
    return r.text


def _find_last_page(html: str) -> int:
    """Find the highest PageNum= link on the page."""
    matches = re.findall(r"PageNum=(\d+)", html)
    if not matches:
        return 1
    return max(int(m) for m in matches)


def _parse_ranking_page(html: str) -> List[FighterRow]:
    """Parse a single ranking page. Extracts one row per fighter."""
    soup = BeautifulSoup(html, "html.parser")
    rows: List[FighterRow] = []
    for tr in soup.find_all("tr", class_="rankRowX"):
        # Alternating rows use "tdRank" and "tdRankAlt" classes
        cells = tr.find_all(
            "td", class_=["tdRank", "tdRankAlt"], recursive=False
        )
        if len(cells) < 3:
            continue
        formula_rank_txt = cells[0].get_text(strip=True)
        try:
            formula_rank = int(formula_rank_txt)
        except ValueError:
            continue
        # Third cell contains the nested fighter table
        anchor = cells[2].find("a", href=re.compile(r"/fighter-profile/"))
        if not anchor:
            continue
        href = anchor["href"]
        m = re.search(r"/fighter-profile/([^/]+)/(\d+)/", href)
        if not m:
            continue
        name = unquote(m.group(1))
        fid = m.group(2)
        country_img = cells[2].find("img", alt=True)
        country = country_img["alt"] if country_img else ""
        # age in span after strong tag
        age = None
        age_span = anchor.find("span", style=re.compile(r"font-size: 8pt"))
        if age_span:
            am = re.search(r"\((\d+)\)", age_span.get_text())
            if am:
                age = int(am.group(1))
        # record: first inner td at width 9%
        record = ""
        rec_span = cells[2].find("span", style=re.compile(r"font-size: 9pt"))
        if rec_span:
            record = rec_span.get_text(strip=True)
        # points: div.tdBar text
        points = None
        bar = cells[2].find("div", class_="tdBar")
        if bar:
            bar_txt = bar.get_text(strip=True)
            try:
                points = float(bar_txt)
            except ValueError:
                points = None
        row = FighterRow(
            fighter_id=fid,
            fighter_name=name,
            country=country,
            age=age,
            record=record,
            profile_url=f"{BASE}{href}",
        )
        # Store formula rank and points under the calling system
        row._formula_rank = formula_rank  # type: ignore[attr-defined]
        row._points = points  # type: ignore[attr-defined]
        rows.append(row)
    return rows


def collect_division(
    session: requests.Session, division_slug: str
) -> Dict[str, FighterRow]:
    """Walk all three systems for a division. Returns {fighter_id: FighterRow}."""
    combined: Dict[str, FighterRow] = {}
    for sys_key, sys_param in SYSTEMS.items():
        url_p1 = f"{BASE}/mma-ranks/{division_slug}/?RF={sys_param}"
        try:
            html = _fetch_html(session, url_p1)
        except Exception as e:
            print(f"  ! {division_slug} {sys_key} page 1 failed: {e}")
            continue
        last_page = _find_last_page(html)
        print(f"  {division_slug} {sys_key}: {last_page} pages", flush=True)
        # Page 1 already fetched
        pages_html = [html]
        remaining = list(range(2, last_page + 1))

        def _grab(pn):
            url = f"{BASE}/mma-ranks/{division_slug}?PageNum={pn}&RF={sys_param}"
            return pn, _fetch_html(session, url)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(_grab, pn) for pn in remaining]
            for fut in as_completed(futures):
                try:
                    _, html_pn = fut.result()
                    pages_html.append(html_pn)
                except Exception as e:
                    print(f"    ! page failed: {e}", flush=True)
        # Parse each page
        for page_html in pages_html:
            for row in _parse_ranking_page(page_html):
                if row.fighter_id not in combined:
                    combined[row.fighter_id] = row
                    combined[row.fighter_id].division = DIV_NAMES.get(
                        division_slug, division_slug
                    )
                target = combined[row.fighter_id]
                target.ratings[sys_key] = row._points  # type: ignore[attr-defined]
                target.ranks[sys_key] = row._formula_rank  # type: ignore[attr-defined]
        time.sleep(REQUEST_DELAY)
    return combined


def write_csv(all_rows: List[FighterRow], out_path: Path) -> None:
    captured = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fieldnames = [
        "fighter_id",
        "fighter_name",
        "division",
        "country",
        "age",
        "record",
        "glicko1",
        "whr",
        "k170",
        "glicko1_rank",
        "whr_rank",
        "k170_rank",
        "blended",
        "profile_url",
        "captured_at",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            g = row.ratings.get("glicko1")
            wh = row.ratings.get("whr")
            k = row.ratings.get("k170")
            blended = None
            if g is not None and wh is not None:
                blended = round(0.45 * g + 0.55 * wh, 2)
            w.writerow(
                {
                    "fighter_id": row.fighter_id,
                    "fighter_name": row.fighter_name,
                    "division": row.division,
                    "country": row.country,
                    "age": row.age if row.age is not None else "",
                    "record": row.record,
                    "glicko1": g if g is not None else "",
                    "whr": wh if wh is not None else "",
                    "k170": k if k is not None else "",
                    "glicko1_rank": row.ranks.get("glicko1", ""),
                    "whr_rank": row.ranks.get("whr", ""),
                    "k170_rank": row.ranks.get("k170", ""),
                    "blended": blended if blended is not None else "",
                    "profile_url": row.profile_url,
                    "captured_at": captured,
                }
            )


def main(argv: List[str]) -> None:
    out_dir = Path(__file__).parent
    out_path = out_dir / "fightmatrix_ratings.csv"
    ckpt_dir = out_dir / "fm_ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    divisions_to_walk = argv[1:] if len(argv) > 1 else DIVISIONS
    all_rows: List[FighterRow] = []
    for div in divisions_to_walk:
        ckpt = ckpt_dir / f"{div}.csv"
        if ckpt.exists():
            print(f"\n== SKIP {div} (checkpoint exists) ==", flush=True)
            # Load the checkpoint
            with ckpt.open() as f:
                reader = csv.DictReader(f)
                for r in reader:
                    row = FighterRow(
                        fighter_id=r["fighter_id"],
                        fighter_name=r["fighter_name"],
                        division=r["division"],
                        country=r["country"],
                        age=int(r["age"]) if r["age"] else None,
                        record=r["record"],
                        profile_url=r["profile_url"],
                    )
                    for s in ("glicko1", "whr", "k170"):
                        if r[s]:
                            row.ratings[s] = float(r[s])
                        if r[f"{s}_rank"]:
                            row.ranks[s] = int(r[f"{s}_rank"])
                    all_rows.append(row)
            continue
        print(f"\n== Collecting {div} ==", flush=True)
        combined = collect_division(session, div)
        rows = list(combined.values())
        write_csv(rows, ckpt)
        all_rows.extend(rows)
        print(f"  -> {len(rows)} unique fighters (checkpoint saved)", flush=True)

    write_csv(all_rows, out_path)
    print(f"\nWrote {len(all_rows)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv)
