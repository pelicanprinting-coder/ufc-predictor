"""
Consolidate fetched OpticOdds tool outputs into opticodds_snapshot.json.

Reads previously-fetched fixture and odds JSON from the current session's
tool call outputs and produces the initial snapshot the Streamlit app reads
when the OpticOdds bridge is unavailable (public deploy).
"""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path

WORKSPACE_TOOL_DIR = Path("current_session_context/tool_calls/call_external_tool")
NEAR_TERM_FILE = Path("/tmp/near_term_fixtures.json")
OUT = Path("/home/user/workspace/ufc-predictor/opticodds_snapshot.json")


def load_tool_output(name: str) -> dict:
    return json.loads((WORKSPACE_TOOL_DIR / name).read_text())


def find_odds_outputs() -> list[dict]:
    """Return list of parsed OpticOdds response payloads (fixtures/odds responses)."""
    results = []
    for path in glob.glob(str(WORKSPACE_TOOL_DIR / "output_*.json")):
        try:
            raw = json.loads(Path(path).read_text())
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        if raw.get("tool") != "opticodds":
            continue
        result = raw.get("result") or {}
        if isinstance(result, dict) and result.get("status_code") == 200:
            results.append(result)
    return results


def main():
    # Load near-term fixture list (fallback discovery)
    near_term = json.loads(NEAR_TERM_FILE.read_text()) if NEAR_TERM_FILE.exists() else []
    fixtures = []
    seen = set()
    for f in near_term:
        if f["id"] in seen:
            continue
        seen.add(f["id"])
        fixtures.append({
            "fixture_id": f["id"],
            "start_date": f.get("start", ""),
            "home_name": f.get("home", ""),
            "away_name": f.get("away", ""),
            "event_name": f.get("event", "") or "UFC",
            "venue": "",
            "has_odds": False,  # will flip to True below if we found odds
        })

    fixture_by_id = {f["fixture_id"]: f for f in fixtures}

    odds_by_fixture = {}
    payloads = find_odds_outputs()
    print(f"Found {len(payloads)} OpticOdds payloads in tool call cache")

    for payload in payloads:
        data = (payload.get("data") or {}).get("data") or []
        for fx in data:
            fx_id = fx.get("id")
            if not fx_id:
                continue
            odds_rows = fx.get("odds") or []
            if not odds_rows:
                continue
            home = (fx.get("home_competitors") or [{}])[0].get("name", "")
            away = (fx.get("away_competitors") or [{}])[0].get("name", "")
            fixture_dict = {
                "fixture_id": fx_id,
                "start_date": fx.get("start_date", ""),
                "home_name": home,
                "away_name": away,
                "event_name": fx.get("season_week", "") or fx.get("season_type", ""),
                "venue": fx.get("venue_name", "") or "",
                "has_odds": True,
            }
            # Ensure the fixture list includes this one
            if fx_id not in fixture_by_id:
                fixture_by_id[fx_id] = fixture_dict
                fixtures.append(fixture_dict)
            else:
                fixture_by_id[fx_id].update(fixture_dict)

            # Merge odds rows; dedupe by id
            existing = odds_by_fixture.setdefault(fx_id, {
                "fixture": fixture_dict,
                "odds": [],
                "fetched_at": time.time(),
            })
            existing_ids = {o["_id"] for o in existing["odds"] if "_id" in o}
            for o in odds_rows:
                if o.get("id") in existing_ids:
                    continue
                existing["odds"].append({
                    "_id": o.get("id"),
                    "sportsbook": o.get("sportsbook", ""),
                    "market": o.get("market", ""),
                    "market_id": o.get("market_id", ""),
                    "name": o.get("name", ""),
                    "selection": o.get("selection", ""),
                    "price": int(o.get("price", 0)),
                    "points": o.get("points"),
                    "grouping_key": o.get("grouping_key", ""),
                    "deep_link": (o.get("deep_link", {}) or {}).get("desktop"),
                    "timestamp": float(o.get("timestamp", time.time())),
                })

    # Strip internal id
    for entry in odds_by_fixture.values():
        for o in entry["odds"]:
            o.pop("_id", None)

    snapshot = {
        "fetched_at": time.time(),
        "sportsbooks": ["DraftKings", "FanDuel", "BetMGM", "Caesars", "ESPN BET", "Fanatics", "Pinnacle"],
        "markets": ["Moneyline", "Method of Victory", "Total Rounds", "Winning Round"],
        "fixtures": fixtures,
        "odds_by_fixture": odds_by_fixture,
    }

    OUT.write_text(json.dumps(snapshot, indent=2))
    print(f"Wrote {OUT}")
    print(f"  fixtures: {len(fixtures)}")
    print(f"  fixtures with odds: {len(odds_by_fixture)}")
    for fx_id, entry in odds_by_fixture.items():
        f = entry["fixture"]
        print(f"    {f['home_name']} vs {f['away_name']} ({len(entry['odds'])} odds rows)")


if __name__ == "__main__":
    main()
