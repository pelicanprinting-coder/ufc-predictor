"""
OpticOdds client for CagePicks Apex V2.

Provides live UFC fixture + odds retrieval through Perplexity Computer's
OpticOdds connector. Ships two backends:

1. Runtime inside Perplexity Computer sandbox: uses `call_external_tool`
   via the local `pplx` bridge (works during development / interactive
   Streamlit sessions running inside the workspace).
2. Fallback: read a cached JSON snapshot committed alongside the app,
   so the public Streamlit deploy still shows the most recent slate.

The public Streamlit deploy will not have the connector bridge available,
so the Refresh button falls back to the snapshot. Users running locally
inside a Perplexity Computer session can hit "Refresh from OpticOdds" to
regenerate the snapshot.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# US-book default basket per OpticOdds skill guidance
DEFAULT_SPORTSBOOKS = [
    "DraftKings",
    "FanDuel",
    "BetMGM",
    "Caesars",
    "ESPN BET",
    "Fanatics",
    "Pinnacle",
]

# Markets we care about for V2 edge computation
CORE_MARKETS = [
    "Moneyline",
    "Method of Victory",
    "Total Rounds",
    "Winning Round",
]

SNAPSHOT_PATH = Path(__file__).parent / "opticodds_snapshot.json"


@dataclass
class Fixture:
    fixture_id: str
    start_date: str
    home_name: str
    away_name: str
    event_name: str
    venue: str
    has_odds: bool


@dataclass
class OddsRow:
    sportsbook: str
    market: str
    market_id: str
    name: str
    selection: str
    price: int  # American odds
    points: float | None
    grouping_key: str
    deep_link: str | None
    timestamp: float


@dataclass
class FixtureOdds:
    fixture: Fixture
    odds: list[OddsRow] = field(default_factory=list)
    fetched_at: float = field(default_factory=time.time)


# --------------------------- backend detection ---------------------------

def _connector_available() -> bool:
    """Return True when the OpticOdds connector bridge is reachable."""
    # Perplexity Computer exposes a shell binary `pplx-tool`; if it exists
    # and we're inside a workspace, we can call it. On Streamlit Cloud the
    # binary is absent and we fall back to snapshot.
    return bool(os.environ.get("PPLX_COMPUTER_SANDBOX")) or Path("/usr/local/bin/pplx-tool").exists()


def _call_opticodds(path: str, params: dict[str, Any]) -> dict:
    """Invoke OpticOdds via the Perplexity Computer sandbox tool bridge."""
    import subprocess
    payload = json.dumps({"path": path, "method": "GET", "params": params})
    result = subprocess.run(
        ["pplx-tool", "opticodds"],
        input=payload,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"OpticOdds bridge failed: {result.stderr}")
    return json.loads(result.stdout)


# --------------------------- public API ---------------------------

def fetch_active_ufc_fixtures() -> list[Fixture]:
    """List upcoming UFC fixtures via OpticOdds (or snapshot)."""
    if _connector_available():
        try:
            resp = _call_opticodds("/fixtures/active", {"league": "ufc"})
            fixtures = []
            for f in (resp.get("data", {}) or {}).get("data", []):
                home = (f.get("home_competitors") or [{}])[0].get("name", "")
                away = (f.get("away_competitors") or [{}])[0].get("name", "")
                fixtures.append(Fixture(
                    fixture_id=f["id"],
                    start_date=f.get("start_date", ""),
                    home_name=home,
                    away_name=away,
                    event_name=f.get("season_week", f.get("season_type", "")),
                    venue=f.get("venue_name", "") or "",
                    has_odds=f.get("has_odds", False),
                ))
            return fixtures
        except Exception as e:
            print(f"[opticodds] live fetch failed, falling back to snapshot: {e}")
    return _load_snapshot_fixtures()


def fetch_fixture_odds(
    fixture_id: str,
    sportsbooks: list[str] | None = None,
    markets: list[str] | None = None,
) -> FixtureOdds | None:
    """Fetch odds for a specific fixture."""
    sportsbooks = sportsbooks or DEFAULT_SPORTSBOOKS
    markets = markets or CORE_MARKETS

    if _connector_available():
        try:
            # Batch sportsbooks max 5 per request per skill guidance
            all_odds: list[dict] = []
            fixture_payload = None
            for i in range(0, len(sportsbooks), 5):
                batch = sportsbooks[i:i + 5]
                resp = _call_opticodds("/fixtures/odds", {
                    "fixture_id": fixture_id,
                    "sportsbook": batch,
                })
                items = (resp.get("data", {}) or {}).get("data", [])
                if not items:
                    continue
                fx = items[0]
                if fixture_payload is None:
                    fixture_payload = fx
                all_odds.extend(fx.get("odds", []) or [])

            if fixture_payload is None:
                return None

            odds_rows = [
                OddsRow(
                    sportsbook=o.get("sportsbook", ""),
                    market=o.get("market", ""),
                    market_id=o.get("market_id", ""),
                    name=o.get("name", ""),
                    selection=o.get("selection", ""),
                    price=int(o.get("price", 0)),
                    points=o.get("points"),
                    grouping_key=o.get("grouping_key", ""),
                    deep_link=(o.get("deep_link", {}) or {}).get("desktop"),
                    timestamp=float(o.get("timestamp", time.time())),
                )
                for o in all_odds
                if o.get("market") in markets or not markets
            ]

            home = (fixture_payload.get("home_competitors") or [{}])[0].get("name", "")
            away = (fixture_payload.get("away_competitors") or [{}])[0].get("name", "")
            return FixtureOdds(
                fixture=Fixture(
                    fixture_id=fixture_payload["id"],
                    start_date=fixture_payload.get("start_date", ""),
                    home_name=home,
                    away_name=away,
                    event_name=fixture_payload.get("season_week", ""),
                    venue=fixture_payload.get("venue_name", "") or "",
                    has_odds=True,
                ),
                odds=odds_rows,
            )
        except Exception as e:
            print(f"[opticodds] live odds fetch failed for {fixture_id}: {e}")
    return _load_snapshot_odds(fixture_id)


def refresh_snapshot(sportsbooks: list[str] | None = None, markets: list[str] | None = None) -> dict:
    """Pull all active UFC fixtures + odds and write to snapshot file."""
    sportsbooks = sportsbooks or DEFAULT_SPORTSBOOKS
    markets = markets or CORE_MARKETS

    fixtures = fetch_active_ufc_fixtures()
    snapshot = {
        "fetched_at": time.time(),
        "sportsbooks": sportsbooks,
        "markets": markets,
        "fixtures": [asdict(f) for f in fixtures],
        "odds_by_fixture": {},
    }
    for fx in fixtures:
        fo = fetch_fixture_odds(fx.fixture_id, sportsbooks, markets)
        if fo:
            snapshot["odds_by_fixture"][fx.fixture_id] = {
                "fixture": asdict(fo.fixture),
                "odds": [asdict(o) for o in fo.odds],
                "fetched_at": fo.fetched_at,
            }

    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2))
    return {"fixtures": len(fixtures), "with_odds": len(snapshot["odds_by_fixture"]), "path": str(SNAPSHOT_PATH)}


# --------------------------- snapshot loading ---------------------------

def _load_snapshot() -> dict | None:
    if not SNAPSHOT_PATH.exists():
        return None
    try:
        return json.loads(SNAPSHOT_PATH.read_text())
    except Exception:
        return None


def _load_snapshot_fixtures() -> list[Fixture]:
    snap = _load_snapshot()
    if not snap:
        return []
    return [Fixture(**f) for f in snap.get("fixtures", [])]


def _load_snapshot_odds(fixture_id: str) -> FixtureOdds | None:
    snap = _load_snapshot()
    if not snap:
        return None
    entry = snap.get("odds_by_fixture", {}).get(fixture_id)
    if not entry:
        return None
    return FixtureOdds(
        fixture=Fixture(**entry["fixture"]),
        odds=[OddsRow(**o) for o in entry.get("odds", [])],
        fetched_at=entry.get("fetched_at", 0),
    )


def snapshot_age_hours() -> float | None:
    snap = _load_snapshot()
    if not snap:
        return None
    return (time.time() - snap.get("fetched_at", 0)) / 3600.0
