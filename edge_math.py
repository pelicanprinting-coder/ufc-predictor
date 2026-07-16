"""
Betting math helpers for CagePicks Apex V2.

American odds → implied probability, no-vig fair prices,
Kelly stake sizing, and best-line detection across sportsbooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


def american_to_implied(price: int) -> float:
    """Convert American odds to implied probability (0-1)."""
    if price == 0:
        return 0.0
    if price > 0:
        return 100.0 / (price + 100)
    return abs(price) / (abs(price) + 100)


def american_to_decimal(price: int) -> float:
    if price > 0:
        return 1 + price / 100.0
    return 1 + 100.0 / abs(price)


def implied_to_american(p: float) -> int:
    """Convert probability back to American odds (rounded)."""
    if p <= 0 or p >= 1:
        return 0
    if p >= 0.5:
        return -int(round(p / (1 - p) * 100))
    return int(round((1 - p) / p * 100))


def no_vig_two_way(p_a: float, p_b: float) -> tuple[float, float]:
    """Strip vig from a two-way market. Returns fair (p_a, p_b) summing to 1."""
    total = p_a + p_b
    if total <= 0:
        return (0.0, 0.0)
    return (p_a / total, p_b / total)


def no_vig_from_prices(price_a: int, price_b: int) -> tuple[float, float]:
    return no_vig_two_way(american_to_implied(price_a), american_to_implied(price_b))


@dataclass
class BookQuote:
    sportsbook: str
    price: int
    implied: float
    deep_link: str | None = None


@dataclass
class MarketSnapshot:
    market: str
    selection_a: str
    selection_b: str
    quotes_a: list[BookQuote]
    quotes_b: list[BookQuote]
    consensus_a: float  # no-vig probability
    consensus_b: float
    best_a: BookQuote | None
    best_b: BookQuote | None
    sharp_a: BookQuote | None = None  # Pinnacle if available
    sharp_b: BookQuote | None = None


def build_moneyline_snapshot(
    odds_rows: list,  # list[OddsRow]
    fighter_a: str,
    fighter_b: str,
) -> MarketSnapshot | None:
    """Build a two-way moneyline snapshot from raw odds rows."""
    quotes_a: list[BookQuote] = []
    quotes_b: list[BookQuote] = []

    for row in odds_rows:
        if row.market != "Moneyline":
            continue
        name = row.name or row.selection
        if not name:
            continue
        q = BookQuote(
            sportsbook=row.sportsbook,
            price=row.price,
            implied=american_to_implied(row.price),
            deep_link=row.deep_link,
        )
        if _name_matches(name, fighter_a):
            quotes_a.append(q)
        elif _name_matches(name, fighter_b):
            quotes_b.append(q)

    if not quotes_a or not quotes_b:
        return None

    # Best price = highest American value for +odds bets, closest to 0 for -odds
    best_a = max(quotes_a, key=lambda q: _price_value(q.price))
    best_b = max(quotes_b, key=lambda q: _price_value(q.price))

    # Consensus no-vig from average implied
    p_a_avg = mean(q.implied for q in quotes_a)
    p_b_avg = mean(q.implied for q in quotes_b)
    consensus_a, consensus_b = no_vig_two_way(p_a_avg, p_b_avg)

    sharp_a = next((q for q in quotes_a if q.sportsbook == "Pinnacle"), None)
    sharp_b = next((q for q in quotes_b if q.sportsbook == "Pinnacle"), None)

    return MarketSnapshot(
        market="Moneyline",
        selection_a=fighter_a,
        selection_b=fighter_b,
        quotes_a=quotes_a,
        quotes_b=quotes_b,
        consensus_a=consensus_a,
        consensus_b=consensus_b,
        best_a=best_a,
        best_b=best_b,
        sharp_a=sharp_a,
        sharp_b=sharp_b,
    )


def _price_value(price: int) -> float:
    """Higher value = better for the bettor. +200 > +150 > -110 > -200."""
    if price > 0:
        return price
    return -1.0 / abs(price) * 10000  # -110 → -90.9, -200 → -50


def _name_matches(a: str, b: str) -> bool:
    return _norm(a) == _norm(b)


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).strip()


# --------------------------- edge computation ---------------------------

@dataclass
class EdgeReport:
    fighter: str
    model_prob: float
    market_prob: float  # no-vig consensus
    edge_pct: float  # (model - market) * 100
    best_price: int
    best_book: str
    ev_per_dollar: float  # EV of $1 stake at best price
    kelly_full: float  # full kelly fraction of bankroll
    kelly_quarter: float
    verdict: str  # PLAY / PASS / LEAN


def compute_edge(
    fighter: str,
    model_prob: float,
    market_prob: float,
    best_quote: BookQuote,
) -> EdgeReport:
    """Compute edge, EV, and Kelly stake for a single side."""
    decimal = american_to_decimal(best_quote.price)
    b = decimal - 1
    p = model_prob
    q = 1 - p

    ev_per_dollar = p * b - q  # expected value on $1 stake
    kelly_full = (b * p - q) / b if b > 0 else 0.0
    kelly_full = max(0.0, kelly_full)

    edge_pct = (model_prob - market_prob) * 100

    # Verdict thresholds — configurable, but sane defaults:
    # PLAY  : edge >= 4%  and  EV > 0
    # LEAN  : edge >= 2%  and  EV > 0
    # PASS  : otherwise
    if edge_pct >= 4.0 and ev_per_dollar > 0:
        verdict = "PLAY"
    elif edge_pct >= 2.0 and ev_per_dollar > 0:
        verdict = "LEAN"
    else:
        verdict = "PASS"

    return EdgeReport(
        fighter=fighter,
        model_prob=model_prob,
        market_prob=market_prob,
        edge_pct=edge_pct,
        best_price=best_quote.price,
        best_book=best_quote.sportsbook,
        ev_per_dollar=ev_per_dollar,
        kelly_full=kelly_full,
        kelly_quarter=kelly_full / 4.0,
        verdict=verdict,
    )


def format_american(price: int) -> str:
    return f"+{price}" if price > 0 else str(price)
