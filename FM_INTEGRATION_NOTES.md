# Fight Matrix Integration — Notes & Caveats

## What we built

1. **fightmatrix_client.py** — Public rating table collector. 14 divisions × 3 systems (Glicko-1, WHR, Elo K170), 8,545 fighters, checkpointed by division.
2. **fightmatrix_math.py** — Standard Elo win-probability formula. Verified against Fight Matrix's own published Program probabilities to 2 decimal places (matches exactly, confirming scale parity).
3. **fighter_lookup_apex_v2_with_fm.csv** — Fuzzy join to Apex V2 roster. 100% match on 2025+ active fighters (711/711); overall 65% since older/retired fighters churn.
4. **live_odds_tab.py Fight Matrix panel** — Per-system win probabilities, blended rating, and three-way consensus banner (Market vs V2 vs FM Blend).

## User-requested blend

**blended_rating = 0.45 · Glicko-1 + 0.55 · WHR**

Verified saturday example:
- Du Plessis blended = 2207.25; Usman blended = 2207.15 → literally 50/50
- Cannonier blended = 1948.95; Duncan blended = 1939.25 → Cannonier 51.4% (market has him at 25.5%)

## Three-way consensus classifier

Given `model_prob_a` (V2), `blended_prob_a` (FM), `market_prob_a`:

- **STRONG_CONSENSUS**: V2 and FM Blend both pick the same fighter, and both disagree with the market by ≥3%.
- **WEAK_CONSENSUS**: V2 and FM Blend agree against market, but edge is below 3%.
- **SPLIT**: V2 and FM Blend pick opposite fighters — pass on the bet.
- **MARKET_ALIGNED**: Everyone agrees with the market — no edge.

## ⚠️ Backtest caveat — LOOK-AHEAD BIAS

The 2026 backtest **cannot** validate FM Blend accuracy because Fight Matrix updates ratings after each fight. The ratings we captured today (2026-07-16) reflect every result from those 200 fights. Any accuracy metric on that data is contaminated by look-ahead.

Observed on 2026 slice (**illustrative, not predictive**):
- V2 baseline: 60.0%
- FM standalone: 94.1%  ← inflated by look-ahead
- V2 ∩ FM agreement: 94.1% ← inflated by look-ahead

## What IS legitimate

1. **Forward-looking use**: For fights that have NOT happened yet (Saturday's card, future events), the current ratings ARE the "as-of" ratings. So the STRONG_CONSENSUS signal for Cannonier +300 is fair.
2. **Direct market disagreement**: When FM Blend calls Du Plessis-Usman a coin flip vs market's 70/30, that's a live piece of information regardless of backtest.
3. **The Elo formula itself is validated** — it matches FM's published win probs exactly, so we trust the math.

## To get an honest backtest lift

Would need historical Fight Matrix rating snapshots (as-of each fight date). Options:
- Recompute Glicko/WHR from scratch over fight history — possible but expensive.
- Contact Fight Matrix for historical rating archives.
- Prospective tracking: capture ratings weekly going forward, backtest lift on new events.

**Recommended next step**: capture a rating snapshot every Monday going forward. After ~10 events, we'll have clean forward-out-of-sample data to measure real lift.
