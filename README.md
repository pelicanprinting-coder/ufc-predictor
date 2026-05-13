# 🥊 UFC Fight Predictor

A machine learning system for predicting UFC fight outcomes, built with a fully leak-free rolling feature pipeline and an ensemble of 5 models.

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy (2023–2026, unseen data)** | **68.45%** |
| Temporal leakage test (train <2020, test 2024+) | 65.91% |
| Baseline (always predict Red corner) | 55.68% |
| Conviction 75%+ accuracy | 86.6% |
| Conviction 75%+ ROI (backtested) | +4.2% |
| Conviction 80%+ accuracy | 89.9% |
| Conviction 80%+ ROI (backtested) | +3.3% |

> Outperforms the best published academic results (66.71% GBDT, Yan et al. ACM ICIIP 2024).

---

## 🧠 Model Architecture

Ensemble of 5 models, trained on a strict temporal split (train: pre-2023, test: 2023–2026):

- **XGBoost** — n_estimators=500, max_depth=3, lr=0.01
- **LightGBM** — n_estimators=500, max_depth=3, lr=0.01
- **Random Forest** — n_estimators=500, max_depth=6
- **Logistic Regression** — with StandardScaler
- **CatBoost** — iterations=500, depth=3, lr=0.01

Final prediction = average of predicted probabilities across all 5 models.

---

## Anti-Leakage Pipeline

All features are computed **rolling** — for each fight, only data from **prior fights** is used. No future information leaks into any feature. Validated with a strict temporal leakage test (train on pre-2020, test on 2024+).

```
For fight at date D:
  features = f(all fights strictly before D)
```

This applies to all offensive stats, defensive stats, finishing rates, strike zone ratios, and fade metrics. Raw per-fight data from UFCStats is used rather than the pre-aggregated ufc-master.csv averages, which contain future data leakage.

---

## Feature Set (42 features)

### Fight Record
| Feature | Description |
|---------|-------------|
| win_streak_dif | Difference in current win streaks |
| lose_streak_dif | Difference in current losing streaks |
| longest_win_streak_dif | Difference in career-best win streaks |
| win_dif | Total wins differential |
| loss_dif | Total losses differential |
| total_round_dif | Total UFC rounds fought differential |
| total_title_bout_dif | Title fight experience differential |
| ko_dif | KO/TKO wins differential |
| sub_dif | Submission wins differential |

### Physical Attributes
| Feature | Description |
|---------|-------------|
| height_dif | Height difference (cm) |
| reach_dif | Reach difference (cm) |
| age_dif | Age difference (years) |

### Offensive Rolling Stats
| Feature | Description |
|---------|-------------|
| SLpM_diff | Significant strikes landed per minute differential |
| sig_str_acc_diff | Striking accuracy differential |
| td_avg_diff | Takedowns per fight differential |
| td_acc_diff | Takedown accuracy differential |
| sub_avg_diff | Submission attempts per fight differential |
| ctrl_avg_diff | Control time per fight (seconds) differential |
| kd_per_fight_diff | Knockdowns per fight differential |
| rev_avg_diff | Reversals per fight differential |

### Strike Zone Ratios (rolling)
| Feature | Description |
|---------|-------------|
| head_ratio_diff | % of strikes targeting the head |
| body_ratio_diff | % of strikes targeting the body |
| leg_ratio_diff | % of strikes targeting the legs |
| distance_ratio_diff | % of strikes thrown at distance |
| clinch_ratio_diff | % of strikes thrown in the clinch |
| ground_ratio_diff | % of strikes thrown on the ground |

### Fade / Cardio Metrics (rolling)
| Feature | Description |
|---------|-------------|
| sig_pct_fade_diff | Change in striking accuracy over career arc |
| td_fade_diff | Change in takedown volume over career arc |
| ctrl_fade_diff | Change in control time over career arc |

### Finishing Stats (rolling)
| Feature | Description |
|---------|-------------|
| finish_rate_diff | % of fights ended by finish |
| ko_rate_diff | % of fights ended by KO/TKO |
| sub_rate_diff | % of fights ended by submission |

### Defensive Stats (rolling)
| Feature | Description |
|---------|-------------|
| SApM_diff | Significant strikes absorbed per minute differential |
| str_def_diff | Strike defence % differential |
| td_def_diff | Takedown defence % differential |

### Style Clash Features
| Feature | Description |
|---------|-------------|
| style_clash_position | Euclidean distance in distance/clinch/ground ratios |
| style_clash_target | Euclidean distance in head/body/leg targeting ratios |
| grappling_advantage | Ground ratio gap x style clash position |

### Rankings
| Feature | Description |
|---------|-------------|
| R_ranked | Whether Red fighter holds a divisional ranking |
| B_ranked | Whether Blue fighter holds a divisional ranking |
| rank_dif | Ranking differential (Blue rank minus Red rank) |

### Market Intelligence
| Feature | Description |
|---------|-------------|
| odds_prob_diff | Implied probability differential from betting odds |

---

## Running Locally

```bash
git clone https://github.com/your-username/ufc-fight-predictor
cd ufc-fight-predictor
pip install -r requirements.txt
streamlit run app.py
```

**The Odds API key** — the app fetches real-time fight odds. Get a free key at [the-odds-api.com](https://the-odds-api.com) (500 requests/month on the free tier) and set it as an environment variable:

```bash
export ODDS_API_KEY=your_key_here        # macOS / Linux
set ODDS_API_KEY=your_key_here           # Windows
```

Or create a `.env` file in the project root:

```
ODDS_API_KEY=your_key_here
```

Odds are optional — the model will still predict fights without them, but market comparison features will be unavailable.

---

## Data Sources

| Source | Description |
|--------|-------------|
| ufc-master.csv | Fight records, physical attributes, rankings, odds (Kaggle) |
| ufc_fight_stats.csv | Per-round striking/grappling stats (UFCStats scrape) |
| ufc_fight_results.csv | Fight outcomes and finish methods |
| ufc_event_details.csv | Event dates and locations |
| The Odds API | Real-time fight odds (decimal/European format) |
| UFCStats.com | Live upcoming fight card via scraping |

---

## App (Streamlit)

**Tab 1 — Upcoming Events**
- Auto-loads next UFC card from UFCStats
- Fetches live odds from The Odds API
- Predicts all fights and sorts by conviction level (high conviction first)

**Tab 2 — Predict Fight**
- Manual fighter selection from 2,241 fighters
- Optional odds input (European decimal format)
- Model vs market probability comparison
- Full stats comparison table

### Conviction Levels
| Label | Threshold | Historical Accuracy |
|-------|-----------|-------------------|
| 🟢 High Conviction | 80%+ | ~90% |
| 🟡 Moderate Conviction | 75%+ | ~87% |
| 🔵 Slight Favourite | 65%+ | ~79% |
| ⚪ Too Close to Call | below 65% | — |

---

## What Was Tested and Rejected

| Approach | Delta | Reason |
|----------|-------|--------|
| Sliding window rolling (w=5) | -0.61pp | Too few fights per window |
| Exponential decay weighted rolling | -0.73pp | Fighter styles are stable long-term |
| ELO overall | -0.31pp | Redundant with odds |
| ELO by style (vs grappler/striker) | -0.18pp | Redundant with odds |
| ELO by finish method | -0.31pp | Low signal |
| Round-level cardio features | -1.04pp | 42% coverage only |
| Strength of schedule | -0.37pp | Captured by odds |
| Stance matchup | -0.43pp | Weak signal |
| Head-to-head record | -0.24pp | Only 44 rematches in dataset |
| Net striking (SLpM minus SApM) | -0.12pp | Redundant with individual features |
| Pace metrics (attempts/min) | -0.18pp | Redundant with accuracy + volume |
| Matchup interaction features | -0.18pp | Learned internally by tree models |
| Isotonic / Platt calibration | -0.31pp | Ensemble is already well-calibrated |
| Model per weight class | +0.06pp | Not worth added complexity |
| Opponent-adjusted stats | -0.61pp | Introduces variance, odds already capture quality |
| Problem reformulation (favourite vs underdog) | -0.97pp | Model already handles Red/Blue bias via odds |
| ELO as market filter | no gain | Odds dominate the signal |

---

## Model Evolution

| Version | Accuracy | Notes |
|---------|----------|-------|
| v9 | 63.04% | Had data leakage |
| v12 | 66.02% | First leak-free model (XGBoost only) |
| Ensemble 4 models | 67.84% | XGB + LGB + RF + LogReg |
| + Defensive features | 68.33% | Added SApM, str_def, td_def |
| + CatBoost | 68.33% | Better high-conviction calibration |
| + Reversals | **68.45%** | **Current best** |

---

## Published Benchmarks

| Study | Accuracy | Method |
|-------|----------|--------|
| Walsh, NCI (2022) | 61.48% | Neural Network, no rolling |
| Apelgren & Eklund, KTH (2024) | 63–70% | Logistic + Bayesian, 20 fights only |
| Yan et al., ACM ICIIP (2024) | 66.71% | GBDT, no temporal split |
| **This model** | **68.45%** | **Ensemble, leak-free rolling** |
