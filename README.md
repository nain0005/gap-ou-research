# Gap-Up OU Mean-Reversion: Filter Ordering Research

Tests how permuting 4 trading filters (VIX level, VIX regime, OU model, signals) across all 24 orderings affects strategy performance on Tech + Financial large-cap gap-up stocks.

## Quick Start

```bash
pip install -r requirements.txt
python src/run_research.py
```

## Strategy (matches QuantConnect version)

- Universe: Tech + Financial Services, >$10B market cap
- Gap-up: >0.5% from previous close
- VIX regime: BEST (rising + backwardation) or OKAY (falling + backwardation)
- OU model: β < 0, half-life in regime range
- Signals: oversold + RSI rising
- SL: 1.5%, TP: 1.75%, hold 2-3 days, no weekend carry

## Outputs

- `results/summary.csv` — all 24 permutation results
- `plots/01_returns.png` — returns by ordering
- `plots/02_categories.png` — regime-first vs model-first vs signal-first
- `plots/03_rejections.png` — which gate kills trades at each position
- `plots/04_scatter.png` — win rate vs net return

## Bugs Fixed from Original QC Code

1. Gap-up check cached once/day (was calling history() per bar = ~23,400 API calls/day)
2. Month halted check added to entry gating
3. Mean-reversion exit fixed (handles negative ou_mean correctly)
4. Cheap filters run before expensive OU model
5. Gap % logged in trade entries
