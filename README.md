# Gap-Up OU Mean-Reversion: Filter Ordering Research (v2)

Simulated research into how the **ordering of trading filters** affects a
gap-up Ornstein-Uhlenbeck mean-reversion strategy on US large caps — and, in
v2, into how much of any apparent "ordering effect" is real versus an
artifact of **state-coupled filter design**.

## Research question

A strategy applies four gates to each gap-up candidate:
`A` VIX level, `B` VIX regime (rising/falling + term structure),
`C` OU model fit (β < 0, half-life in range), `D` entry signals
(oversold residual + RSI).

Since logical AND is commutative, *pure* gates must admit an identical trade
set under all 24 orderings. v1 of this project nonetheless found a return
spread across orderings — because gate B wrote regime parameters that gates
C and D silently defaulted when B hadn't run yet. **v2 makes that coupling an
object of study instead of a hidden bug:**

* **Coupled mode** reproduces the original side-effecting gate chain.
* **Decoupled mode** computes all context (regime, OU fit) before any gate
  runs; gates become pure predicates. The engine then *verifies* trade-set
  order-invariance at runtime.
* The difference in return spread between the two modes **is the measured
  coupling artifact**. What legitimately remains order-dependent in decoupled
  mode: rejection attribution ("which gate kills trades") and compute cost
  (cheap filters first).

## What v2 adds over v1

| Issue in v1 | Fix in v2 |
|---|---|
| Ordering effect partly a state-coupling artifact | Coupled vs decoupled modes; runtime order-invariance verification |
| Zero transaction costs | Round-trip cost model, sensitivity grid 0/5/10/20 bps |
| Arbitrary fixed SL/TP (1.5%/1.75%) in price space | Sigma-space exits: stop at entry residual − k·σ(OU), target at fitted OU mean; k sensitivity grid |
| Kruskal–Wallis on 24 correlated runs (invalid i.i.d. assumption) | Friedman test across non-overlapping sub-period blocks + moving-block bootstrap CI on the best-vs-worst spread |
| Sharpe annualized from per-trade returns | Sharpe from the daily equity curve |
| Everything in-sample 2020–2026 | Walk-forward (12m train / 3m test, k\_stop selected per window) + untouched 2025+ holdout evaluated once |
| `MAX_POS_PER_DAY` only ever filled one slot | Fills up to `MAX_CONCURRENT_POS` from the ranked candidate list |
| Monolithic 617-line script | Modules: `config / data / indicators / gates / engine / stats_tests / walkforward / plots` + unit tests |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -v        # unit tests (OU half-life recovery, gate invariance)
python src/run_research.py        # full pipeline (downloads data via yfinance on first run)
python src/run_research.py --no-holdout   # while iterating, keep the holdout sealed
```

## Pipeline

1. Coupled-mode 24-permutation experiment on the development period (≤ 2024).
2. Decoupled-mode rerun → order-invariance verification → coupling artifact = spread(coupled) − spread(decoupled).
3. Cost sensitivity across 0/5/10/20 bps round-trip.
4. Sub-period replication (4 blocks) → Friedman test; block-bootstrap 95% CI on the best-minus-worst per-trade P&L difference.
5. Walk-forward over the dev period (k\_stop re-selected each window); reports in-sample → out-of-sample decay.
6. Holdout (2025–) evaluated once with the frozen configuration.

## Results

> Run the pipeline on real data and paste the headline numbers + plots here:
> coupling artifact (pp), cost-survival table, Friedman p, bootstrap CI,
> walk-forward decay, holdout row. Plots land in `plots/`, tables in `results/`.

A methodological footnote surfaced by the unit tests: OLS β on a pure random
walk is downward-biased (the Dickey–Fuller problem), so the β < 0 check alone
almost never rejects a random walk. The effective mean-reversion filter is
the **half-life range** (random walks fit with median half-life ≈ 67 on
500-obs windows vs ≈ 5 for a true OU(5); the HL ≤ 30 gate rejects ~85% of
them).

## Known limitations

* **Survivorship-biased universe**: today's >$10B Tech/Financial large caps
  applied retroactively to 2020. Absolute returns are inflated; the
  ordering *comparison* is less affected (shared universe), but no
  performance number here should be read as attainable.
* **Daily bars, close entries**: signals use same-day close-derived values,
  so entries are at that close; intraday stop fills are approximated via the
  day's low against an EMA-mapped residual threshold. Same-bar stop/target
  ambiguity is resolved stop-first (conservative).
* **Cost model is flat bps**: no market impact, borrow, or volume
  constraints. Fine at this size for liquid large caps; not for scale.
* **yfinance data**: adjusted closes, no point-in-time corrections.
* **Walk-forward optimizes one parameter** (k\_stop). Threshold sets in
  `REGIME_PARAMS` remain hand-set and are a residual in-sample choice,
  disclosed here rather than hidden.

## Repo layout

```
src/        config, data, indicators, gates, engine, stats_tests, walkforward, plots, run_research
tests/      unit tests (OU half-life recovery, RW rejection, RSI/EMA sanity, gate order-invariance)
data/       cached CSVs (gitignored)
results/    output tables
plots/      output charts
```
