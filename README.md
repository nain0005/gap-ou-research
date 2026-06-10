# Filter-Ordering Effects in a Gap-Up OU Mean-Reversion Strategy

Simulated research into whether the **ordering of trading filters** affects a
gap-up Ornstein–Uhlenbeck mean-reversion strategy on US large caps — and how
much of any apparent "ordering effect" is real versus an artifact of
**state-coupled filter design**.

**Headline result: the entire ordering effect is an artifact.** With coupled
gates the 24 orderings show a 3.09pp net-return spread; with decoupled (pure)
gates the trade set is runtime-verified identical across all 24 orderings
(spread: 0.00pp). The residual spread is statistically indistinguishable from
noise, and the strategy's edge does not survive 5 bps of round-trip costs.

## Research question

The strategy applies four gates to each gap-up candidate:
`A` VIX level, `B` VIX regime (rising/falling + term structure),
`C` OU model fit (β < 0, half-life in range), `D` entry signals
(oversold residual + RSI).

Since logical AND is commutative, *pure* gates must admit an identical trade
set under every ordering — so any P&L difference across orderings can only
come from gates sharing state. Here, gate B sets regime-conditional
parameters that gates C and D consume; if C or D runs before B, they fall
back to default parameters. The study quantifies this by running both
implementations:

* **Coupled mode**: the side-effecting gate chain (B writes parameters, C/D
  default when B hasn't run).
* **Decoupled mode**: all context (regime, OU fit) is computed before any
  gate runs; gates become pure predicates, and the engine *verifies*
  trade-set order-invariance at runtime.

The difference in return spread between the modes **is the measured coupling
artifact**. What legitimately remains order-dependent in decoupled mode:
rejection attribution ("which gate kills trades") and compute cost (cheap
filters first).

## Results (2020-07 – 2024-12 development period, 5 bps costs unless noted)

**Coupling artifact.** Coupled spread **3.09pp** (best ordering −0.05%, worst
−3.14%); decoupled spread **0.00pp** with order-invariance VERIFIED — all 24
orderings produce the identical 23 trades. 100% of the apparent ordering
effect was the coupling artifact.

**Residual spread is noise.** Friedman test across four sub-period blocks
(categories as treatments): χ² = 0.667, **p = 0.72**. Moving-block bootstrap
on best-vs-worst per-trade P&L: +0.38pp, **95% CI [−0.75, +2.44]** — includes
zero. The best ordering and best category also flip across sub-periods.

**Costs kill the strategy.** Orderings profitable: 8/24 at 0 bps, **0/24 at
5 bps** and above. Median net return at a realistic 5 bps: −1.61%.

**Walk-forward (14 windows, 12m train / 3m test, stop-width k selected per
window):** mean in-sample −0.44% vs out-of-sample −0.23%; only 3/14 windows
OOS-positive; 7/14 windows contained zero trades.

**Gate funnel (why so few trades):** of 11,275 gap-up candidates, gate B's
backwardation requirement (VIX > VIX3M) admits only **2.5%** — backwardation
occurred on just **2.9% of trading days** in the period. Gate C (OU fit +
half-life range) passes **95%** of candidates and is therefore nearly
non-selective; gate D passes 37% of A∧B∧C survivors. The strategy is
structurally a *stressed-market-only* strategy, dormant ~97% of the time —
which means it produces too few trades (~5/yr) to ever reach statistical
significance at daily frequency. Trade scarcity is a design property, not a
bug.

**Holdout (2025-01 – 2026-04, frozen config, evaluated once):** 25 trades,
64% win rate, +22.1% net, Sharpe 1.38. **Not evidence of edge**: n = 25 from
a crisis-gated strategy is noise-compatible and contradicts every dev-period
result; reported for completeness only.

**Why the OU model barely selects:** two compounding effects. OLS β on short
windows is downward-biased (the Dickey–Fuller problem), so β < 0 holds even
for random walks (unit-tested: random walks fit with median half-life ≈ 67 vs
≈ 5 for a true OU(5); the half-life range does the real filtering). And
EMA-residuals are mean-reverting *by construction*, so an OU fit on them
nearly always lands in a wide half-life band. Net: the OU model contributes
the exit target (reversion to the fitted mean) but negligible entry
selection — entries are determined by the VIX regime and oversold/RSI gates.

![Returns by ordering](plots/01_returns_coupled.png)
![Cost sensitivity](plots/05_cost_sensitivity.png)
![Walk-forward](plots/06_walkforward.png)

## Methodology safeguards

| Pitfall | Mitigation |
|---|---|
| State-coupled gates masquerading as a finding | Coupled vs decoupled modes; runtime order-invariance verification |
| Zero transaction costs | Round-trip cost model, sensitivity grid 0/5/10/20 bps |
| Arbitrary fixed SL/TP in price space | Sigma-space exits: stop at entry residual − k·σ(OU), target at the fitted OU mean; k sensitivity grid |
| Kruskal–Wallis on 24 correlated runs (invalid i.i.d. assumption) | Friedman test across non-overlapping sub-period blocks + moving-block bootstrap CI |
| Sharpe annualized from per-trade returns | Sharpe from the daily equity curve |
| In-sample-only evaluation | Walk-forward (12m train / 3m test) + untouched 2025+ holdout evaluated once |
| Silent assumptions | Same-bar stop/target ambiguity resolved stop-first; all parameters centralized and documented in `config.py` |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -v          # unit tests (OU half-life recovery, gate invariance)
python src/run_research.py          # full pipeline (downloads data via yfinance on first run)
python src/run_research.py --no-holdout   # keep the holdout sealed while iterating
python src/diagnose.py              # gate funnel diagnostic
```

## Pipeline

1. Coupled-mode 24-permutation experiment on the development period (≤ 2024).
2. Decoupled-mode rerun → order-invariance verification → coupling artifact = spread(coupled) − spread(decoupled).
3. Cost sensitivity across 0/5/10/20 bps round-trip.
4. Sub-period replication (4 blocks) → Friedman test; block-bootstrap 95% CI on the best-minus-worst per-trade P&L difference.
5. Walk-forward over the dev period (stop-width k re-selected each window); reports in-sample → out-of-sample decay.
6. Holdout (2025–) evaluated once with the frozen configuration.
7. Gate funnel diagnostic: unconditional and sequential pass rates per gate.

## Known limitations

* **Survivorship-biased universe**: today's >$10B Tech/Financial large caps
  applied retroactively to 2020. Absolute returns are inflated; the ordering
  *comparison* is less affected (shared universe), but no performance number
  here should be read as attainable.
* **Daily bars, close entries**: signals use same-day close-derived values,
  so entries are at that close; intraday stop fills are approximated via the
  day's low against an EMA-mapped residual threshold.
* **Cost model is flat bps**: no market impact, borrow, or volume
  constraints. Reasonable at this size for liquid large caps; not for scale.
* **yfinance data**: adjusted closes, no point-in-time corrections.
* **Walk-forward optimizes one parameter** (stop width k). Threshold sets in
  `REGIME_PARAMS` remain hand-set — a residual in-sample choice, disclosed
  rather than hidden.
* **Sample size**: the backwardation gate caps the strategy at ~5 trades/yr
  on daily bars, below any threshold for statistical validation. Testing this
  strategy class properly requires intraday data.

## Repo layout

```
src/        config, data, indicators, gates, engine, stats_tests, walkforward, plots, diagnose, run_research
tests/      unit tests (OU half-life recovery, RW half-life bias, RSI/EMA sanity, gate order-invariance)
results/    output tables (summaries, cost sensitivity, walk-forward, holdout, funnel)
plots/      output charts
data/       price CSV cache (created on first run; gitignored)
```
