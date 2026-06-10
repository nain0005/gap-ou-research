"""Central configuration. Every research choice lives here, documented."""

TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "ORCL",
    "CRM", "ADBE", "INTC", "AMD", "QCOM", "NOW", "INTU", "AMAT",
    # Financial Services
    "JPM", "GS", "MS", "WFC", "BLK", "SCHW", "BX", "BAC", "C",
    "USB", "PNC", "AXP", "COF", "SPGI", "MCO", "ICE", "CME",
]
# KNOWN LIMITATION: this is a hindsight-selected universe (today's large caps
# applied retroactively). Survivorship bias inflates absolute returns; the
# filter-ordering *comparison* is less affected since all orderings share the
# same universe, but absolute performance numbers should not be trusted.

START = "2020-01-01"
END = "2026-04-30"

# ── Out-of-sample split ─────────────────────────────────────────────────────
# Development period: all design iteration happens here.
# Holdout period: touched exactly once, for the final table in the README.
DEV_END = "2024-12-31"
HOLDOUT_START = "2025-01-01"

# Sub-periods within the development window, used to replicate the ordering
# experiment for the Friedman test (fixes the broken i.i.d. assumption of a
# single-window Kruskal-Wallis).
DEV_SUBPERIODS = [
    ("2020-07-01", "2021-09-30"),
    ("2021-10-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
]

# ── Signal / strategy parameters ────────────────────────────────────────────
GAP_THRESHOLD = 0.005
VIX_MIN = 15.0
VIX_LOOKBACK = 6
BACKWARDATION_PROXY = 25.0

REGIME_PARAMS = {
    "BEST": {"pos_weight": 0.50, "oversold_sigma": 0.60,
             "rsi_floor": 15, "rsi_ceiling": 75, "min_hl": 1, "max_hl": 30},
    "OKAY": {"pos_weight": 0.20, "oversold_sigma": 1.0,
             "rsi_floor": 15, "rsi_ceiling": 50, "min_hl": 2, "max_hl": 18},
}

# ── Exits ───────────────────────────────────────────────────────────────────
# EXIT_MODE:
#   "sigma"     — exits defined in OU residual space (model-consistent).
#                 Stop at entry_residual - K_STOP * ou_sigma, target at the
#                 fitted OU mean. This replaces the arbitrary fixed 1.5%/1.75%.
#   "fixed_pct" — legacy mode kept for comparison (SL_PCT / TP_PCT below).
EXIT_MODE = "sigma"
K_STOP = 1.0                  # stop distance in OU sigmas below entry residual
K_STOP_GRID = [0.5, 1.0, 1.5, 2.0]   # sensitivity grid + walk-forward search space

SL_PCT = 0.015                # legacy fixed-pct mode only
TP_PCT = 0.0175               # legacy fixed-pct mode only

MIN_HOLD = 2
MAX_HOLD = 3
MAX_CONCURRENT_POS = 2        # renamed from MAX_POS_PER_DAY: it caps *concurrent*
                              # positions, and the engine now actually fills up
                              # to this limit (original code only ever took the
                              # single best candidate).

# Same-bar ambiguity rule (documented assumption): if a day's range touches
# both the stop and the target, we assume the STOP fired first (conservative).

INITIAL_CAP = 1_000_000

# ── Transaction costs ───────────────────────────────────────────────────────
# Round-trip cost in basis points applied to every trade's return.
# Large-cap US equities, marketable orders: ~2-5 bps round-trip is realistic
# for spread + fees at this size; we sweep a grid to show robustness.
COST_BPS = 5.0
COST_BPS_GRID = [0.0, 5.0, 10.0, 20.0]

# ── Indicators ──────────────────────────────────────────────────────────────
EMA_PERIOD = 21
RSI_PERIOD = 21
RESID_WINDOW = 60

MONTHLY_LOSS = -0.10
MONTHLY_PROFIT = 0.15

# ── Walk-forward ────────────────────────────────────────────────────────────
WF_TRAIN_MONTHS = 12
WF_TEST_MONTHS = 3
WF_STEP_MONTHS = 3

GATES = ["A", "B", "C", "D"]   # A=VIX level, B=VIX regime, C=OU model, D=signals
