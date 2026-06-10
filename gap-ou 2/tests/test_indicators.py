"""Unit tests. Run: python -m pytest tests/ -v"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from indicators import fit_ou, rsi, ema
from gates import apply_gates_decoupled, compute_context
from itertools import permutations


def simulate_ou(n, half_life, mu=0.0, sigma_eps=0.01, seed=0):
    """X_t = mu + phi (X_{t-1} - mu) + eps, phi = 2^(-1/HL)."""
    rng = np.random.default_rng(seed)
    phi = 2.0 ** (-1.0 / half_life)
    x = np.empty(n)
    x[0] = mu
    for i in range(1, n):
        x[i] = mu + phi * (x[i - 1] - mu) + rng.normal(0, sigma_eps)
    return x


def test_ou_half_life_recovery():
    """fit_ou should recover a known half-life within tolerance on a long series."""
    for hl_true in [2.0, 5.0, 10.0]:
        x = simulate_ou(20_000, hl_true, mu=0.02, sigma_eps=0.005, seed=1)
        fit = fit_ou(x)
        assert fit is not None
        assert abs(fit["half_life"] - hl_true) / hl_true < 0.15, \
            f"HL {hl_true}: got {fit['half_life']:.2f}"
        assert abs(fit["mean"] - 0.02) < 0.01


def test_random_walk_yields_long_half_life():
    """OLS beta on a random walk is downward-biased (Dickey-Fuller), so the
    beta<0 check alone rarely rejects a RW. What separates RWs from genuine
    OU processes is the FITTED HALF-LIFE: RWs fit with long half-lives that
    the strategy's HL-range gate (1-30) then rejects most of the time."""
    rng = np.random.default_rng(7)
    rw_hls = []
    for s in range(50):
        x = np.cumsum(rng.normal(0, 0.01, 500))
        f = fit_ou(x)
        if f is not None:
            rw_hls.append(f["half_life"])
    ou_hls = [fit_ou(simulate_ou(500, 5.0, seed=s))["half_life"]
              for s in range(50)]
    assert np.median(rw_hls) > 40          # RWs look slow-reverting
    assert np.median(ou_hls) < 10          # true OU(5) recovered as fast
    # the HL<=30 gate filters the majority of random walks
    assert np.mean(np.array(rw_hls) <= 30) < 0.35


def test_rsi_bounds():
    rng = np.random.default_rng(3)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
    r = rsi(p, 21)
    valid = r[~np.isnan(r)]
    assert len(valid) > 0
    assert valid.min() >= 0 and valid.max() <= 100


def test_rsi_monotone_up():
    p = np.linspace(100, 200, 100)  # strictly rising
    r = rsi(p, 14)
    assert r[~np.isnan(r)][-1] == 100.0


def test_ema_converges():
    p = np.full(300, 50.0)
    e = ema(p, 21)
    assert abs(e[-1] - 50.0) < 1e-9


def _synthetic_bar(seed=0):
    x = simulate_ou(80, 5.0, mu=0.0, sigma_eps=0.01, seed=seed)
    return {
        "ticker": "TEST", "close": 100.0, "gap": 0.01,
        "res": x[-1], "res_prev": x[-2] - 0.05,  # force deeply oversold
        "resid_window": x[:-1],
        "rsi": 40.0, "rsi_prev": 35.0,
        "vix": 22.0, "vix_prev": 20.0, "vix3m": 21.0,
    }


def test_decoupled_gates_are_order_invariant():
    """Pure gates: pass/fail must be identical under every ordering."""
    for seed in range(10):
        bar = _synthetic_bar(seed)
        outcomes = set()
        for perm in permutations(["A", "B", "C", "D"]):
            ok, _, _, _ = apply_gates_decoupled(dict(bar), list(perm))
            outcomes.add(ok)
        assert len(outcomes) == 1, f"seed {seed}: ordering changed the outcome"


def test_context_regime_assignment():
    bar = _synthetic_bar(1)
    ctx = compute_context(bar)
    assert ctx["regime"] == "BEST"      # vix 22 > vix_prev 20 => rising
    assert ctx["backwardated"] is True  # vix 22 > vix3m 21
