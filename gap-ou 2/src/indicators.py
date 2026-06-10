"""Indicator and model-fitting functions. Pure, vectorized where possible, unit-tested."""

import numpy as np


def ema(prices: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(prices), np.nan, dtype=float)
    if len(prices) < period:
        return out
    out[period - 1] = np.mean(prices[:period])
    k = 2.0 / (period + 1)
    for i in range(period, len(prices)):
        out[i] = prices[i] * k + out[i - 1] * (1 - k)
    return out


def rsi(prices: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(prices), np.nan, dtype=float)
    if len(prices) < period + 1:
        return out
    d = np.diff(prices)
    gains = np.where(d > 0, d, 0.0)
    losses = np.where(d < 0, -d, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    out[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss) if avg_loss != 0 else 100.0
    for i in range(period, len(d)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss) if avg_loss != 0 else 100.0
    return out


def fit_ou(resids) -> dict | None:
    """Fit a discrete-time OU (AR(1)) process to a residual window.

    dX_t = alpha + beta * X_{t-1} + eps  =>  X_t = mu + phi(X_{t-1} - mu) + eps
    with phi = 1 + beta. Mean reversion requires beta < 0 (phi < 1).
    Half-life = -ln(2) / ln(phi).
    """
    arr = np.asarray(resids, dtype=float)
    if len(arr) < 30:
        return None
    d = np.diff(arr)
    lag = arr[:-1]
    A = np.column_stack([np.ones(len(lag)), lag])
    try:
        coef, _, _, _ = np.linalg.lstsq(A, d, rcond=None)
    except np.linalg.LinAlgError:
        return None
    alpha, beta = float(coef[0]), float(coef[1])
    if beta >= 0:
        return None
    phi = 1.0 + beta
    if phi <= 0:
        return None
    half_life = -np.log(2.0) / np.log(phi)
    if not np.isfinite(half_life) or half_life <= 0:
        return None
    mu = -alpha / beta
    sigma = float(np.std(arr - mu, ddof=1))
    if sigma <= 0:
        return None
    return {"alpha": alpha, "beta": beta, "half_life": half_life,
            "mean": mu, "sigma": sigma}
