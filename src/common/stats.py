"""Interval estimates and equivalence tests used across the experiments."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def wilson_interval(successes: int, trials: int, conf: float = 0.95):
    """Wilson score interval for a binomial proportion."""
    if trials == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(0.5 + conf / 2)
    p = successes / trials
    den = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / den
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_t_interval(values, conf: float = 0.95):
    """Mean, sample standard deviation and Student t interval of the mean.
    The values must be independent replicates (here: independent devices)."""
    x = np.asarray(values, dtype=float)
    n = x.size
    mean = float(x.mean())
    if n < 2:
        return {"n": int(n), "mean": mean, "sd": float("nan"),
                "lo": float("nan"), "hi": float("nan")}
    sd = float(x.std(ddof=1))
    half = stats.t.ppf(0.5 + conf / 2, n - 1) * sd / math.sqrt(n)
    return {"n": int(n), "mean": mean, "sd": sd, "lo": mean - half, "hi": mean + half}


def tost_one_sample(values, centre: float, margin: float):
    """Two one-sided t tests (TOST) for equivalence of the mean to `centre`
    within +/- `margin`. Returns the larger of the two one-sided p values;
    equivalence at level alpha is declared when it is below alpha.

    H0: |mean - centre| >= margin        H1: |mean - centre| < margin
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    if n < 2:
        return {"p_tost": float("nan"), "t_lower": float("nan"), "t_upper": float("nan")}
    se = x.std(ddof=1) / math.sqrt(n)
    if se == 0:
        inside = abs(x.mean() - centre) < margin
        return {"p_tost": 0.0 if inside else 1.0, "t_lower": float("inf"),
                "t_upper": float("-inf")}
    t_lower = (x.mean() - (centre - margin)) / se      # H0: mean <= centre - margin
    t_upper = (x.mean() - (centre + margin)) / se      # H0: mean >= centre + margin
    p_lower = 1 - stats.t.cdf(t_lower, n - 1)
    p_upper = stats.t.cdf(t_upper, n - 1)
    return {"p_tost": float(max(p_lower, p_upper)), "t_lower": float(t_lower),
            "t_upper": float(t_upper)}


def one_sample_t(values, centre: float):
    """Two-sided one-sample t test against `centre` plus Cohen's d."""
    x = np.asarray(values, dtype=float)
    if x.size < 2 or x.std(ddof=1) == 0:
        return {"t": float("nan"), "p": float("nan"), "cohen_d": float("nan")}
    res = stats.ttest_1samp(x, centre)
    return {"t": float(res.statistic), "p": float(res.pvalue),
            "cohen_d": float((x.mean() - centre) / x.std(ddof=1))}


def binom_tail_ge(n: int, p: float, k: int) -> float:
    """P[X >= k] for X ~ Binomial(n, p), exact."""
    return float(stats.binom.sf(k - 1, n, p))
