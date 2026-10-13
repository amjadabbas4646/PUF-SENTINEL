"""Bit-error models for the weak-PUF reading used in key reconstruction.

All three models are parameterised by the same mean bit-error rate p, so the
curves are comparable. None of them is calibrated to hardware. They are
sensitivity cases, chosen to show how much the answer depends on the
independence assumption.

iid
    Every bit flips independently with probability p.

heterogeneous
    Gaussian mismatch model. Cell i has a fixed mismatch m_i ~ N(0, 1) and
    every reading adds noise N(0, s^2). The enrolled bit is the noise-free
    sign of m_i, so cell i flips with probability Phi(-|m_i| / s). The scale
    s = tan(pi p) makes the mean over cells equal to p. Errors are independent
    across cells but concentrated in a minority of unstable cells. The cell
    mismatches are redrawn for every trial, i.e. every trial is a new device.

common_mode
    Every reading draws its own bit-error rate P from a Beta distribution with
    mean p and intra-reading correlation rho_cm (default 0.05), then flips
    bits independently with probability P. This models a disturbance shared
    by all cells of one reading, for example a supply or temperature
    excursion, and gives overdispersed error counts.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

MODELS = ("iid", "heterogeneous", "common_mode")


def sample_errors(model: str, p: float, n_bits: int, rng: np.random.Generator,
                  rho_cm: float = 0.05) -> np.ndarray:
    if model == "iid":
        return (rng.random(n_bits) < p).astype(np.uint8)
    if model == "heterogeneous":
        s = math.tan(math.pi * p)
        mism = rng.normal(size=n_bits)
        p_cell = stats.norm.cdf(-np.abs(mism) / s)
        return (rng.random(n_bits) < p_cell).astype(np.uint8)
    if model == "common_mode":
        a = p * (1 - rho_cm) / rho_cm
        b = (1 - p) * (1 - rho_cm) / rho_cm
        p_read = rng.beta(a, b)
        return (rng.random(n_bits) < p_read).astype(np.uint8)
    raise ValueError(f"unknown error model {model!r}")


def sample_errors_batch(model: str, p: float, n_bits: int, trials: int,
                        rng: np.random.Generator, rho_cm: float = 0.05) -> np.ndarray:
    """Vectorised version of sample_errors: (trials, n_bits) error patterns."""
    u = rng.random((trials, n_bits))
    if model == "iid":
        return (u < p).astype(np.uint8)
    if model == "heterogeneous":
        s = math.tan(math.pi * p)
        p_cell = stats.norm.cdf(-np.abs(rng.normal(size=(trials, n_bits))) / s)
        return (u < p_cell).astype(np.uint8)
    if model == "common_mode":
        a = p * (1 - rho_cm) / rho_cm
        b = (1 - p) * (1 - rho_cm) / rho_cm
        return (u < rng.beta(a, b, size=(trials, 1))).astype(np.uint8)
    raise ValueError(f"unknown error model {model!r}")


def analytic_failure_iid(family: str, n: int, t: int, bits_per_symbol: int,
                         blocks: int, p: float) -> float:
    """Exact key-recovery failure probability under the iid model for a
    bounded-distance decoder: 1 - P[every block has at most t symbol errors]."""
    p_sym = 1 - (1 - p) ** bits_per_symbol
    # 1 - (1 - block_fail)**blocks, computed without cancellation
    block_fail = stats.binom.sf(t, n, p_sym)
    return float(-math.expm1(blocks * math.log1p(-block_fail))) if block_fail < 1 else 1.0
