"""Statistical quality metrics for one-bit-per-challenge PUFs.

Data layout: R is a (D, N) array of bits, row d = device d, column j = the
response of every device to the same challenge c_j. All devices answer the
same N challenges.

uniformity_d   = mean_j R[d, j]                           (per device)
bit_aliasing_j = mean_d R[d, j]                           (per challenge)
uniqueness     = mean over the D(D-1)/2 device pairs of the fractional Hamming
                 distance between rows
entropy        = marginal Shannon entropy and most-common-value min-entropy of
                 the response bit of one device, estimated over challenges.
                 Both describe the bit frequency only. They are upper bounds
                 on the entropy per response bit and say nothing about joint
                 entropy across challenges or about predictability by a model.
flip rate      = P[r(c) != r(c with bit i flipped)], per challenge-bit
                 position i (strict avalanche criterion asks for 0.5 at every
                 position)

The pairwise Hamming distances are not independent, because every device
appears in D - 1 pairs. Intervals for uniqueness therefore resample devices,
not pairs.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy import stats


def uniformity(R: np.ndarray) -> np.ndarray:
    return np.asarray(R, dtype=float).mean(axis=1)


def bit_aliasing(R: np.ndarray) -> np.ndarray:
    return np.asarray(R, dtype=float).mean(axis=0)


def pairwise_hd(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.uint8)
    return np.array([np.mean(R[i] != R[j])
                     for i, j in combinations(range(R.shape[0]), 2)])


def uniqueness(R: np.ndarray) -> float:
    return float(pairwise_hd(R).mean())


def uniqueness_device_bootstrap(R: np.ndarray, rng: np.random.Generator,
                                n_boot: int = 2000, conf: float = 0.95):
    """Percentile bootstrap interval for uniqueness that resamples devices.
    Pairs formed by two copies of the same device are excluded."""
    R = np.asarray(R, dtype=np.uint8)
    D = R.shape[0]
    full = np.zeros((D, D))
    for i, j in combinations(range(D), 2):
        full[i, j] = full[j, i] = np.mean(R[i] != R[j])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, D, size=D)
        sub = full[np.ix_(idx, idx)]
        mask = idx[:, None] != idx[None, :]
        if mask.any():
            vals.append(sub[mask].mean())
    lo, hi = np.quantile(vals, [(1 - conf) / 2, (1 + conf) / 2])
    return float(lo), float(hi)


def shannon_entropy_bit(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return float(-p * math.log2(p) - (1 - p) * math.log2(1 - p))


def mcv_min_entropy(bits: np.ndarray) -> float:
    """Most-common-value min-entropy estimate per bit in the manner of
    NIST SP 800-90B section 6.3.1: upper bound on the probability of the most
    common value with z = 2.576, then -log2."""
    bits = np.asarray(bits)
    n = bits.size
    p_hat = max(bits.mean(), 1 - bits.mean())
    z = 2.576
    p_u = min(1.0, p_hat + z * math.sqrt(p_hat * (1 - p_hat) / (n - 1)))
    return float(-math.log2(p_u))


def flip_rate_profile(puf, challenges: np.ndarray) -> np.ndarray:
    """Noise-free probability that the response flips when challenge bit i is
    flipped, for every position i. Shape (n,)."""
    base = puf.eval(challenges)
    out = np.empty(challenges.shape[1])
    for i in range(challenges.shape[1]):
        c2 = challenges.copy()
        c2[:, i] ^= 1
        out[i] = np.mean(puf.eval(c2) != base)
    return out


# -- two tests of NIST SP 800-22 Rev. 1a ---------------------------------
def nist_frequency_p(bits: np.ndarray) -> float:
    """Frequency (monobit) test, SP 800-22 Rev. 1a section 2.1."""
    bits = np.asarray(bits, dtype=np.int64)
    n = bits.size
    s_obs = abs(int(2 * bits.sum() - n)) / math.sqrt(n)
    return math.erfc(s_obs / math.sqrt(2))


def nist_runs_p(bits: np.ndarray) -> float:
    """Runs test, SP 800-22 Rev. 1a section 2.3. Returns 0.0 when the
    frequency prerequisite |pi - 1/2| < 2 / sqrt(n) is not met."""
    bits = np.asarray(bits, dtype=np.int64)
    n = bits.size
    pi = bits.mean()
    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return 0.0
    v_obs = 1 + int(np.sum(bits[:-1] != bits[1:]))
    num = abs(v_obs - 2 * n * pi * (1 - pi))
    den = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    return math.erfc(num / den)


def nist_summary(p_values, alpha: float = 0.01) -> dict:
    """Pass proportion with the SP 800-22 section 4.2.1 acceptance band and a
    Kolmogorov-Smirnov test of the p values against the uniform distribution."""
    p = np.asarray(p_values, dtype=float)
    s = p.size
    prop = float(np.mean(p >= alpha))
    half = 3 * math.sqrt(alpha * (1 - alpha) / s)
    ks = stats.kstest(p, "uniform")
    return {"sequences": int(s), "pass_proportion": prop,
            "accept_lo": 1 - alpha - half, "accept_hi": min(1.0, 1 - alpha + half),
            "proportion_ok": bool(prop >= 1 - alpha - half),
            "ks_p_uniform": float(ks.pvalue)}
