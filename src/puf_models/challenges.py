"""Challenge generation with explicit distributions and duplicate removal.

uniform
    Each bit is an independent fair coin.

markov(rho)
    First-order two-state Markov chain along the stages. The first bit is a
    fair coin. For j >= 2, with probability rho the bit copies bit j-1 and
    with probability 1 - rho it is drawn afresh as a fair coin. Hence
    P[c_j = c_{j-1}] = (1 + rho) / 2 and the correlation of adjacent bits is
    rho. The marginal distribution of every bit stays uniform. This is the
    process the original manuscript called "adversarially correlated". It is a
    non-uniform sampling distribution, not an adaptive chosen-challenge
    strategy, and it is named "correlated" here.
"""
from __future__ import annotations

import numpy as np


def _draw(n_rows: int, n: int, dist: str, rho: float,
          rng: np.random.Generator) -> np.ndarray:
    if dist == "uniform":
        return rng.integers(0, 2, size=(n_rows, n), dtype=np.uint8)
    if dist == "markov":
        c = np.empty((n_rows, n), dtype=np.uint8)
        c[:, 0] = rng.integers(0, 2, size=n_rows)
        fresh = rng.integers(0, 2, size=(n_rows, n), dtype=np.uint8)
        copy = rng.random((n_rows, n)) < rho
        for j in range(1, n):
            c[:, j] = np.where(copy[:, j], c[:, j - 1], fresh[:, j])
        return c
    raise ValueError(f"unknown challenge distribution {dist!r}")


def unique_challenges(n_rows: int, n: int, rng: np.random.Generator,
                      dist: str = "uniform", rho: float = 0.7) -> np.ndarray:
    """Return exactly n_rows pairwise distinct challenges, in draw order.
    Duplicates are dropped and replaced by further draws, so any split of the
    returned array into train, validation and test parts is disjoint."""
    seen: set[bytes] = set()
    rows = []
    while len(rows) < n_rows:
        batch = _draw(max(1024, int(1.1 * (n_rows - len(rows)))), n, dist, rho, rng)
        packed = np.packbits(batch, axis=1)
        for i in range(batch.shape[0]):
            key = packed[i].tobytes()
            if key not in seen:
                seen.add(key)
                rows.append(batch[i])
                if len(rows) == n_rows:
                    break
    return np.stack(rows)


def count_duplicates(challenges: np.ndarray) -> int:
    packed = np.packbits(np.asarray(challenges, dtype=np.uint8), axis=1)
    return int(packed.shape[0] - np.unique(packed, axis=0).shape[0])
