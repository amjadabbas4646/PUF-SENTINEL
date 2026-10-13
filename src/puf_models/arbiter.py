"""Additive delay model of the arbiter PUF and its XOR composition.

Bit conventions (used everywhere in the project)
------------------------------------------------
* A challenge is a vector c in {0,1}^n, stage 1 first.
* The parity feature vector phi in {-1,+1}^(n+1) is
      phi_i = prod_{j=i..n} (1 - 2 c_j)   for i = 1..n,     phi_{n+1} = 1.
* One arbiter chain with weight vector w in R^(n+1) has delay difference
      Delta(c) = <w, phi(c)> + eta,   eta ~ N(0, sigma_noise^2) per evaluation,
  and outputs the bit 1 when Delta(c) > 0 and 0 otherwise.
* A k-XOR arbiter PUF evaluates k independent chains on the same challenge and
  outputs the XOR of their bits.

Simulation assumptions (stated, not measured)
---------------------------------------------
* Weights are i.i.d. N(0, 1), including the bias weight w_{n+1}.
* Evaluation noise is i.i.d. Gaussian per chain and per evaluation, with
  sigma_noise = noisiness * sqrt(n + 1), i.e. `noisiness` is the ratio of the
  noise standard deviation to the standard deviation of the noise-free delay
  difference. For one chain the probability that a noisy evaluation differs
  from the noise-free bit is arctan(noisiness) / pi.
* No dependence on temperature, voltage or ageing is modelled.
"""
from __future__ import annotations

import math

import numpy as np


def parity_features(challenges: np.ndarray) -> np.ndarray:
    """Map (N, n) challenges in {0,1} to (N, n+1) parity features in {-1,+1}."""
    c = np.asarray(challenges)
    if c.ndim != 2:
        raise ValueError("challenges must be a 2-D array (N, n)")
    d = (1 - 2 * c.astype(np.int8))
    phi = np.cumprod(d[:, ::-1], axis=1, dtype=np.int8)[:, ::-1]
    out = np.empty((c.shape[0], c.shape[1] + 1), dtype=np.float32)
    out[:, :-1] = phi
    out[:, -1] = 1.0
    return out


def expected_single_chain_ber(noisiness: float) -> float:
    """P[noisy bit != noise-free bit] for one chain under the model above."""
    return math.atan(noisiness) / math.pi


class XorArbiterPUF:
    """k-XOR arbiter PUF (k = 1 is the plain arbiter PUF)."""

    def __init__(self, n: int, k: int, rng: np.random.Generator,
                 noisiness: float = 0.0):
        if n < 1 or k < 1:
            raise ValueError("n and k must be positive")
        self.n, self.k, self.noisiness = n, k, float(noisiness)
        self.weights = rng.normal(0.0, 1.0, size=(k, n + 1))

    # -- internals -----------------------------------------------------
    def delays(self, challenges: np.ndarray) -> np.ndarray:
        """Noise-free delay differences, shape (N, k)."""
        return parity_features(challenges).astype(np.float64) @ self.weights.T

    def chain_bits(self, challenges: np.ndarray,
                   noise_rng: np.random.Generator | None = None) -> np.ndarray:
        """Per-chain bits, shape (N, k). Noise is added only when a generator
        is passed and the device noisiness is non-zero."""
        d = self.delays(challenges)
        if noise_rng is not None and self.noisiness > 0:
            sigma = self.noisiness * math.sqrt(self.n + 1)
            d = d + noise_rng.normal(0.0, sigma, size=d.shape)
        return (d > 0).astype(np.uint8)

    # -- public --------------------------------------------------------
    def eval(self, challenges: np.ndarray,
             noise_rng: np.random.Generator | None = None) -> np.ndarray:
        """Response bits, shape (N,), dtype uint8."""
        return np.bitwise_xor.reduce(self.chain_bits(challenges, noise_rng), axis=1)


class ZeroPUF:
    """Degenerate base primitive that always answers 0. Used only to build the
    'mask only' control, where the response is the keyed mask and nothing else."""

    def __init__(self, n: int):
        self.n, self.k, self.noisiness = n, 0, 0.0

    def eval(self, challenges: np.ndarray, noise_rng=None) -> np.ndarray:
        return np.zeros(np.asarray(challenges).shape[0], dtype=np.uint8)
