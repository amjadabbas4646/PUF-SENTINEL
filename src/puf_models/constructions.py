"""Generic members of published obfuscation classes for strong PUFs.

The benchmark controls are meant for any design of the form "PUF plus a
function around it". This module implements one generic member of each class
that recurs in the literature, so that the same controls can be applied to all
of them. They are re-implementations of the class in the simulation model of
this project. They are not the hardware or the parameters of any cited design.

Class  Where the function sits      Secret             Response r(c)
-----  ---------------------------  -----------------  ------------------------------
R      response side, keyed PRF     device key K       b(c) XOR M_K(c)   (mask.py)
H      response side, public hash   none               msb H(c, b(c_1), ..., b(c_m))
X      challenge side, XOR key      n-bit key kappa    b(c XOR kappa)
L      challenge side, linear map   invertible A       b(A c)  over GF(2)
P      challenge side, keyed PRF    device key K       b(F_K(c)),  F_K: n bits to n bits

In class H the sub-challenges c_1..c_m are a public expansion of c and the
response bit is a public hash of the challenge and of the m PUF bits. Nothing
in H is secret except the PUF itself.

Bit conventions are those of arbiter.py. All functions are deterministic given
their arguments. Noise enters only through the base PUF.
"""
from __future__ import annotations

import hashlib
import hmac
import struct

import numpy as np

CHAL_LABEL = b"PUF-SENTINEL/challenge/v1"
EXPAND_LABEL = b"PUF-SENTINEL/expand/v1"
HASH_LABEL = b"PUF-SENTINEL/hashout/v1"
CLASSES = ("X", "L", "P", "H")


def gf2_rank(matrix: np.ndarray) -> int:
    """Rank of a binary matrix over GF(2), by Gaussian elimination."""
    a = np.array(matrix, dtype=np.uint8) & 1
    rank = 0
    rows, cols = a.shape
    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if a[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != rank:
            a[[rank, pivot]] = a[[pivot, rank]]
        mask = a[:, col].astype(bool)
        mask[rank] = False
        a[mask] ^= a[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def random_invertible_gf2(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random invertible n by n binary matrix (rejection sampling)."""
    while True:
        a = rng.integers(0, 2, size=(n, n), dtype=np.uint8)
        if gf2_rank(a) == n:
            return a


def _bits_of_digests(digests: list[bytes], n_bits: int) -> np.ndarray:
    raw = np.frombuffer(b"".join(digests), dtype=np.uint8).reshape(len(digests), -1)
    return np.unpackbits(raw, axis=1)[:, :n_bits]


def challenge_xor(challenges: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Class X: the internal challenge is c XOR kappa."""
    return np.asarray(challenges, dtype=np.uint8) ^ np.asarray(kappa, dtype=np.uint8)


def challenge_linear(challenges: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Class L: the internal challenge is A c over GF(2)."""
    c = np.asarray(challenges, dtype=np.uint8)
    return ((c.astype(np.int32) @ np.asarray(a, dtype=np.int32).T) & 1).astype(np.uint8)


def challenge_prf(key: bytes, challenges: np.ndarray,
                  device_id: bytes = b"\x00" * 16) -> np.ndarray:
    """Class P: the internal challenge is the first n bits of
    HMAC-SHA256(K, label || 0x00 || device_id || uint16(n) || pack(c))."""
    c = np.asarray(challenges, dtype=np.uint8)
    n = c.shape[1]
    if n > 256:
        raise ValueError("at most 256 challenge bits")
    prefix = CHAL_LABEL + b"\x00" + device_id + struct.pack(">H", n)
    packed = np.packbits(c, axis=1)
    return _bits_of_digests([hmac.digest(key, prefix + row.tobytes(), "sha256")
                             for row in packed], n)


def expand_challenges(challenges: np.ndarray, m: int) -> np.ndarray:
    """Class H, public step 1: m sub-challenges per challenge, shape (N, m, n).
    Sub-challenge i is the first n bits of SHA-256(label || uint16(i) || pack(c))."""
    c = np.asarray(challenges, dtype=np.uint8)
    n = c.shape[1]
    if n > 256:
        raise ValueError("at most 256 challenge bits")
    packed = np.packbits(c, axis=1)
    out = np.empty((c.shape[0], m, n), dtype=np.uint8)
    for i in range(m):
        prefix = EXPAND_LABEL + struct.pack(">H", i)
        out[:, i, :] = _bits_of_digests(
            [hashlib.sha256(prefix + row.tobytes()).digest() for row in packed], n)
    return out


def hash_output(challenges: np.ndarray, puf_bits: np.ndarray) -> np.ndarray:
    """Class H, public step 2: one response bit per challenge, the first bit of
    SHA-256(label || pack(c) || pack(b(c_1), ..., b(c_m)))."""
    c = np.asarray(challenges, dtype=np.uint8)
    packed_c = np.packbits(c, axis=1)
    packed_b = np.packbits(np.asarray(puf_bits, dtype=np.uint8), axis=1)
    digests = [hashlib.sha256(HASH_LABEL + pc.tobytes() + pb.tobytes()).digest()
               for pc, pb in zip(packed_c, packed_b)]
    return _bits_of_digests(digests, 1)[:, 0]


class ObfuscatedPUF:
    """One generic member of class X, L, P or H around a base PUF.

    secret: kappa (X), matrix A (L), key bytes (P), None (H).
    `internal(c)` is what an adversary who knows the secret can compute: the
    challenges that actually reach the PUF. For H it is the public expansion.
    """

    def __init__(self, base, kind: str, secret=None, device_id: bytes = b"\x00" * 16,
                 m: int = 8):
        if kind not in CLASSES:
            raise ValueError(f"unknown class {kind!r}")
        if kind != "H" and secret is None:
            raise ValueError(f"class {kind} needs a secret")
        self.base, self.kind, self.secret = base, kind, secret
        self.device_id, self.m, self.n = device_id, m, base.n

    def internal(self, challenges: np.ndarray) -> np.ndarray:
        if self.kind == "X":
            return challenge_xor(challenges, self.secret)
        if self.kind == "L":
            return challenge_linear(challenges, self.secret)
        if self.kind == "P":
            return challenge_prf(self.secret, challenges, self.device_id)
        return expand_challenges(challenges, self.m)

    def eval(self, challenges: np.ndarray, noise_rng=None, internal=None) -> np.ndarray:
        """Response bits. `internal` may carry a cached result of internal(c),
        which saves the hashing when the same challenges are evaluated again."""
        inner = self.internal(challenges) if internal is None else internal
        if self.kind != "H":
            return self.base.eval(inner, noise_rng)
        n_rows = inner.shape[0]
        bits = self.base.eval(inner.reshape(n_rows * self.m, self.n), noise_rng)
        return hash_output(challenges, bits.reshape(n_rows, self.m))
