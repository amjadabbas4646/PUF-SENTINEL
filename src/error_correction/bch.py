"""Binary primitive narrow-sense BCH codes of length n = 2^m - 1.

The generator polynomial is the least common multiple of the minimal
polynomials of alpha^1 .. alpha^(2t). The dimension k = n - deg g follows from
the construction and is not typed in by hand. For n = 127 this gives
    t = 7  -> (127, 78),  designed distance 15
    t = 10 -> (127, 64),  designed distance 21
    t = 15 -> (127, 36),  designed distance 31
The decoder is a bounded-distance decoder: syndromes, Berlekamp-Massey, Chien
search. It corrects every pattern of at most t bit errors. For heavier
patterns it either reports failure or returns a different codeword.

Codeword layout (lowest degree first): parity bits c_0..c_{n-k-1}, then the k
message bits.
"""
from __future__ import annotations

import numpy as np

from .gf import GF


class BCH:
    def __init__(self, m: int, t: int):
        self.gf = GF(m)
        self.n = (1 << m) - 1
        self.t = t
        self.g = self._generator()
        self.k = self.n - (len(self.g) - 1)
        if self.k <= 0:
            raise ValueError("t too large for this length")
        self.d_design = 2 * t + 1
        # alpha^(j*i) lookup for syndrome computation: rows j = 1..2t
        j = np.arange(1, 2 * t + 1)[:, None]
        i = np.arange(self.n)[None, :]
        self._synd_tab = np.array(self.gf.exp, dtype=np.int64)[(j * i) % self.n]

    def _generator(self) -> list[int]:
        gf, n = self.gf, self.n
        roots: set[int] = set()
        for j in range(1, 2 * self.t + 1):
            e = j % n
            while e not in roots:           # cyclotomic coset of j
                roots.add(e)
                e = (2 * e) % n
        g = [1]
        for e in sorted(roots):
            g = gf.poly_mul(g, [gf.alpha_pow(e), 1])
        if any(c not in (0, 1) for c in g):
            raise AssertionError("generator polynomial is not binary")
        return g

    def encode(self, msg: np.ndarray) -> np.ndarray:
        msg = np.asarray(msg, dtype=np.uint8)
        if msg.shape != (self.k,):
            raise ValueError(f"message must have {self.k} bits")
        r = self.n - self.k
        shifted = [0] * r + [int(b) for b in msg]
        parity = self.gf.poly_mod(shifted, self.g)
        return np.array(parity + [int(b) for b in msg], dtype=np.uint8)

    def syndromes(self, word: np.ndarray) -> list[int]:
        pos = np.flatnonzero(word)
        if pos.size == 0:
            return [0] * (2 * self.t)
        return [int(v) for v in np.bitwise_xor.reduce(self._synd_tab[:, pos], axis=1)]

    def decode(self, word: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return (codeword estimate, success flag). success = False means the
        decoder detected an uncorrectable pattern."""
        word = np.asarray(word, dtype=np.uint8).copy()
        synd = self.syndromes(word)
        if not any(synd):
            return word, True
        lam, L = self.gf.berlekamp_massey(synd)
        if L > self.t:
            return word, False
        positions = [i for i in range(self.n)
                     if self.gf.poly_eval(lam, self.gf.alpha_pow(-i)) == 0]
        if len(positions) != L:
            return word, False
        word[positions] ^= 1
        if any(self.syndromes(word)):
            return word, False
        return word, True

    def extract(self, codeword: np.ndarray) -> np.ndarray:
        return np.asarray(codeword, dtype=np.uint8)[self.n - self.k:]
