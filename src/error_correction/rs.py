"""Shortened Reed-Solomon codes over GF(2^8).

RS(n, k) with n <= 255 symbols of 8 bits, generator polynomial with roots
alpha^1 .. alpha^(n-k), minimum distance d = n - k + 1, and guaranteed
correction of t = floor((n - k) / 2) symbol errors. The decoder is an
errors-only bounded-distance decoder: syndromes, Berlekamp-Massey, Chien
search, Forney's formula.

Codeword layout (lowest degree first): n - k parity symbols, then k message
symbols.
"""
from __future__ import annotations

import numpy as np

from .gf import GF


class RS:
    def __init__(self, n: int, k: int):
        if not 0 < k < n <= 255:
            raise ValueError("need 0 < k < n <= 255")
        self.gf = GF(8)
        self.n, self.k = n, k
        self.nroots = n - k
        self.t = self.nroots // 2
        self.d_min = n - k + 1
        g = [1]
        for j in range(1, self.nroots + 1):
            g = self.gf.poly_mul(g, [self.gf.alpha_pow(j), 1])
        self.g = g

    def encode(self, msg: np.ndarray) -> np.ndarray:
        msg = np.asarray(msg, dtype=np.uint8)
        if msg.shape != (self.k,):
            raise ValueError(f"message must have {self.k} symbols")
        shifted = [0] * self.nroots + [int(s) for s in msg]
        parity = self.gf.poly_mod(shifted, self.g)
        return np.array(parity + [int(s) for s in msg], dtype=np.uint8)

    def syndromes(self, word) -> list[int]:
        w = [int(s) for s in word]
        return [self.gf.poly_eval(w, self.gf.alpha_pow(j))
                for j in range(1, self.nroots + 1)]

    def decode(self, word: np.ndarray) -> tuple[np.ndarray, bool]:
        gf = self.gf
        word = np.asarray(word, dtype=np.uint8).copy()
        synd = self.syndromes(word)
        if not any(synd):
            return word, True
        lam, L = gf.berlekamp_massey(synd)
        if L > self.t:
            return word, False
        positions = [i for i in range(self.n)
                     if gf.poly_eval(lam, gf.alpha_pow(-i)) == 0]
        if len(positions) != L:
            return word, False            # roots outside the shortened length
        omega = gf.poly_mul(synd, lam)[:self.nroots]
        lam_deriv = [lam[i] if i % 2 == 1 else 0 for i in range(1, len(lam))]
        for i in positions:
            x_inv = gf.alpha_pow(-i)
            den = gf.poly_eval(lam_deriv, x_inv)
            if den == 0:
                return word, False
            word[i] ^= gf.div(gf.poly_eval(omega, x_inv), den)
        if any(self.syndromes(word)):
            return word, False
        return word, True

    def extract(self, codeword: np.ndarray) -> np.ndarray:
        return np.asarray(codeword, dtype=np.uint8)[self.nroots:]
