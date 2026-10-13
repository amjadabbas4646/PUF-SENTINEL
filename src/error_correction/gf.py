"""Arithmetic in GF(2^m) with log and antilog tables.

Polynomials are Python lists of field elements, lowest degree first.
"""
from __future__ import annotations

PRIMITIVE_POLY = {4: 0x13,    # x^4 + x + 1 (length 15, used by the unit tests only)
                  7: 0x89,    # x^7 + x^3 + 1
                  8: 0x11D}   # x^8 + x^4 + x^3 + x^2 + 1


class GF:
    def __init__(self, m: int, prim: int | None = None):
        self.m = m
        self.q = 1 << m
        self.order = self.q - 1
        prim = PRIMITIVE_POLY[m] if prim is None else prim
        self.exp = [0] * (2 * self.order)
        self.log = [0] * self.q
        x = 1
        for i in range(self.order):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & self.q:
                x ^= prim
        if x != 1:
            raise ValueError("polynomial is not primitive")
        for i in range(self.order, 2 * self.order):
            self.exp[i] = self.exp[i - self.order]

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]

    def div(self, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError("division by zero in GF(2^m)")
        if a == 0:
            return 0
        return self.exp[(self.log[a] - self.log[b]) % self.order]

    def inv(self, a: int) -> int:
        return self.div(1, a)

    def alpha_pow(self, e: int) -> int:
        return self.exp[e % self.order]

    # -- polynomials ---------------------------------------------------
    def poly_eval(self, p: list[int], x: int) -> int:
        y = 0
        for coef in reversed(p):
            y = self.mul(y, x) ^ coef
        return y

    def poly_mul(self, a: list[int], b: list[int]) -> list[int]:
        out = [0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            if ai:
                for j, bj in enumerate(b):
                    if bj:
                        out[i + j] ^= self.mul(ai, bj)
        return out

    def poly_mod(self, a: list[int], g: list[int]) -> list[int]:
        """Remainder of a(x) divided by the monic polynomial g(x)."""
        r = list(a)
        dg = len(g) - 1
        for i in range(len(r) - 1, dg - 1, -1):
            c = r[i]
            if c:
                for j in range(dg + 1):
                    r[i - dg + j] ^= self.mul(c, g[j])
        return r[:dg]

    def berlekamp_massey(self, synd: list[int]) -> tuple[list[int], int]:
        """Shortest LFSR for the syndrome sequence S_1..S_2t.
        Returns (error locator polynomial Lambda, its degree L)."""
        C, B = [1], [1]
        L, shift, b = 0, 1, 1
        for n, s in enumerate(synd):
            d = s
            for i in range(1, L + 1):
                if i < len(C) and C[i]:
                    d ^= self.mul(C[i], synd[n - i])
            if d == 0:
                shift += 1
                continue
            T = list(C)
            coef = self.div(d, b)
            need = len(B) + shift
            if len(C) < need:
                C = C + [0] * (need - len(C))
            for i, bv in enumerate(B):
                if bv:
                    C[i + shift] ^= self.mul(coef, bv)
            if 2 * L <= n:
                L, B, b, shift = n + 1 - L, T, d, 1
            else:
                shift += 1
        C = (C + [0] * (L + 1))[:L + 1]
        return C, L
