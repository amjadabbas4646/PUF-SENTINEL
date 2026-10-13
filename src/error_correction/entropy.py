"""Conditional min-entropy of a code-offset secret given its public helper data.

Setting
-------
One block of the code-offset construction of fuzzy_extractor.py: the weak-PUF
reading W has N independent bits with P[bit = 1] = p, the message S is uniform
on k_bits bits, C is a binary linear [N, k_bits] code (a Reed-Solomon code over
GF(2^8) is used through its binary image) and the helper data are
H = W XOR C(S).

Exact expression
----------------
The average conditional min-entropy (Dodis et al.) is

    H~_inf(S | H) = -log2  sum_h  max_s  P[S = s, H = h]
                  = -log2  sum over cosets of C of  max_{e in coset} P_W(e).

Proof sketch: P[S = s, H = h] = 2^-k_bits P_W(h XOR C(s)). For a fixed h the
words h XOR C(s) run through the coset h + C, and each coset is met by 2^k_bits
values of h, which cancels the factor 2^-k_bits.

For p <= 1/2 the most probable word of a coset is its coset leader (a word of
minimum weight), so with a_i the number of cosets whose leader has weight i,

    sum = sum_i a_i p^i (1 - p)^(N - i).

The coset leader weight distribution a_i of the codes used here is not known in
closed form, but two facts bound it:
    (1) a_i = binom(N, i) for i <= t, because a code with minimum distance at
        least 2t + 1 has every word of weight at most t as a unique leader;
    (2) a_i <= binom(N, i) for all i and sum_i a_i = 2^(N - k_bits).
Since p^i (1 - p)^(N - i) decreases in i for p <= 1/2, the sum is largest when
the remaining cosets take the smallest weights allowed by (2) and smallest when
they contribute nothing. This gives a lower and an upper bound on the entropy.
The same counting argument is used for biased PUFs by Maes et al. (CHES 2015)
and Delvaux et al. (CHES 2016). For p = 1/2 both bounds equal k_bits.

For the Reed-Solomon codes the binary image has minimum distance at least the
symbol distance, so fact (1) holds with the symbol correction radius t.

Assumptions (stated in the manuscript)
    * the bits of W are independent with a common, known bias p <= 1/2;
    * blocks are independent, so entropies add over blocks;
    * the helper data are read by the adversary and are not modified.
Spatial correlation of the weak-PUF cells is not covered.
"""
from __future__ import annotations

import math
from itertools import product


def _log2_binom(n: int, i: int) -> float:
    return (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)) / math.log(2)


def _log2_sum(terms: list[float]) -> float:
    """log2 of a sum of numbers given by their base-2 logarithms."""
    if not terms:
        return float("-inf")
    top = max(terms)
    return top + math.log2(sum(2.0 ** (x - top) for x in terms))


def block_entropy_bounds(n_bits: int, k_bits: int, t: int, p: float) -> tuple[float, float]:
    """(lower, upper) bound in bits on H~_inf(S | H) for one block.

    n_bits, k_bits: length and dimension of the binary code.
    t: correction radius guaranteed by the minimum distance.
    p: P[weak-PUF bit = 1], 0 < p <= 1/2 (by symmetry of the bound, a bias
       towards one is entered as 1 - p for codes that contain the all-ones
       word, which holds for the primitive BCH codes used here).
    """
    if not 0.0 < p <= 0.5:
        raise ValueError("p must lie in (0, 1/2]")
    if not 0 <= t <= n_bits or not 0 < k_bits <= n_bits:
        raise ValueError("inconsistent code parameters")
    lp, lq = math.log2(p), math.log2(1.0 - p)
    cosets = 2 ** (n_bits - k_bits)

    def term(i: int, log2_count: float) -> float:
        return log2_count + i * lp + (n_bits - i) * lq

    sure = [term(i, _log2_binom(n_bits, i)) for i in range(t + 1)]
    used = sum(math.comb(n_bits, i) for i in range(t + 1))
    if used > cosets:
        raise ValueError("more correctable patterns than cosets: parameters are impossible")
    upper_terms = list(sure)                 # largest possible sum -> entropy lower bound
    remaining, i = cosets - used, t + 1
    while remaining > 0 and i <= n_bits:
        take = min(remaining, math.comb(n_bits, i))
        upper_terms.append(term(i, math.log2(take)))
        remaining -= take
        i += 1
    lower = -_log2_sum(upper_terms)
    upper = -_log2_sum(sure)
    return max(0.0, lower), min(float(k_bits), upper)


def naive_entropy_bound(n_bits: int, k_bits: int, p: float) -> float:
    """The generic bound H_inf(W) - (N - k_bits): min-entropy of the reading
    minus the number of redundancy bits revealed by the helper data."""
    return max(0.0, -n_bits * math.log2(max(p, 1.0 - p)) - (n_bits - k_bits))


def exact_block_entropy(generator_rows: list[list[int]], p: float) -> float:
    """Exact H~_inf(S | H) by enumeration of all cosets. Only for tiny codes;
    used by the unit tests to check the bounds."""
    k = len(generator_rows)
    n = len(generator_rows[0])
    if n > 20:
        raise ValueError("enumeration is limited to n <= 20")
    rows = [int("".join(map(str, r)), 2) for r in generator_rows]
    code = []
    for coeffs in product((0, 1), repeat=k):
        w = 0
        for c, r in zip(coeffs, rows):
            if c:
                w ^= r
        code.append(w)
    best_weight = {}                          # coset representative -> minimum weight
    seen = bytearray(1 << n)
    total = 0.0
    for v in range(1 << n):
        if seen[v]:
            continue
        wmin = n + 1
        for cw in code:
            e = v ^ cw
            seen[e] = 1
            wmin = min(wmin, bin(e).count("1"))
        best_weight[v] = wmin
        q = min(p, 1.0 - p)
        total += q ** wmin * (1.0 - q) ** (n - wmin)
    return -math.log2(total)


def key_entropy_bounds(n: int, k: int, t: int, bits_per_symbol: int, blocks: int,
                       p: float) -> dict:
    """Bounds for a whole key: `blocks` independent blocks of an (n, k) code
    over symbols of `bits_per_symbol` bits."""
    nb, kb = n * bits_per_symbol, k * bits_per_symbol
    lo, hi = block_entropy_bounds(nb, kb, t, p)
    return {"bias_p": p, "secret_bits": blocks * kb,
            "entropy_lower": blocks * lo, "entropy_upper": blocks * hi,
            "naive_lower": blocks * naive_entropy_bound(nb, kb, p),
            "reading_min_entropy": -blocks * nb * math.log2(max(p, 1.0 - p)),
            "helper_redundancy_bits": blocks * (nb - kb)}
