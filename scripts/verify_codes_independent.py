"""Independent verification of the error-correction layer (E7).

Nothing in sections 1 to 3 imports the project's field or code classes. The
project implementation is then compared against these independent results.

  1  BCH generator polynomials of length 127, rebuilt with integer bit masks
     over GF(2) from cyclotomic cosets, in two different field representations
     (x^7 + x^3 + 1 and x^7 + x + 1). Checks: degree, dimension, divisibility
     of x^127 + 1, the 2t consecutive roots, weight of g at least 2t + 1.
  2  Reed-Solomon generator polynomials over GF(2^8) (x^8+x^4+x^3+x^2+1) with a
     separately written multiplication routine. Checks: roots alpha^1..alpha^(n-k).
  3  Key-reconstruction failure probability under independent bit errors in
     exact rational arithmetic (fractions.Fraction, math.comb), and the largest
     bit error rate with failure probability at most 1e-6 by bisection.
  4  Comparison with the project: generator polynomials, dimensions, the
     floating-point failure formula and the stored design thresholds.
  5  Decoder behaviour: every error pattern of weight 1 and 2, random patterns
     of every weight up to t, and random patterns beyond t, classified as
     corrected, detected failure or miscorrection.
  6  Zero-failure Monte Carlo points: exact one-sided 95 % upper bounds.

Usage:  python scripts/verify_codes_independent.py
Output: experiments/results/code_verification.json
"""
from __future__ import annotations

import csv
import json
import sys
from fractions import Fraction
from itertools import combinations
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "results"
report: dict = {"checks": []}


def check(name: str, ok: bool, detail: str = "") -> None:
    report["checks"].append({"check": name, "passed": bool(ok), "detail": detail})
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))


# ---------------------------------------------------------------- GF(2)[x]
def pdeg(a: int) -> int:
    return a.bit_length() - 1


def pmul2(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        b >>= 1
    return r


def pmod2(a: int, m: int) -> int:
    dm = pdeg(m)
    while a and pdeg(a) >= dm:
        a ^= m << (pdeg(a) - dm)
    return a


# ---------------------------------------------------------------- GF(2^m)
def field_mul(a: int, b: int, prim: int, m: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a >> m:
            a ^= prim
    return r


def field_pow(a: int, e: int, prim: int, m: int) -> int:
    r = 1
    for _ in range(e):
        r = field_mul(r, a, prim, m)
    return r


def eval_binary_poly(g: int, x: int, prim: int, m: int) -> int:
    """Evaluate a GF(2)[x] polynomial (bit mask) at a field element."""
    y = 0
    for i in range(pdeg(g), -1, -1):
        y = field_mul(y, x, prim, m) ^ ((g >> i) & 1)
    return y


def minimal_polynomial(e: int, prim: int, m: int) -> int:
    """Minimal polynomial over GF(2) of alpha^e, as a bit mask."""
    n = (1 << m) - 1
    coset, x = [], e % n
    while x not in coset:
        coset.append(x)
        x = (2 * x) % n
    poly = [1]                                    # coefficients in GF(2^m), low first
    for c in coset:
        root = field_pow(2, c, prim, m)
        nxt = [0] * (len(poly) + 1)
        for i, a in enumerate(poly):
            nxt[i] ^= field_mul(a, root, prim, m)
            nxt[i + 1] ^= a
        poly = nxt
    assert all(a in (0, 1) for a in poly), "minimal polynomial is not binary"
    return sum(a << i for i, a in enumerate(poly))


def bch_generator(t: int, prim: int, m: int = 7) -> int:
    g, seen = 1, set()
    for j in range(1, 2 * t + 1):
        mp = minimal_polynomial(j, prim, m)
        if mp not in seen:
            seen.add(mp)
            g = pmul2(g, mp)
    return g


# ---------------------------------------------------------------- 1. BCH
EXPECTED_K = {7: 78, 10: 64, 15: 36}
bch_info = {}
for prim in (0x89, 0x83):
    for t, k_expected in EXPECTED_K.items():
        g = bch_generator(t, prim)
        k = 127 - pdeg(g)
        roots_ok = all(eval_binary_poly(g, field_pow(2, j, prim, 7), prim, 7) == 0
                       for j in range(1, 2 * t + 1))
        divides = pmod2((1 << 127) | 1, g) == 0
        weight = bin(g).count("1")
        tag = f"BCH t={t}, field 0x{prim:X}"
        check(f"{tag}: dimension {k_expected}", k == k_expected, f"deg g = {pdeg(g)}")
        check(f"{tag}: alpha^1..alpha^{2 * t} are roots of g", roots_ok)
        check(f"{tag}: g divides x^127 + 1", divides)
        check(f"{tag}: weight of g at least 2t + 1", weight >= 2 * t + 1, f"weight {weight}")
        bch_info[f"t{t}_0x{prim:X}"] = {"k": k, "deg_g": pdeg(g), "g_hex": hex(g), "weight_g": weight}
report["bch"] = bch_info


# ---------------------------------------------------------------- 2. RS
def rs_generator(nroots: int, prim: int = 0x11D) -> list[int]:
    g = [1]
    for j in range(1, nroots + 1):
        root = field_pow(2, j, prim, 8)
        nxt = [0] * (len(g) + 1)
        for i, a in enumerate(g):
            nxt[i] ^= field_mul(a, root, prim, 8)
            nxt[i + 1] ^= a
        g = nxt
    return g


def rs_eval(g: list[int], x: int, prim: int = 0x11D) -> int:
    y = 0
    for a in reversed(g):
        y = field_mul(y, x, prim, 8) ^ a
    return y


RS_CODES = {(24, 16): 4, (32, 16): 8, (48, 16): 16}
rs_gens = {}
for (n, k), t in RS_CODES.items():
    g = rs_generator(n - k)
    rs_gens[(n, k)] = g
    ok = all(rs_eval(g, field_pow(2, j, 0x11D, 8)) == 0 for j in range(1, n - k + 1))
    check(f"RS({n},{k}): generator has roots alpha^1..alpha^{n - k}, degree {n - k}",
          ok and len(g) == n - k + 1 and g[-1] == 1)
    check(f"RS({n},{k}): t = floor((n-k)/2) = {t}, minimum distance {n - k + 1}",
          (n - k) // 2 == t)


# ---------------------------------------------------------------- 3. exact failure
def exact_failure(n: int, t: int, bits_per_symbol: int, blocks: int, p: Fraction) -> Fraction:
    ps = 1 - (1 - p) ** bits_per_symbol
    ok = sum(Fraction(comb(n, i)) * ps ** i * (1 - ps) ** (n - i) for i in range(t + 1))
    return 1 - ok ** blocks


CONFIGS = {"BCH(127,78,7) x2": (127, 7, 1, 2), "BCH(127,64,10) x2": (127, 10, 1, 2),
           "BCH(127,36,15) x4": (127, 15, 1, 4), "RS(24,16,4)": (24, 4, 8, 1),
           "RS(32,16,8)": (32, 8, 8, 1), "RS(48,16,16)": (48, 16, 8, 1)}


def threshold(n, t, u, blocks, target=Fraction(1, 10 ** 6)) -> Fraction:
    lo, hi = Fraction(0), Fraction(1, 2)
    for _ in range(40):                           # resolution 2^-41
        mid = (lo + hi) / 2
        if exact_failure(n, t, u, blocks, mid.limit_denominator(10 ** 9)) <= target:
            lo = mid
        else:
            hi = mid
    return lo


thresholds = {name: threshold(*cfg) for name, cfg in CONFIGS.items()}
report["threshold_ber_for_1e-6_exact"] = {k: float(v) for k, v in thresholds.items()}
for name, v in thresholds.items():
    print(f"      exact threshold  {name:20s} {100 * float(v):.4f} %")
best = thresholds["BCH(127,36,15) x4"]
f_at = exact_failure(127, 15, 1, 4, Fraction(261, 10000))
f_above = exact_failure(127, 15, 1, 4, Fraction(262, 10000))
check("exact arithmetic: BCH(127,36,15) x4 fails with probability <= 1e-6 at BER 2.61 %",
      f_at <= Fraction(1, 10 ** 6), f"{float(f_at):.3e}")
check("exact arithmetic: and above 1e-6 at BER 2.62 %, the value printed in the first revision",
      f_above > Fraction(1, 10 ** 6), f"{float(f_above):.3e}")
report["bch_127_36_15_x4"] = {"failure_at_2.61_percent": float(f_at),
                              "failure_at_2.62_percent": float(f_above),
                              "threshold_percent": 100 * float(best)}

# ---------------------------------------------------------------- 4. project
sys.path.insert(0, str(ROOT / "src"))
import numpy as np                                                    # noqa: E402

from common.seeding import rng_for                                    # noqa: E402
from error_correction.bch import BCH                                  # noqa: E402
from error_correction.error_models import analytic_failure_iid        # noqa: E402
from error_correction.rs import RS                                    # noqa: E402

for t, k in EXPECTED_K.items():
    code = BCH(7, t)
    mask = sum(int(c) << i for i, c in enumerate(code.g))
    check(f"project BCH t={t}: generator equals the independent one (field 0x89)",
          mask == bch_generator(t, 0x89) and code.k == k)
for (n, k), g in rs_gens.items():
    check(f"project RS({n},{k}): generator equals the independent one", RS(n, k).g == g)

design = json.loads((OUT / "full" / "ecc_design.json").read_text(encoding="utf-8"))
worst_rel = 0.0
for name, (n, t, u, blocks) in CONFIGS.items():
    fam = "BCH" if u == 1 else "RS"
    for p in (Fraction(1, 100), Fraction(3, 100), Fraction(5, 100), Fraction(1, 10), Fraction(1, 5)):
        exact = float(exact_failure(n, t, u, blocks, p))
        flt = analytic_failure_iid(fam, n, t, u, blocks, float(p))
        if exact > 1e-300:
            worst_rel = max(worst_rel, abs(flt - exact) / exact)
    check(f"stored design threshold of {name} equals the exact one",
          abs(design[name]["max_ber_fail_1e-6"] - float(thresholds[name])) < 1e-7,
          f"stored {100 * design[name]['max_ber_fail_1e-6']:.4f} %, exact {100 * float(thresholds[name]):.4f} %")
check("floating-point failure formula agrees with exact arithmetic (relative error < 1e-6)",
      worst_rel < 1e-6, f"largest relative error {worst_rel:.2e}")

# ---------------------------------------------------------------- 5. decoders
rng = rng_for("verify", "decoders")
decoder_stats = {}
for t in (7, 10, 15):
    code = BCH(7, t)
    cw = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
    ok = True
    for pos in list(combinations(range(127), 1)) + list(combinations(range(127), 2)):
        w = cw.copy()
        w[list(pos)] ^= 1
        out, flag = code.decode(w)
        ok &= flag and np.array_equal(out, cw)
    check(f"BCH t={t}: all {127 + comb(127, 2)} patterns of weight 1 and 2 corrected", ok)
    ok_all, stat = True, {"detected": 0, "miscorrected": 0}
    for wt in range(0, t + 1):
        for _ in range(60):
            cw = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
            w = cw.copy()
            w[rng.choice(127, size=wt, replace=False)] ^= 1
            out, flag = code.decode(w)
            ok_all &= flag and np.array_equal(out, cw)
    check(f"BCH t={t}: 60 random patterns of every weight 0..{t} corrected", ok_all)
    wrong = 0
    for _ in range(2000):
        cw = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
        w = cw.copy()
        w[rng.choice(127, size=int(rng.integers(t + 1, t + 8)), replace=False)] ^= 1
        out, flag = code.decode(w)
        if flag and np.array_equal(out, cw):
            wrong += 1
        elif flag:
            stat["miscorrected"] += 1
            assert not any(code.syndromes(out)), "decoder returned a non-codeword"
        else:
            stat["detected"] += 1
    check(f"BCH t={t}: none of 2000 patterns beyond t decoded to the sent codeword", wrong == 0,
          f"detected {stat['detected']}, miscorrected to another codeword {stat['miscorrected']}")
    decoder_stats[f"BCH_t{t}"] = stat
for (n, k), t in RS_CODES.items():
    code = RS(n, k)
    ok_all, stat, wrong = True, {"detected": 0, "miscorrected": 0}, 0
    for wt in range(0, t + 1):
        for _ in range(60):
            cw = code.encode(rng.integers(0, 256, size=k, dtype=np.uint8))
            w = cw.copy()
            pos = rng.choice(n, size=wt, replace=False)
            w[pos] ^= rng.integers(1, 256, size=wt, dtype=np.uint8)
            out, flag = code.decode(w)
            ok_all &= flag and np.array_equal(out, cw)
    check(f"RS({n},{k}): 60 random patterns of every weight 0..{t} corrected", ok_all)
    for _ in range(2000):
        cw = code.encode(rng.integers(0, 256, size=k, dtype=np.uint8))
        w = cw.copy()
        wt = int(rng.integers(t + 1, min(n, t + 7) + 1))
        pos = rng.choice(n, size=wt, replace=False)
        w[pos] ^= rng.integers(1, 256, size=wt, dtype=np.uint8)
        out, flag = code.decode(w)
        if flag and np.array_equal(out, cw):
            wrong += 1
        elif flag:
            stat["miscorrected"] += 1
        else:
            stat["detected"] += 1
    check(f"RS({n},{k}): none of 2000 patterns beyond t decoded to the sent codeword", wrong == 0,
          f"detected {stat['detected']}, miscorrected to another codeword {stat['miscorrected']}")
    decoder_stats[f"RS_{n}_{k}"] = stat
report["decoder_beyond_radius"] = decoder_stats

# ---------------------------------------------------------------- 6. zero-failure bounds
zero = []
with open(OUT / "full" / "ecc_raw.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if int(r["weight_failures"]) == 0:
            n_tr = int(r["weight_trials"])
            zero.append({"code": r["code"], "model": r["model"], "ber": float(r["ber"]),
                         "trials": n_tr, "upper_95": 1 - 0.05 ** (1 / n_tr)})
report["zero_failure_points"] = zero
ub = 1 - 0.05 ** (1 / 200000)
check("zero observed failures in 200000 patterns bound the failure probability only to 1.5e-5",
      abs(ub - 1.498e-5) < 1e-7, f"upper 95 % bound {ub:.3e}; {len(zero)} such points")

report["passed"] = all(c["passed"] for c in report["checks"])
(OUT / "code_verification.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
print(f"\n{sum(c['passed'] for c in report['checks'])} of {len(report['checks'])} checks passed")
sys.exit(0 if report["passed"] else 1)
