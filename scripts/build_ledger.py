"""Build RESULTS_VERIFICATION_LEDGER.csv.

Every number that the manuscript prints through a macro is listed with its
source file and calculation. Wherever feasible the value is RECOMPUTED here
from the raw result files with code that shares nothing with
experiments/analyze.py (plain csv, statistics, fractions, scipy.stats.t), and
compared with the value the manuscript uses. The value of the first revision
(archive/v1_2026-10-11) is recorded next to it.

Status values
    VERIFIED            independent recomputation equals the manuscript value
    VERIFIED, CHANGED   as above, and the value differs from the first revision
    VERIFIED, NEW       as above, and the quantity is new in this revision
    CONFIG              a configuration constant, checked against the manifest
    HOST TIMING         wall-clock measurement, not reproducible exactly
    NOT RECOMPUTED      produced by analyze.py only (listed with the reason)
    MISMATCH            independent value differs from the manuscript value

Usage:  python scripts/build_ledger.py        exit code 1 on any MISMATCH
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics as st
import sys
from collections import defaultdict
from fractions import Fraction
from math import comb
from pathlib import Path

from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "experiments" / "results" / "full"
V1 = ROOT / "archive" / "v1_2026-10-11" / "tables" / "numbers.tex"


def rd(name):
    with open(FULL / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def thousands(n):
    return f"{int(n):,}".replace(",", "\\,")


def p1(x):
    return f"{100 * x:.1f}"


def p2(x):
    return f"{100 * x:.2f}"


def macros_of(path):
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", path.read_text(encoding="utf-8")))


cfg = json.loads((FULL / "manifest.json").read_text(encoding="utf-8"))["config"]
A = cfg["attack"]
BMAX = max(A["budgets"])
ORDER = A["attacks"]

# ------------------------------------------------------------------ attacks
att = defaultdict(dict)
for r in rd("attack_raw.csv"):
    att[(r["construction"], r["dist"], float(r["noisiness"]), int(r["budget"]), r["attack"])][int(r["device"])] = r


def cell(c, a, b=BMAX, dist="uniform", nu=0.0, field="test_acc"):
    d = att[(c, dist, nu, b, a)]
    return [float(d[k][field]) for k in sorted(d)]


def best(c, b=BMAX, dist="uniform", nu=0.0, field="test_acc"):
    out = []
    for dev in sorted(att[(c, dist, nu, b, ORDER[0])]):
        top, top_v = None, -1.0
        for a in ORDER:                              # first attack wins ties
            v = float(att[(c, dist, nu, b, a)][dev]["val_acc"])
            if v > top_v:
                top, top_v = a, v
        out.append(float(att[(c, dist, nu, b, top)][dev][field]))
    return out


def tci(x, conf=0.95):
    m, s, n = st.mean(x), st.stdev(x), len(x)
    h = sps.t.ppf(0.5 + conf / 2, n - 1) * s / math.sqrt(n)
    return m - h, m + h


def tost(x, margin=A["equivalence_margin"]):
    m, s, n = st.mean(x), st.stdev(x), len(x)
    se = s / math.sqrt(n)
    return max(1 - sps.t.cdf((m - (0.5 - margin)) / se, n - 1), sps.t.cdf((m - (0.5 + margin)) / se, n - 1))


def first_budget(c, a, thr):
    for b in sorted(A["budgets"]):
        if st.mean(cell(c, a, b)) >= thr:
            return thousands(b)
    return "n.a."


MASKED = ("mask_only", "mask_apuf", "mask_xor2")
CONDS = [(c["dist"], c["noisiness"]) for c in A["conditions"]]
m_cells = [(k, [float(v[d]["test_acc"]) for d in sorted(v)]) for k, v in att.items()
           if k[0] in MASKED and not k[4].startswith("oracle")]
single = [x for k, x in m_cells if k[3] == BMAX]
prim = [best(c, dist=d, nu=nu) for c in MASKED for d, nu in CONDS]
pvals = sorted(sps.ttest_1samp(x, 0.5).pvalue for x in single)
holm = 0
for i, pv in enumerate(pvals):
    if pv <= 0.05 / (len(pvals) - i):
        holm += 1
    else:
        break
binom_var = 0.25 / A["n_test"]


def margin_supported(x):
    lo, hi = tci(x, 0.90)
    return max(abs(lo - 0.5), abs(hi - 0.5))


E = {}                                              # macro -> (value, source, method)


def add(name, value, source, method):
    E[name] = (str(value), source, method)


SRC = "attack_raw.csv"
add("ApufLrMax", p1(st.mean(cell("apuf", "lr"))), SRC, "mean test accuracy over devices, LR, 30000 CRPs, uniform, noise free")
add("ApufRfMax", p1(st.mean(cell("apuf", "rf"))), SRC, "same for RF")
add("ApufGbMax", p1(st.mean(cell("apuf", "gb"))), SRC, "same for GB")
add("ApufMlpMax", p1(st.mean(cell("apuf", "mlp"))), SRC, "same for MLP")
add("ApufLrMarkovMax", p1(st.mean(cell("apuf", "lr", dist="markov"))), SRC, "LR, correlated challenges")
add("ApufLrNoisyMax", p1(st.mean(cell("apuf", "lr", nu=0.1))), SRC, "LR, noisy condition, against noisy test bits")
add("ApufLrNoisyIdealMax", p1(st.mean(best("apuf", nu=0.1, field="test_acc_ideal"))), SRC,
    "selected attack, noisy condition, against noise-free test bits")
add("ApufLrFirstNinetyFive", first_budget("apuf", "lr", 0.95), SRC, "smallest budget with mean LR accuracy >= 95 %")
add("ApufLrFirstNinetyNine", first_budget("apuf", "lr", 0.99), SRC, "smallest budget with mean LR accuracy >= 99 %")
for tag, a in (("Lr", "lr"), ("Svm", "svm"), ("Rf", "rf"), ("Gb", "gb"), ("Mlp", "mlp"), ("Prod", "lr_xor")):
    add(f"Xor{tag}Max", p1(st.mean(cell("xor2", a))), SRC, f"mean test accuracy, 2-XOR, {a}, 30000 CRPs")
add("XorBestMax", p1(st.mean(best("xor2"))), SRC, "selected attack (best validation accuracy per device), 2-XOR")
add("XorBestMarkovMax", p1(st.mean(best("xor2", dist="markov"))), SRC, "selected attack, correlated challenges")
add("XorBestNoisyMax", p1(st.mean(best("xor2", nu=0.1))), SRC, "selected attack, noisy condition")
add("XorBestNoisyIdealMax", p1(st.mean(best("xor2", nu=0.1, field="test_acc_ideal"))), SRC,
    "selected attack, noisy condition, against noise-free test bits")
add("XorProdFirstNinetyFive", first_budget("xor2", "lr_xor", 0.95), SRC, "smallest budget with mean product LR accuracy >= 95 %")
fb = best("mask_xor2")
add("FullBestMax", p2(st.mean(fb)), SRC, "selected attack on the masked 2-XOR PUF, mean over devices")
add("FullBestLo", p2(tci(fb)[0]), SRC, "lower end of the 95 % t interval over devices")
add("FullBestHi", p2(tci(fb)[1]), SRC, "upper end of the 95 % t interval over devices")
add("FullOracleMax", p1(st.mean(cell("mask_xor2", "oracle_lr_xor"))), SRC, "product LR after removing the mask with the key")
add("MaskApufOracleMax", p1(st.mean(cell("mask_apuf", "oracle_lr"))), SRC, "LR after removing the mask with the key")
add("MaskedCells", len(m_cells), SRC, "count of masked cells: 3 constructions x 3 conditions x 6 budgets x 6 attacks")
add("MaskedFits", thousands(sum(len(x) for _, x in m_cells)), SRC, "count of masked fits")
add("MaskedCellMin", p2(min(st.mean(x) for _, x in m_cells)), SRC, "smallest masked cell mean")
add("MaskedCellMax", p2(max(st.mean(x) for _, x in m_cells)), SRC, "largest masked cell mean")
add("MaskedDevMin", p2(min(min(x) for _, x in m_cells)), SRC, "smallest single-device accuracy in masked cells")
add("MaskedDevMax", p2(max(max(x) for _, x in m_cells)), SRC, "largest single-device accuracy in masked cells")
add("MaskedSingleCells", len(single), SRC, "single-attack masked cells at 30000 CRPs")
add("MaskedSingleEquiv", sum(tost(x) < 0.05 for x in single), SRC, "cells passing TOST at 0.05 with margin 0.01")
add("MaskedSingleSig", sum(pv < 0.05 for pv in pvals), SRC, "cells with uncorrected two-sided t test p < 0.05")
add("MaskedBinomSd", p1(math.sqrt(binom_var)), "configs/full.json", "sqrt(0.25 / n_test)")
add("StatMaxTostP", f"{max(tost(x) for x in prim):.3f}", SRC, "largest TOST p value over the nine primary cells")
add("StatSdMin", p2(min(st.stdev(x) for x in prim)), SRC, "smallest SD of device accuracies, primary cells")
add("StatSdMax", p2(max(st.stdev(x) for x in prim)), SRC, "largest SD of device accuracies, primary cells")
add("StatVrMin", f"{min(st.variance(x) / binom_var for x in prim):.2f}", SRC, "variance between devices / binomial variance, minimum")
add("StatVrMax", f"{max(st.variance(x) / binom_var for x in prim):.2f}", SRC, "same, maximum")
add("StatMarginSupported", p2(max(margin_supported(x) for x in prim)), SRC, "largest distance of a 90 % t interval end from 0.5")
add("StatHolmSig", holm, SRC, "cells significant after Holm step-down at 0.05 over the 54 t tests")

# ------------------------------------------------------------------ reliability (E5b)
rel = defaultdict(list)
relu = defaultdict(list)
for r in rd("reliability_raw.csv"):
    rel[int(r["budget"])].append(float(r["hidden_model_acc"]))
    relu[int(r["budget"])].append(float(r["unstable_fraction"]))
rb = sorted(rel)
SRC = "reliability_raw.csv"
add("RelRepeats", cfg["reliability_attack"]["repeats"], "configs/full.json", "repetitions T")
add("RelBudgetMin", thousands(rb[0]), SRC, "smallest challenge budget")
add("RelBudgetSecond", thousands(rb[1]), SRC, "second budget")
add("RelBudgetMax", thousands(rb[-1]), SRC, "largest budget")
add("RelAccMin", p1(st.mean(rel[rb[0]])), SRC, "mean hidden-model agreement at the smallest budget")
add("RelAccMinWorst", p1(min(rel[rb[0]])), SRC, "worst device at the smallest budget")
add("RelAccMinBest", p1(max(rel[rb[0]])), SRC, "best device at the smallest budget")
add("RelAccSecond", p1(st.mean(rel[rb[1]])), SRC, "mean agreement at the second budget")
add("RelAccSecondWorst", p1(min(rel[rb[1]])), SRC, "worst device at the second budget")
add("RelAccMax", p1(st.mean(rel[rb[-1]])), SRC, "mean agreement at the largest budget (abstract)")
add("RelAccMaxWorst", p1(min(rel[rb[-1]])), SRC, "worst device at the largest budget")
add("RelUnstable", p1(st.mean(relu[rb[-1]])), SRC, "mean fraction of unstable challenges")

# ------------------------------------------------------------------ stability models
sm = rd("stability_models_raw.csv")
sc = cfg["stability_models"]
GB = sc["grid_budget"]
by = defaultdict(dict)
for r in sm:
    by[(float(r["noisiness"]), int(r["repeats"]), int(r["budget"]), r["model"])][int(r["device"])] = r


def sacc(nu, T, b, model):
    d = by[(nu, T, b, model)]
    return [float(d[k]["hidden_model_acc"]) for k in sorted(d)]


grid = [(nu, T) for nu in sc["noisiness"] for T in sc["repeats"]]
bn, bT = sc["baseline"]["noisiness"], sc["baseline"]["repeats"]
bb = sorted(sc["baseline"]["budgets"])
SRC = "stability_models_raw.csv"
add("StabGridBudget", thousands(GB), "configs/full.json", "challenges per grid cell")
add("StabCells", len(grid), "configs/full.json", "noise levels x repetition counts")
add("StabFits", len(grid) * sc["devices"], SRC, "fits per estimator on the grid")
add("StabBaseSurr", p1(st.mean(sacc(bn, bT, bb[-1], "logistic"))), SRC, "surrogate, main setting, largest budget")
add("StabBaseSurrWorst", p1(min(sacc(bn, bT, bb[-1], "logistic"))), SRC, "worst device")
add("StabBaseExact", p1(st.mean(sacc(bn, bT, bb[-1], "exact_warm"))), SRC, "exact likelihood, warm start, main setting, largest budget")
add("StabBaseExactWorst", p1(min(sacc(bn, bT, bb[-1], "exact_warm"))), SRC, "worst device")
add("StabBaseBudgetSecond", thousands(bb[1]), "configs/full.json", "second budget")
add("StabBaseExactSecond", p1(st.mean(sacc(bn, bT, bb[1], "exact_warm"))), SRC, "exact, second budget")
add("StabBaseSurrSecond", p1(st.mean(sacc(bn, bT, bb[1], "logistic"))), SRC, "surrogate, second budget")
add("StabBaseExactFirst", p1(st.mean(sacc(bn, bT, bb[0], "exact_warm"))), SRC, "exact, smallest budget")
add("StabBaseExactFirstWorst", p1(min(sacc(bn, bT, bb[0], "exact_warm"))), SRC, "worst device")
add("StabBaseSurrFirst", p1(st.mean(sacc(bn, bT, bb[0], "logistic"))), SRC, "surrogate, smallest budget")
add("StabBaseSurrFirstWorst", p1(min(sacc(bn, bT, bb[0], "logistic"))), SRC, "worst device")
gap = 0.0
for key, d in by.items():
    obs = st.mean(float(v["unstable_fraction_all"]) for v in d.values())
    pred = st.mean(float(v["predicted_unstable_fraction"]) for v in d.values())
    gap = max(gap, abs(obs - pred))
add("StabGap", p2(gap), SRC, "largest |observed - predicted| unstable fraction over cells (exact likelihood with true weights)")
per_dev = defaultdict(list)
cold_fail = 0
for nu, T in grid:
    w, l, c = sacc(nu, T, GB, "exact_warm"), sacc(nu, T, GB, "logistic"), sacc(nu, T, GB, "exact_cold")
    for d in range(len(w)):
        per_dev[d].append(w[d] - l[d])
    cold_fail += sum(x < 0.95 for x in c)
dd = [st.mean(v) for v in per_dev.values()]
add("StabDiffMean", f"{100 * st.mean(dd):+.2f}", SRC, "exact (warm) minus surrogate, averaged per device over the grid, then over devices")
add("StabDiffLo", f"{100 * tci(dd)[0]:+.2f}", SRC, "lower end of the 95 % t interval over devices")
add("StabDiffHi", f"{100 * tci(dd)[1]:+.2f}", SRC, "upper end")
add("StabColdFail", cold_fail, SRC, "exact-likelihood fits from a random start with agreement below 95 %")
add("StabGridMinSurr", p1(min(st.mean(sacc(nu, T, GB, "logistic")) for nu, T in grid)), SRC, "smallest cell mean, surrogate")
add("StabGridMinExact", p1(min(st.mean(sacc(nu, T, GB, "exact_warm")) for nu, T in grid)), SRC, "smallest cell mean, exact warm")
add("StabGridMinDevSurr", p1(min(min(sacc(nu, T, GB, "logistic")) for nu, T in grid)), SRC, "worst device in any cell, surrogate")
add("StabGridMinDevExact", p1(min(min(sacc(nu, T, GB, "exact_warm")) for nu, T in grid)), SRC, "worst device in any cell, exact warm")
nr = [st.mean(float(v["norm_ratio"]) for v in by[(nu, T, GB, "exact_warm")].values()) for nu, T in grid]
add("StabNormLo", f"{min(nr):.2f}", SRC, "smallest cell mean of fitted norm / true norm of w / sigma")
add("StabNormHi", f"{max(nr):.2f}", SRC, "largest cell mean of the same ratio")

# ------------------------------------------------------------------ XOR sweep, restarts, legacy
xs = defaultdict(list)
for r in rd("xor_scaling_raw.csv"):
    xs[(r["construction"], int(r["budget"]), r["attack"])].append(float(r["test_acc"]))
xb = sorted(cfg["xor_scaling"]["budgets"])
km = max(cfg["xor_scaling"]["k"])
SRC = "xor_scaling_raw.csv"
add("XsKmax", km, "configs/full.json", "largest number of chains")
add("XsBudgetLow", thousands(xb[0]), "configs/full.json", "budget")
add("XsBudgetHigh", thousands(xb[-1]), "configs/full.json", "budget")
add("XsMaxMlpLow", p1(st.mean(xs[(f"xor{km}", xb[0], "mlp")])), SRC, "MLP on 4-XOR, small budget")
add("XsMaxMlpHigh", p1(st.mean(xs[(f"xor{km}", xb[-1], "mlp")])), SRC, "MLP on 4-XOR, large budget")
add("XsMaxProdHighMin", p1(min(xs[(f"xor{km}", xb[-1], "lr_xor")])), SRC, "product LR on 4-XOR, worst device")
add("XsMaxProdHighMax", p1(max(xs[(f"xor{km}", xb[-1], "lr_xor")])), SRC, "product LR on 4-XOR, best device")
pr = rd("prodlr_restarts_raw.csv")
acc = [float(r["test_acc"]) for r in pr]
conv = [a for a in acc if a >= 0.80]
devs = sorted({int(r["device"]) for r in pr})
SRC = "prodlr_restarts_raw.csv"
add("PrK", pr[0]["k"], "configs/full.json", "chains")
add("PrRestarts", len(pr) // len(devs), "configs/full.json", "restarts per device")
add("PrTotal", len(pr), SRC, "restarts in total")
add("PrConverged", len(conv), SRC, "restarts with test accuracy >= 80 %")
add("PrConvMean", p1(st.mean(conv)), SRC, "mean accuracy of converged restarts")
add("PrConvMin", p1(min(conv)), SRC, "smallest accuracy of a converged restart")
add("PrConvMax", p1(max(conv)), SRC, "largest accuracy of a converged restart")
add("PrFailMax", p1(max(a for a in acc if a < 0.80)), SRC, "largest accuracy of a restart that did not converge")
bbv = []
for d in devs:
    rr = [r for r in pr if int(r["device"]) == d]
    bbv.append(float(max(rr, key=lambda r: float(r["val_acc"]))["test_acc"]))
add("PrBestMin", p1(min(bbv)), SRC, "worst device, restart selected on validation")
add("PrBestMax", p1(max(bbv)), SRC, "best device, restart selected on validation")
add("PrDevNoConv", sum(all(float(r["test_acc"]) < 0.80 for r in pr if int(r["device"]) == d) for d in devs), SRC, "devices without a converged restart")
leg = defaultdict(dict)
for r in rd("legacy_raw.csv"):
    leg[(r["construction"], int(r["budget"]), r["attack"])][int(r["device"])] = r
lb = max(k[1] for k in leg)
lsel = []
for dev in sorted(leg[("mask_xor2", lb, "lr")]):
    cands = [a for a in ORDER if ("mask_xor2", lb, a) in leg]
    top = max(cands, key=lambda a: (float(leg[("mask_xor2", lb, a)][dev]["val_acc"]), -cands.index(a)))
    lsel.append(float(leg[("mask_xor2", lb, top)][dev]["test_acc"]))
SRC = "legacy_raw.csv"
add("LegacyDevices", len(lsel), SRC, "devices")
add("LegacyBudget", thousands(lb), SRC, "budget")
add("LegacyFullMean", p2(st.mean(lsel)), SRC, "selected attack, PBKDF2-masked 2-XOR")
add("LegacyFullLo", p2(tci(lsel)[0]), SRC, "95 % t interval, lower")
add("LegacyFullHi", p2(tci(lsel)[1]), SRC, "95 % t interval, upper")

# ------------------------------------------------------------------ quality
qd = defaultdict(list)
for r in rd("quality_per_device.csv"):
    qd[r["construction"]].append(r)
qs = json.loads((FULL / "quality_summary.json").read_text(encoding="utf-8"))
SRC, SRC2 = "quality_per_device.csv", "quality_summary.json"
add("QualDevices", cfg["quality"]["devices"], "configs/full.json", "devices")
add("QualChallenges", thousands(cfg["quality"]["n_challenges"]), "configs/full.json", "challenges")
add("QualNistSeqs", cfg["quality"]["devices"] * (cfg["quality"]["n_challenges"] // cfg["quality"]["nist_sequence_bits"]),
    "configs/full.json", "devices x floor(challenges / sequence length)")
for c, tag in (("apuf", "Apuf"), ("xor2", "Xor"), ("mask_xor2", "Full")):
    u = [float(r["uniformity"]) for r in qd[c]]
    add(f"Qual{tag}Unif", f"{st.mean(u):.3f}", SRC, "mean uniformity over devices")
    add(f"Qual{tag}UnifMin", f"{min(u):.3f}", SRC, "smallest device uniformity")
    add(f"Qual{tag}UnifMax", f"{max(u):.3f}", SRC, "largest device uniformity")
    add(f"Qual{tag}Shannon", f"{st.mean(float(r['shannon_marginal']) for r in qd[c]):.3f}", SRC, "mean marginal Shannon entropy")
    add(f"Qual{tag}Ber", p1(st.mean(float(r["intra_ber"]) for r in qd[c])), SRC, "mean intra-device bit error rate at noisiness 0.1")
    hd = qs[c]["pairwise_hd"]
    add(f"Qual{tag}Uniq", f"{st.mean(hd):.3f}", SRC2, "mean of the stored pairwise Hamming distances")
    prof = qs[c]["flip_rate"]["profile"]
    add(f"Qual{tag}FlipMin", f"{min(prof):.2f}", SRC2, "smallest value of the stored flip-rate profile")
    add(f"Qual{tag}FlipMax", f"{max(prof):.2f}", SRC2, "largest value of the stored flip-rate profile")
    add(f"Qual{tag}NistFreq", f"{qs[c]['nist_frequency']['pass_proportion']:.2f}", SRC2, "stored pass proportion (per-sequence p values are not stored)")
add("QualApufUnifSd", f"{st.stdev(float(r['uniformity']) for r in qd['apuf']):.3f}", SRC, "SD of device uniformity")
add("QualFullUnifSd", f"{st.stdev(float(r['uniformity']) for r in qd['mask_xor2']):.3f}", SRC, "SD of device uniformity")
add("QualApufNistRuns", f"{qs['apuf']['nist_runs']['pass_proportion']:.2f}", SRC2, "stored pass proportion")
add("QualFullNistRuns", f"{qs['mask_xor2']['nist_runs']['pass_proportion']:.2f}", SRC2, "stored pass proportion")
s_ = cfg["quality"]["devices"] * (cfg["quality"]["n_challenges"] // cfg["quality"]["nist_sequence_bits"])
add("QualNistBandLo", f"{0.99 - 3 * math.sqrt(0.01 * 0.99 / s_):.3f}", "configs/full.json", "1 - alpha - 3 sqrt(alpha (1 - alpha) / s), alpha = 0.01")
add("QualApufMinEnt", f"{st.mean(float(r['min_entropy_mcv']) for r in qd['apuf']):.3f}", SRC, "mean MCV min-entropy")
add("QualFullMinEnt", f"{st.mean(float(r['min_entropy_mcv']) for r in qd['mask_xor2']):.3f}", SRC, "mean MCV min-entropy")

# ------------------------------------------------------------------ ECC
ecc = rd("ecc_raw.csv")


def exact_failure(n, t, u, blocks, p):
    ps = 1 - (1 - p) ** u
    ok = sum(Fraction(comb(n, i)) * ps ** i * (1 - ps) ** (n - i) for i in range(t + 1))
    return 1 - ok ** blocks


def thr(n, t, u, blocks):
    lo, hi = Fraction(0), Fraction(1, 2)
    for _ in range(40):
        mid = (lo + hi) / 2
        if exact_failure(n, t, u, blocks, mid.limit_denominator(10 ** 9)) <= Fraction(1, 10 ** 6):
            lo = mid
        else:
            hi = mid
    return float(lo)


codes = {c["name"]: (c["n"], c["t"], c["bits_per_symbol"], c["blocks"]) for c in cfg["ecc"]["codes"]}
ths = {k: thr(*v) for k, v in codes.items()}
bestc = max(ths, key=ths.get)


def floor2(x):
    return f"{math.floor(100 * x * 100 + 1e-9) / 100:.2f}"


def fail(code, model, ber):
    r = [x for x in ecc if x["code"] == code and x["model"] == model and abs(float(x["ber"]) - ber) < 1e-9][0]
    v = int(r["weight_failures"]) / int(r["weight_trials"])
    return f"{v:.4f}" if v < 0.01 else f"{v:.3f}"


SRC = "ecc_raw.csv"
add("EccDecoderTrials", thousands(sum(int(r["decoder_trials"]) for r in ecc)), SRC, "sum of decoder trials")
add("EccAgreement", p1(min(int(r["decoder_agrees_with_radius"]) / int(r["decoder_trials"]) for r in ecc)), SRC, "smallest agreement between decoder outcome and radius criterion")
add("EccWeightTrials", thousands(cfg["ecc"]["weight_trials"]), "configs/full.json", "error patterns per point")
add("EccBestName", bestc.replace(" x", " $\\times$"), "exact arithmetic", "configuration with the largest threshold")
add("EccBestBer", floor2(ths[bestc]), "exact arithmetic (fractions)", "largest BER with failure <= 1e-6, bisection, ROUNDED DOWN")
add("EccBestBerExact", f"{100 * ths[bestc]:.4f}", "exact arithmetic (fractions)", "same threshold to four decimals")
add("EccWorstBer", floor2(min(ths.values())), "exact arithmetic (fractions)", "smallest threshold over configurations, rounded down")
for b, tag in ((0.05, "Five"), (0.1, "Ten")):
    add(f"EccBestFailIid{tag}", fail(bestc, "iid", b), SRC, f"Monte Carlo failure fraction, best code, i.i.d., BER {b}")
    add(f"EccBestFailHet{tag}", fail(bestc, "heterogeneous", b), SRC, f"same, heterogeneous model")
    add(f"EccBestFailCom{tag}", fail(bestc, "common_mode", b), SRC, f"same, common-mode model")
add("EccZeroPoints", sum(int(r["weight_failures"]) == 0 for r in ecc), SRC, "points with zero observed failures")
ub = 1 - 0.05 ** (1 / cfg["ecc"]["weight_trials"])
add("EccZeroUpper", "$1.5\\times10^{-5}$" if abs(ub - 1.498e-5) < 1e-7 else f"{ub:.1e}", "closed form", "1 - 0.05^(1/N), exact one-sided 95 % bound for zero failures")
tc = json.loads((ROOT / "experiments" / "results" / "ecc_tail_check.json").read_text(encoding="utf-8"))
add("EccTailTrials", thousands(tc["trials"]), "ecc_tail_check.json", "patterns in the re-check")
add("EccTailFails", tc["failures"], "ecc_tail_check.json", "failures in the re-check")
add("EccTailExpected", f"{float(exact_failure(48, 16, 8, 1, Fraction(1, 100))) * tc['trials']:.1f}", "exact arithmetic", "exact failure probability x patterns")
add("EccTailMainP", f"{1 - (1 - float(exact_failure(48, 16, 8, 1, Fraction(1, 100)))) ** 200000:.2f}", "exact arithmetic", "P[at least one failure in 200000]")
cv = json.loads((ROOT / "experiments" / "results" / "code_verification.json").read_text(encoding="utf-8"))
add("EccIndepChecks", len(cv["checks"]), "code_verification.json", "independent checks, all must pass" if cv["passed"] else "SOME CHECKS FAILED")

# ------------------------------------------------------------------ hardware
hw = defaultdict(list)
for r in rd("hardware_raw.csv"):
    hw[(r["dataset"], r["variant"], int(r["budget"]), r["attack"])].append(float(r["test_acc"]))
meta = json.loads((FULL / "hardware_meta.json").read_text(encoding="utf-8"))
ab, xbh = max(meta["fpga_apuf"]["budgets"]), max(meta["fpga_xor4"]["budgets"])
SRC = "hardware_raw.csv"
add("HwApufBudget", thousands(ab), "hardware_meta.json", "budget")
add("HwXorBudget", thousands(xbh), "hardware_meta.json", "budget")
add("HwApufLr", p1(st.mean(hw[("fpga_apuf", "measured", ab, "lr")])), SRC, "mean over 5 splits of one device")
add("HwApufMaskedLr", p1(st.mean(hw[("fpga_apuf", "masked", ab, "lr")])), SRC, "same, software-masked")
add("HwApufOracle", p1(st.mean(hw[("fpga_apuf", "oracle", ab, "lr_xor")])), SRC, "same, mask removed with key")
add("HwXorProd", p1(st.mean(hw[("fpga_xor4", "measured", xbh, "lr_xor")])), SRC, "mean over 5 splits of one device")
add("HwXorMlp", p1(st.mean(hw[("fpga_xor4", "measured", xbh, "mlp")])), SRC, "same")
add("HwXorLr", p1(st.mean(hw[("fpga_xor4", "measured", xbh, "lr")])), SRC, "same")
add("HwXorMaskedProd", p1(st.mean(hw[("fpga_xor4", "masked", xbh, "lr_xor")])), SRC, "same, software-masked")
add("HwApufOnes", p1(meta["fpga_apuf"]["fraction_of_ones"]), "hardware_meta.json", "fraction of ones after mapping")
add("HwXorOnes", p1(meta["fpga_xor4"]["fraction_of_ones"]), "hardware_meta.json", "fraction of ones after mapping")

# ------------------------------------------------------------------ config and others
CONFIG = {"NumDevices": A["devices"], "NumTest": thousands(A["n_test"]), "BudgetMax": thousands(BMAX),
          "EqMargin": f"{100 * A['equivalence_margin']:.0f}"}
NOT_RECOMPUTED = {
    "StatPowerZero": "seeded simulation inside analyze.py (200000 draws, largest observed SD)",
    "StatPowerHalf": "seeded simulation inside analyze.py",
    "ReproTests": "number of tests collected by pytest",
    "ReproFiles": "number of raw files found identical in the clean-environment comparison",
}
TIMING = {"OvHmacUs", "OvPbkdfUs", "OvRatio", "EccRepMsMax"}

# ------------------------------------------------------------------ assemble
cur = macros_of(ROOT / "tables" / "numbers.tex")
old = macros_of(V1) if V1.exists() else {}
texts = {"manuscript": (ROOT / "CEE_SUBMISSION" / "manuscript.tex").read_text(encoding="utf-8"),
         "supplement": (ROOT / "CEE_SUBMISSION" / "supplement.tex").read_text(encoding="utf-8")}


def locations(macro):
    out = []
    for doc, t in texts.items():
        section, in_abs = "front matter", False
        for line in t.split("\n"):
            if "\\begin{abstract}" in line:
                in_abs = True
            if "\\end{abstract}" in line:
                in_abs = False
            ms = re.search(r"\\(?:sub)?section\*?\{([^}]*)\}", line)
            if ms:
                section = ms.group(1)
            if re.search(r"\\" + macro + r"(?![A-Za-z])", line):
                where = "Abstract" if in_abs else section
                tag = where if doc == "manuscript" else f"Supplement: {where}"
                if tag not in out:
                    out.append(tag)
    return "; ".join(out) if out else "not used in the text"


rows, mismatch = [], 0
for name, value in cur.items():
    v1 = old.get(name, "")
    if name in E:
        indep, source, method = E[name]
        okv = indep == value
        if not okv:
            status = "MISMATCH"
            mismatch += 1
        elif not v1:
            status = "VERIFIED, NEW"
        elif v1 != value:
            status = "VERIFIED, CHANGED"
        else:
            status = "VERIFIED"
    elif name in CONFIG:
        indep, source, method = str(CONFIG[name]), "manifest.json (config)", "configuration constant"
        status = "CONFIG" if indep == value else "MISMATCH"
        mismatch += status == "MISMATCH"
    elif name in TIMING:
        indep, source, method, status = "", "overhead.json", "wall-clock measurement on the host", "HOST TIMING"
    else:
        indep, source, method = "", "analyze.py", NOT_RECOMPUTED.get(name, "not independently recomputed")
        status = "NOT RECOMPUTED"
    rows.append({"macro": name, "manuscript_locations": locations(name), "source_file": source,
                 "calculation": method, "configuration": "configs/full.json, master seed 20261011",
                 "value_first_revision": v1, "value_in_manuscript": value,
                 "independently_recomputed_value": indep, "status": status})
with open(ROOT / "RESULTS_VERIFICATION_LEDGER.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
count = defaultdict(int)
for r in rows:
    count[r["status"]] += 1
print(dict(count))
for r in rows:
    if r["status"] in ("MISMATCH", "VERIFIED, CHANGED", "NOT RECOMPUTED"):
        print(f"  {r['status']:18s} {r['macro']:24s} v1={r['value_first_revision']!r} now={r['value_in_manuscript']!r} indep={r['independently_recomputed_value']!r}")
sys.exit(1 if mismatch else 0)
