"""Run every experiment of the PUF-SENTINEL revision and store raw outputs.

Usage
-----
    python experiments/run_all.py --config configs/full.json  --jobs 14
    python experiments/run_all.py --config configs/smoke.json --jobs 4
    python experiments/run_all.py --config configs/full.json --stages ecc,overhead

Outputs go to experiments/results/<profile>/ (or --out). Every random stream
is derived from the master seed in src/common/seeding.py, so two runs with the
same configuration give identical raw files apart from wall-clock timings.

Stages and the experiment identifiers of EXPERIMENT_PLAN.md
    attack        E1 E2 E3 E4 E5a   modelling attacks on five constructions
    legacy        E3b               original PBKDF2 mask, same attacks
    xor_scaling   E5c               XOR size sweep with the two strongest attacks
    reliability   E5b               reliability-informed recovery through the mask
    stability_models  E5b           exact likelihood against the logistic surrogate,
                                    over noise levels and repetition counts
    prodlr_restarts   E5d           per-restart outcome of product LR on the 4-XOR PUF
    quality       E6                statistical quality with uncertainty
    ecc           E7                key reconstruction with real decoders
    overhead      E9                storage and host run time
E8 (temperature and voltage) is deliberately not run, see EXPERIMENT_PLAN.md.
E10 (reproduction) is scripts/reproduce_all.sh plus scripts/validate_outputs.py.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from joblib import Parallel, delayed

from attacks.models import ATTACK_CONFIG, fit_predict, product_lr_single_restart
from attacks.reliability import (RELIABILITY_CONFIG, exact_unstable_probability,
                                 fit_stability_model, sign_free_accuracy,
                                 unstable_indicator)
from common.seeding import MASTER_SEED, int_seed, rng_for
from common.stats import mean_t_interval
from error_correction.error_models import (analytic_failure_iid, sample_errors,
                                           sample_errors_batch)
from error_correction.fuzzy_extractor import CodeOffsetExtractor, CodeSpec
from metrics import quality as Q
from puf_models.arbiter import XorArbiterPUF, ZeroPUF, parity_features
from puf_models.challenges import count_duplicates, unique_challenges
from puf_models.mask import (MaskedPUF, hmac_mask, new_device_key,
                             pbkdf2_mask_legacy)

STAGES = ("attack", "legacy", "xor_scaling", "reliability", "stability_models",
          "prodlr_restarts", "quality", "ecc", "overhead")


# --------------------------------------------------------------------------
# construction factory
# --------------------------------------------------------------------------
def parse_construction(name: str) -> tuple[int, bool]:
    """Return (number of XORed arbiter chains, masked?)."""
    masked = name.startswith("mask_")
    core = name[5:] if masked else name
    if core == "only":
        return 0, True
    if core == "apuf":
        return 1, masked
    m = re.fullmatch(r"xor(\d+)", core)
    if not m:
        raise ValueError(f"unknown construction {name!r}")
    return int(m.group(1)), masked


def build(name: str, n: int, device: int, noisiness: float, legacy: bool = False):
    """Return (puf, base, k). Devices with the same index share base weights
    and key across constructions, so masked and unmasked variants differ only
    by the mask."""
    k, masked = parse_construction(name)
    base = ZeroPUF(n) if k == 0 else XorArbiterPUF(n, k, rng_for("weights", k, device),
                                                   noisiness)
    if not masked:
        return base, base, k
    key = new_device_key(rng_for("key", device))
    device_id = hashlib.sha256(f"device-{device}".encode()).digest()[:16]
    return MaskedPUF(base, key, device_id, legacy_pbkdf2=legacy), base, k


# --------------------------------------------------------------------------
# E1 to E5: modelling attacks
# --------------------------------------------------------------------------
def attack_unit(section: str, name: str, device: int, dist: str, noisiness: float,
                n: int, budgets: list[int], n_test: int, val_fraction: float,
                attacks: list[str], rho: float, legacy: bool = False) -> list[dict]:
    max_n = max(budgets)
    C = unique_challenges(max_n + n_test, n, rng_for("challenges", device, dist, max_n, n_test),
                          dist=dist, rho=rho)
    assert count_duplicates(C) == 0
    puf, base, k = build(name, n, device, noisiness, legacy)
    masked = isinstance(puf, MaskedPUF)
    noise_rng = rng_for("noise", device, dist, max_n) if noisiness > 0 else None
    base_noisy = base.eval(C, noise_rng)
    base_ideal = base.eval(C)
    mask = puf.mask(C) if masked else np.zeros(len(C), dtype=np.uint8)
    y, y_ideal = base_noisy ^ mask, base_ideal ^ mask
    phi = parity_features(C)
    Xte, yte, yte_ideal = phi[max_n:], y[max_n:], y_ideal[max_n:]
    k_model = k if k > 0 else 2
    rows = []

    def record(attack, N, n_tr, res, target, target_ideal):
        rows.append({
            "section": section, "construction": name, "device": device, "dist": dist,
            "noisiness": noisiness, "budget": N, "n_train": n_tr, "n_val": N - n_tr,
            "n_test": n_test, "attack": attack, "val_acc": round(res["val_acc"], 6),
            "test_acc": round(float(np.mean(res["test_pred"] == target)), 6),
            "test_acc_ideal": round(float(np.mean(res["test_pred"] == target_ideal)), 6),
            "seconds": round(res["seconds"], 3)})

    for N in budgets:
        n_val = max(1, int(round(val_fraction * N)))
        n_tr = N - n_val
        for atk in attacks:
            seed = int_seed("fit", section, name, device, dist, str(noisiness), N, atk)
            res = fit_predict(atk, phi[:n_tr], y[:n_tr], phi[n_tr:N], y[n_tr:N], Xte,
                              seed, k_model)
            record(atk, N, n_tr, res, yte, yte_ideal)
        if masked and k > 0 and not legacy:
            # Known-key diagnostic: the adversary strips the mask and attacks
            # what is left. Not a realistic capability; it isolates the mask.
            atk = "lr" if k == 1 else "lr_xor"
            y_or = y ^ mask
            assert np.array_equal(y_or, base_noisy)
            seed = int_seed("fit", section, name, device, dist, str(noisiness), N, "oracle")
            res = fit_predict(atk, phi[:n_tr], y_or[:n_tr], phi[n_tr:N], y_or[n_tr:N],
                              Xte, seed, k_model)
            record("oracle_" + atk, N, n_tr, res, base_noisy[max_n:], base_ideal[max_n:])
    return rows


def stage_attack(cfg, jobs):
    a = cfg["attack"]
    units = [(name, d, c["dist"], c["noisiness"])
             for name in a["constructions"] for c in a["conditions"]
             for d in range(a["devices"])]
    out = Parallel(n_jobs=jobs, verbose=5)(
        delayed(attack_unit)("attack", name, d, dist, nu, cfg["n_stages"], a["budgets"],
                             a["n_test"], a["val_fraction"], a["attacks"], a["markov_rho"])
        for name, d, dist, nu in units)
    return {"attack_raw.csv": [r for rows in out for r in rows]}


def stage_legacy(cfg, jobs):
    a, l = cfg["attack"], cfg["legacy_pbkdf2"]
    out = Parallel(n_jobs=jobs, verbose=5)(
        delayed(attack_unit)("legacy", name, d, "uniform", 0.0, cfg["n_stages"],
                             l["budgets"], l["n_test"], a["val_fraction"], l["attacks"],
                             a["markov_rho"], True)
        for name in ("mask_apuf", "mask_xor2") for d in range(l["devices"]))
    return {"legacy_raw.csv": [r for rows in out for r in rows]}


def stage_xor_scaling(cfg, jobs):
    a, x = cfg["attack"], cfg["xor_scaling"]
    out = Parallel(n_jobs=jobs, verbose=5)(
        delayed(attack_unit)("xor_scaling", "apuf" if k == 1 else f"xor{k}", d, "uniform",
                             0.0, cfg["n_stages"], x["budgets"], x["n_test"],
                             a["val_fraction"], x["attacks"], a["markov_rho"])
        for k in x["k"] for d in range(x["devices"]))
    return {"xor_scaling_raw.csv": [r for rows in out for r in rows]}


# --------------------------------------------------------------------------
# E5b: reliability-informed recovery through the mask
# --------------------------------------------------------------------------
def reliability_unit(device: int, n: int, budgets, n_test, noisiness, repeats):
    max_n = max(budgets)
    C = unique_challenges(max_n + n_test, n, rng_for("rel", "challenges", device))
    masked, base, _ = build("mask_apuf", n, device, noisiness)
    u = unstable_indicator(masked, C[:max_n], repeats, rng_for("rel", "noise", device))
    u_base = unstable_indicator(base, C[:max_n], repeats, rng_for("rel", "noise", device))
    phi = parity_features(C)
    hidden = base.eval(C[max_n:])
    w_true = base.weights[0]
    rows = []
    for N in budgets:
        w_hat = fit_stability_model(phi[:N], u[:N], int_seed("rel", "fit", device, N))
        cos = float(abs(w_hat @ w_true) / (np.linalg.norm(w_hat) * np.linalg.norm(w_true)))
        rows.append({"device": device, "budget": N, "queries": N * repeats,
                     "noisiness": noisiness, "repeats": repeats,
                     "unstable_fraction": round(float(u[:N].mean()), 6),
                     "mask_cancels": bool(np.array_equal(u, u_base)),
                     "hidden_model_acc": round(sign_free_accuracy(w_hat, phi[max_n:], hidden), 6),
                     "abs_cosine": round(cos, 6)})
    return rows


def stage_reliability(cfg, jobs):
    r = cfg["reliability_attack"]
    out = Parallel(n_jobs=jobs, verbose=5)(
        delayed(reliability_unit)(d, cfg["n_stages"], r["budgets"], r["n_test"],
                                  r["noisiness"], r["repeats"])
        for d in range(r["devices"]))
    return {"reliability_raw.csv": [x for rows in out for x in rows]}


# --------------------------------------------------------------------------
# E5b extended: exact stability likelihood against the logistic surrogate
# --------------------------------------------------------------------------
def stability_unit(device: int, nu: float, n: int, s: dict) -> list[dict]:
    """The mask cancels in the stability indicator (verified bit for bit in
    stage `reliability` and by a unit test), so the unmasked chain is queried
    here. That avoids several million HMAC calls and changes nothing."""
    base_cfg = s["baseline"]
    budgets_all = sorted(set(base_cfg["budgets"]) | {s["grid_budget"]})
    max_n = max(budgets_all)
    C = unique_challenges(max_n + s["n_test"], n, rng_for("stab", "challenges", device))
    base, _, _ = build("apuf", n, device, nu)
    phi = parity_features(C)
    hidden = base.eval(C[max_n:])
    w_true = base.weights[0]
    v_true = w_true / (nu * np.sqrt(n + 1))            # identifiable parameter
    z_true = phi[:max_n].astype(np.float64) @ v_true
    rows = []
    for T in s["repeats"]:
        u = unstable_indicator(base, C[:max_n], T, rng_for("stab", "noise", device, str(nu), T))
        predicted = float(exact_unstable_probability(z_true, T).mean())
        is_base = abs(nu - base_cfg["noisiness"]) < 1e-12 and T == base_cfg["repeats"]
        for N in (budgets_all if is_base else [s["grid_budget"]]):
            fitted = {}
            for model in s["models"]:           # logistic first: it seeds the warm start
                seed = int_seed("stab", "fit", device, str(nu), T, N, model)
                if model == "logistic":
                    w_hat, info = fit_stability_model(phi[:N], u[:N], seed, model="logistic",
                                                      repeats=T, return_info=True)
                elif model == "exact_cold":
                    w_hat, info = fit_stability_model(phi[:N], u[:N], seed, model="exact",
                                                      repeats=T, return_info=True)
                elif model == "exact_warm":
                    w_hat, info = fit_stability_model(phi[:N], u[:N], seed, model="exact",
                                                      repeats=T, init=fitted["logistic"],
                                                      return_info=True)
                else:
                    raise ValueError(model)
                fitted[model] = w_hat
                cos = float(abs(w_hat @ w_true) / (np.linalg.norm(w_hat) * np.linalg.norm(w_true)))
                rows.append({
                    "device": device, "noisiness": nu, "repeats": T, "budget": N,
                    "queries": N * T, "model": model,
                    "unstable_fraction": round(float(u[:N].mean()), 6),
                    "unstable_fraction_all": round(float(u.mean()), 6),
                    "predicted_unstable_fraction": round(predicted, 6),
                    "hidden_model_acc": round(sign_free_accuracy(w_hat, phi[max_n:], hidden), 6),
                    "abs_cosine": round(cos, 6),
                    "norm_ratio": round(float(np.linalg.norm(w_hat) / np.linalg.norm(v_true)), 6),
                    "val_loss": round(info["val_loss"], 6)})
    return rows


def stage_stability_models(cfg, jobs):
    s = cfg["stability_models"]
    out = Parallel(n_jobs=jobs, verbose=5)(
        delayed(stability_unit)(d, nu, cfg["n_stages"], s)
        for nu in s["noisiness"] for d in range(s["devices"]))
    return {"stability_models_raw.csv": [x for rows in out for x in rows]}


# --------------------------------------------------------------------------
# E5d: per-restart outcome of the product-model attack on the k-XOR PUF
# --------------------------------------------------------------------------
def prodlr_unit(device: int, restart: int, n: int, a: dict, x: dict, r: dict) -> dict:
    k, N, n_test = r["k"], r["budget"], x["n_test"]
    max_n = max(x["budgets"])                 # same challenges as stage xor_scaling
    C = unique_challenges(max_n + n_test, n, rng_for("challenges", device, "uniform", max_n, n_test))
    puf, _, _ = build(f"xor{k}", n, device, 0.0)
    y = puf.eval(C)
    phi = parity_features(C)
    n_val = int(round(a["val_fraction"] * N))
    n_tr = N - n_val
    res = product_lr_single_restart(phi[:n_tr], y[:n_tr], phi[n_tr:N], y[n_tr:N], phi[max_n:],
                                    int_seed("prodlr", device, k, N, restart), k)
    return {"k": k, "device": device, "restart": restart, "budget": N,
            "val_acc": round(res["val_acc"], 6),
            "test_acc": round(float(np.mean(res["test_pred"] == y[max_n:])), 6)}


def stage_prodlr_restarts(cfg, jobs):
    r = cfg["prodlr_restarts"]
    rows = Parallel(n_jobs=jobs, verbose=5)(
        delayed(prodlr_unit)(d, i, cfg["n_stages"], cfg["attack"], cfg["xor_scaling"], r)
        for d in range(r["devices"]) for i in range(r["restarts"]))
    return {"prodlr_restarts_raw.csv": rows}


# --------------------------------------------------------------------------
# E6: statistical quality
# --------------------------------------------------------------------------
def quality_unit(name: str, cfg: dict) -> dict:
    q, n = cfg["quality"], cfg["n_stages"]
    D, N = q["devices"], q["n_challenges"]
    C = rng_for("quality", "challenges").integers(0, 2, size=(N, n), dtype=np.uint8)
    Cf = C[:q["flip_challenges"]]
    R = np.empty((D, N), dtype=np.uint8)
    flip = np.empty((D, n))
    intra = np.empty(D)
    for d in range(D):
        puf, base, _ = build(name, n, d, q["noisiness_for_reliability"])
        R[d] = puf.eval(C)
        flip[d] = Q.flip_rate_profile(puf, Cf)
        # the mask cancels in the XOR of two readings, so the base is enough
        intra[d] = np.mean(base.eval(C, rng_for("quality", "noise", d)) != base.eval(C))
    uni = Q.uniformity(R)
    pair = Q.pairwise_hd(R)
    lo, hi = Q.uniqueness_device_bootstrap(R, rng_for("quality", "boot", name), q["bootstrap"])
    L = q["nist_sequence_bits"]
    seqs = R[:, :(N // L) * L].reshape(-1, L)
    freq_p = [Q.nist_frequency_p(s) for s in seqs]
    runs_p = [Q.nist_runs_p(s) for s in seqs]
    shannon = [Q.shannon_entropy_bit(float(u)) for u in uni]
    minent = [Q.mcv_min_entropy(R[d]) for d in range(D)]
    prof = flip.mean(axis=0)
    return {
        "construction": name, "devices": D, "challenges": N,
        "uniformity": {**mean_t_interval(uni), "min": float(uni.min()), "max": float(uni.max())},
        "bit_aliasing_sd": float(Q.bit_aliasing(R).std(ddof=1)),
        "bit_aliasing_sd_expected_ideal": float(0.5 / np.sqrt(D)),
        "uniqueness": {"value": float(pair.mean()), "boot_lo": lo, "boot_hi": hi,
                       "pairs": int(pair.size), "pair_sd": float(pair.std(ddof=1))},
        "shannon_marginal": mean_t_interval(shannon),
        "min_entropy_mcv": mean_t_interval(minent),
        "flip_rate": {"mean": float(prof.mean()), "min": float(prof.min()),
                      "max": float(prof.max()), "profile": [round(float(x), 5) for x in prof],
                      "profile_sd_devices": [round(float(x), 5) for x in flip.std(axis=0, ddof=1)]},
        "nist_frequency": Q.nist_summary(freq_p), "nist_runs": Q.nist_summary(runs_p),
        "intra_ber": {**mean_t_interval(intra), "noisiness": q["noisiness_for_reliability"]},
        "per_device": [{"construction": name, "device": d, "uniformity": round(float(uni[d]), 6),
                        "shannon_marginal": round(shannon[d], 6),
                        "min_entropy_mcv": round(minent[d], 6),
                        "intra_ber": round(float(intra[d]), 6)} for d in range(D)],
        "pairwise_hd": [round(float(x), 6) for x in pair],
    }


def stage_quality(cfg, jobs):
    names = cfg["quality"]["constructions"]
    out = Parallel(n_jobs=min(jobs, len(names)), verbose=5)(
        delayed(quality_unit)(name, cfg) for name in names)
    per_device = [r for o in out for r in o.pop("per_device")]
    return {"quality_summary.json": {o["construction"]: o for o in out},
            "quality_per_device.csv": per_device}


# --------------------------------------------------------------------------
# E7: key reconstruction with real decoders
# --------------------------------------------------------------------------
def ecc_unit(spec_d: dict, model: str, ber: float, e: dict) -> dict:
    spec = CodeSpec(**spec_d)
    fe = CodeOffsetExtractor(spec)
    rng = rng_for("ecc", spec.name, model, str(ber))
    succ = agree = 0
    for _ in range(e["decoder_trials"]):
        w = rng.integers(0, 2, size=spec.puf_bits, dtype=np.uint8)
        key, helper = fe.gen(w, rng)
        err = sample_errors(model, ber, spec.puf_bits, rng, e["rho_common_mode"])
        key2, _ = fe.rep(w ^ err, helper)
        ok = key2 == key
        succ += ok
        agree += ok == bool((fe.block_error_weights(err) <= spec.t).all())
    fails, done = 0, 0
    while done < e["weight_trials"]:
        b = min(20000, e["weight_trials"] - done)
        E = sample_errors_batch(model, ber, spec.puf_bits, b, rng, e["rho_common_mode"])
        if spec.bits_per_symbol == 1:
            wts = E.reshape(b, spec.blocks, spec.n).sum(axis=2)
        else:
            wts = E.reshape(b, spec.blocks, spec.n, spec.bits_per_symbol).any(axis=3).sum(axis=2)
        fails += int((wts > spec.t).any(axis=1).sum())
        done += b
    analytic = (analytic_failure_iid(spec.family, spec.n, spec.t, spec.bits_per_symbol,
                                     spec.blocks, ber) if model == "iid" else "")
    return {"code": spec.name, "model": model, "ber": ber,
            "decoder_trials": e["decoder_trials"], "decoder_success": int(succ),
            "decoder_agrees_with_radius": int(agree),
            "weight_trials": e["weight_trials"], "weight_failures": fails,
            "analytic_failure_iid": analytic}


def _max_ber(spec: CodeSpec, target: float) -> float:
    lo, hi = 1e-6, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        f = analytic_failure_iid(spec.family, spec.n, spec.t, spec.bits_per_symbol,
                                 spec.blocks, mid)
        lo, hi = (mid, hi) if f <= target else (lo, mid)
    return lo


def stage_ecc(cfg, jobs):
    e = cfg["ecc"]
    rows = Parallel(n_jobs=jobs, verbose=5)(
        delayed(ecc_unit)(s, m, b, e) for s in e["codes"] for m in e["models"]
        for b in e["bers"])
    g0, g1, gn = e["analytic_grid"]
    grid = np.linspace(g0, g1, int(gn))
    analytic, design = [], {}
    for s in e["codes"]:
        spec = CodeSpec(**s)
        code = CodeOffsetExtractor(spec).code
        for p in grid:
            analytic.append({"code": spec.name, "ber": round(float(p), 6),
                             "failure": analytic_failure_iid(spec.family, spec.n, spec.t,
                                                             spec.bits_per_symbol,
                                                             spec.blocks, float(p))})
        design[spec.name] = {
            "family": spec.family, "field": "GF(2)" if spec.family == "BCH" else "GF(2^8)",
            "n": spec.n, "k": spec.k, "t": spec.t,
            "d": code.d_design if spec.family == "BCH" else code.d_min,
            "d_kind": "designed distance (lower bound)" if spec.family == "BCH" else "minimum distance",
            "blocks": spec.blocks, "puf_bits": spec.puf_bits, "secret_bits": spec.secret_bits,
            "helper_bits": spec.helper_bits, "leakage_bound_bits": spec.leakage_bound_bits,
            "max_ber_fail_1e-6": _max_ber(spec, 1e-6), "max_ber_fail_1e-3": _max_ber(spec, 1e-3)}
    return {"ecc_raw.csv": rows, "ecc_analytic.csv": analytic, "ecc_design.json": design}


# --------------------------------------------------------------------------
# E9: overhead
# --------------------------------------------------------------------------
def stage_overhead(cfg, jobs):
    o, n = cfg["overhead"], cfg["n_stages"]
    rng = rng_for("overhead")
    key = new_device_key(rng)

    def best_time(fn, reps):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return min(ts), float(np.median(ts))

    Ch = rng.integers(0, 2, size=(o["hmac_challenges"], n), dtype=np.uint8)
    Cp = rng.integers(0, 2, size=(o["pbkdf2_challenges"], n), dtype=np.uint8)
    h_min, h_med = best_time(lambda: hmac_mask(key, Ch, 1), o["timing_repeats"])
    p_min, p_med = best_time(lambda: pbkdf2_mask_legacy(key, Cp, 1), o["timing_repeats"])
    codes = {}
    for s in cfg["ecc"]["codes"]:
        spec = CodeSpec(**s)
        fe = CodeOffsetExtractor(spec)
        w = rng.integers(0, 2, size=spec.puf_bits, dtype=np.uint8)
        _, helper = fe.gen(w, rng)
        e = np.zeros(spec.puf_bits, dtype=np.uint8)
        e[np.arange(spec.t) * spec.bits_per_symbol] = 1     # t errors in block 1
        ts = []
        for _ in range(o["decoder_repeats"]):
            t0 = time.perf_counter()
            fe.rep(w ^ e, helper)
            ts.append(time.perf_counter() - t0)
        codes[spec.name] = {"helper_bits": spec.helper_bits, "puf_bits": spec.puf_bits,
                            "secret_bits": spec.secret_bits,
                            "rep_ms_median": 1e3 * float(np.median(ts))}
    return {"overhead.json": {
        "note": "host timings of the Python reference code, not embedded measurements",
        "hmac_mask_us_per_challenge": {"min": 1e6 * h_min / len(Ch), "median": 1e6 * h_med / len(Ch)},
        "pbkdf2_mask_us_per_challenge": {"min": 1e6 * p_min / len(Cp), "median": 1e6 * p_med / len(Cp)},
        "pbkdf2_over_hmac_ratio": p_med / len(Cp) / (h_med / len(Ch)),
        "hmac_invocations_per_mask": {"hmac": 1, "pbkdf2_1000": 1000},
        "sha256_compressions_per_mask": {
            "hmac": "4 (two for the padded key blocks, one for the 49-byte message, one outer)",
            "pbkdf2_1000": "about 2002 with precomputed key pads (2 per iteration)"},
        "mask_storage_bits": {"device_key": 256, "device_id": 128},
        "codes": codes}}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def write_output(path: Path, obj) -> None:
    if path.suffix == ".json":
        path.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    else:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(obj[0].keys()))
            w.writeheader()
            w.writerows(obj)


def versions() -> dict:
    import joblib
    import scipy
    import sklearn
    import torch
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "torch": torch.__version__,
            "joblib": joblib.__version__, "cpu_count": os.cpu_count()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.json")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else ROOT / "experiments" / "results" / cfg["profile"]
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {"stages": {}})
    manifest.update({"master_seed": MASTER_SEED, "config_file": args.config, "config": cfg,
                     "attack_config": ATTACK_CONFIG, "reliability_config": RELIABILITY_CONFIG,
                     "versions": versions(), "jobs": args.jobs})
    fns = {"attack": stage_attack, "legacy": stage_legacy, "xor_scaling": stage_xor_scaling,
           "reliability": stage_reliability, "stability_models": stage_stability_models,
           "prodlr_restarts": stage_prodlr_restarts, "quality": stage_quality, "ecc": stage_ecc,
           "overhead": stage_overhead}
    for stage in args.stages.split(","):
        t0 = time.time()
        print(f"=== stage {stage} ===", flush=True)
        files = fns[stage](cfg, args.jobs)
        for fname, obj in files.items():
            write_output(out / fname, obj)
        manifest["stages"][stage] = {
            "seconds": round(time.time() - t0, 1),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in files}}
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        print(f"=== stage {stage} done in {time.time() - t0:.0f} s ===", flush=True)


if __name__ == "__main__":
    main()
