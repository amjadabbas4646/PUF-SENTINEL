"""Experiments added in the second revision. They extend, and do not replace,
the stages of run_all.py. Raw outputs go to the same results folder.

Usage
-----
    python experiments/run_extensions.py --config configs/extensions_full.json  --jobs 14
    python experiments/run_extensions.py --config configs/extensions_smoke.json --jobs 4
    python experiments/run_extensions.py --stages entropy,classes

Stages
    classes          the benchmark controls applied to generic members of four
                     published obfuscation classes (challenge XOR key, secret
                     linear challenge map, keyed challenge PRF, keyless hashed
                     output), with a 2-XOR arbiter PUF as the base
    class_stability  what repeated queries reveal for each class (arbiter base)
    xor_stability    stability attack on the masked k-XOR arbiter PUF, with the
                     three outcomes kept apart: model recovery up to a global
                     sign, prediction of masked responses without the key and
                     prediction after a key compromise
    convergence      epochs, solver iterations, restarts and run time of every
                     attack on three constructions, with the seeds of the main
                     run, so the accuracies equal the stored ones
    entropy          bounds on the conditional min-entropy of the code-offset
                     secret given the helper data, for biased weak-PUF bits
    fpga             the ablation on sixteen measured FPGA instances: six
                     arbiter chains and ten 2-XOR PUFs composed in software
                     from pairs of measured chains (needs data/raw, see
                     scripts/download_public_data.py)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import sys
import time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
from joblib import Parallel, delayed

from attacks.models import ATTACK_CONFIG, fit_predict
from attacks.reliability import RELIABILITY_CONFIG, fit_stability_model, unstable_indicator
from attacks.stability_joint import (JOINT_CONFIG, fit_hashed_stability, fit_xor_stability,
                                     matched_cosines, sign_free_agreement, xor_model_bits)
from common.seeding import MASTER_SEED, int_seed, rng_for
from error_correction.entropy import key_entropy_bounds
from puf_models.arbiter import XorArbiterPUF, parity_features
from puf_models.challenges import count_duplicates, unique_challenges
from puf_models.constructions import ObfuscatedPUF, hash_output, random_invertible_gf2
from puf_models.mask import hmac_mask, new_device_key
from run_all import build, versions, write_output

STAGES = ("classes", "class_stability", "xor_stability", "convergence", "entropy", "fpga")
RAW = ROOT / "data" / "raw"


def device_id_of(device: int) -> bytes:
    return hashlib.sha256(f"device-{device}".encode()).digest()[:16]


def class_secret(kind: str, device: int, n: int):
    """Device secret of a class. The key of class P is the key of the mask, so
    that the classes differ only in where the keyed function sits."""
    if kind == "X":
        return rng_for("secret", "X", device).integers(0, 2, size=n, dtype=np.uint8)
    if kind == "L":
        return random_invertible_gf2(n, rng_for("secret", "L", device))
    if kind == "P":
        return new_device_key(rng_for("key", device))
    return None


def public_reference(n: int, k: int) -> XorArbiterPUF:
    """A fixed k-XOR arbiter model that is the same for every device and is
    known to the adversary. It replaces the PUF in the substitution control."""
    return XorArbiterPUF(n, k, rng_for("public-reference", k), 0.0)


# --------------------------------------------------------------------------
# controls applied to published obfuscation classes
# --------------------------------------------------------------------------
def classes_unit(kind: str, variant: str, device: int, n: int, c: dict, a: dict) -> list[dict]:
    budgets, n_test, k = c["budgets"], a["n_test"], c["base_k"]
    max_n = max(budgets)
    main_n = max(a["budgets"])                 # challenges and test set of the main run
    C = unique_challenges(main_n + n_test, n,
                          rng_for("challenges", device, "uniform", main_n, n_test))
    C = C[np.r_[0:max_n, main_n:main_n + n_test]]
    assert count_duplicates(C) == 0
    base = (public_reference(n, k) if variant == "substituted"
            else XorArbiterPUF(n, k, rng_for("weights", k, device), 0.0))
    puf = ObfuscatedPUF(base, kind, class_secret(kind, device, n), device_id_of(device), c["hash_bits"])
    inner = puf.internal(C)
    y = puf.eval(C, internal=inner)
    phi = parity_features(C)
    rows = []

    def record(attack, N, res, note=""):
        rows.append({"class": kind, "variant": variant, "device": device, "base_k": k,
                     "budget": N, "attack": attack, "n_test": n_test,
                     "val_acc": round(res["val_acc"], 6),
                     "test_acc": round(float(np.mean(res["test_pred"] == y[max_n:])), 6),
                     "majority_baseline": round(float(max(y[max_n:].mean(),
                                                          1 - y[max_n:].mean())), 6),
                     "seconds": round(res["seconds"], 3), "epochs": res["epochs"],
                     "restarts": res["restarts"], "note": note})

    for N in budgets:
        n_val = max(1, int(round(a["val_fraction"] * N)))
        n_tr = N - n_val
        for atk in c["attacks"]:
            seed = int_seed("fit", "classes", kind, variant, device, N, atk)
            record(atk, N, fit_predict(atk, phi[:n_tr], y[:n_tr], phi[n_tr:N], y[n_tr:N],
                                       phi[max_n:], seed, k))
        if variant == "full" and kind != "H":
            # known-secret diagnostic: the adversary computes the internal
            # challenges and runs the attack matched to the base PUF
            phi_in = parity_features(inner)
            seed = int_seed("fit", "classes", kind, "known", device, N)
            record("known_secret", N,
                   fit_predict("lr_xor", phi_in[:n_tr], y[:n_tr], phi_in[n_tr:N], y[n_tr:N],
                               phi_in[max_n:], seed, k), "product LR on internal challenges")
        if variant == "substituted" and kind == "H":
            # class H has no secret besides the PUF: with a public function in
            # its place the adversary evaluates the pipeline directly
            t0 = time.perf_counter()
            direct = ObfuscatedPUF(public_reference(n, k), "H", None, device_id_of(device),
                                   c["hash_bits"]).eval(C[max_n:])
            record("direct_evaluation", N,
                   {"val_acc": 1.0, "test_pred": direct, "seconds": time.perf_counter() - t0,
                    "epochs": 0, "restarts": 0}, "public pipeline evaluated, no learning")
    return rows


def stage_classes(cfg, jobs):
    c, a = cfg["classes"], cfg["attack"]
    out = Parallel(n_jobs=jobs, verbose=5)(
        delayed(classes_unit)(kind, variant, d, cfg["n_stages"], c, a)
        for kind in c["kinds"] for variant in ("full", "substituted")
        for d in range(c["devices"]))
    return {"classes_raw.csv": [r for rows in out for r in rows]}


# --------------------------------------------------------------------------
# what repeated queries reveal for each class (arbiter base)
# --------------------------------------------------------------------------
def class_stability_unit(kind: str, device: int, n: int, s: dict) -> list[dict]:
    N, n_test, nu, T = s["budget"], s["n_test"], s["noisiness"], s["repeats"]
    C = unique_challenges(N + n_test, n, rng_for("cstab", "challenges", device))
    base = XorArbiterPUF(n, 1, rng_for("weights", 1, device), nu)
    puf = ObfuscatedPUF(base, kind, class_secret(kind, device, n), device_id_of(device), s["hash_bits"])
    inner = puf.internal(C)
    rng = rng_for("cstab", "noise", kind, device)
    first = puf.eval(C[:N], rng, internal=inner[:N])
    z = np.zeros(N, dtype=bool)
    for _ in range(T - 1):
        z |= puf.eval(C[:N], rng, internal=inner[:N]) != first
    z = z.astype(np.uint8)
    truth = puf.eval(C[N:], internal=inner[N:])          # noise-free response of the construction
    w_true = base.weights[0]
    t0 = time.perf_counter()
    if kind == "H":
        phi_sub = parity_features(inner.reshape(-1, n)).reshape(len(C), s["hash_bits"], n + 1)
        w_hat, info = fit_hashed_stability(phi_sub[:N], z, int_seed("cstab", "fit", kind, device))
        train_acc, test_acc = [], []
        for sgn in (1.0, -1.0):
            bits_tr = (phi_sub[:N].astype(np.float64) @ (sgn * w_hat) > 0).astype(np.uint8)
            bits_te = (phi_sub[N:].astype(np.float64) @ (sgn * w_hat) > 0).astype(np.uint8)
            train_acc.append(float(np.mean(hash_output(C[:N], bits_tr) == first)))
            test_acc.append(float(np.mean(hash_output(C[N:], bits_te) == truth)))
        pick = int(np.argmax(train_acc))                 # sign chosen on the adversary's CRPs
        response_acc, sign_free = test_acc[pick], max(test_acc)
        cos = float(abs(w_hat @ w_true) / (np.linalg.norm(w_hat) * np.linalg.norm(w_true)))
    else:
        # the adversary knows only the public challenge, so the surrogate is
        # fitted on the parity features of c, not of the internal challenge
        phi = parity_features(C)
        w_hat, info = fit_stability_model(phi[:N], z, int_seed("cstab", "fit", kind, device),
                                          return_info=True)
        info = {**info, "epochs": -1, "restarts": RELIABILITY_CONFIG["restarts"]}
        pred = (phi[N:].astype(np.float64) @ w_hat > 0).astype(np.uint8)
        sign_free = sign_free_agreement(pred, truth)
        # one global sign bit, chosen on the adversary's own first answers
        pred_tr = (phi[:N].astype(np.float64) @ w_hat > 0).astype(np.uint8)
        flip = np.mean(pred_tr == first) < 0.5
        response_acc = float(np.mean((pred ^ np.uint8(flip)) == truth))
        w_eff = w_true.copy()
        if kind == "X":                                  # c XOR kappa flips the signs of parity features
            w_eff = w_true * parity_features((puf.secret[None, :]))[0]
        cos = float(abs(w_hat @ w_eff) / (np.linalg.norm(w_hat) * np.linalg.norm(w_eff)))
    return [{"class": kind, "device": device, "budget": N, "repeats": T, "queries": N * T,
             "noisiness": nu, "unstable_fraction": round(float(z.mean()), 6),
             "response_acc": round(response_acc, 6), "sign_free_acc": round(sign_free, 6),
             "abs_cosine": round(cos, 6), "val_loss": round(info["val_loss"], 6),
             "seconds": round(time.perf_counter() - t0, 2)}]


def stage_class_stability(cfg, jobs):
    s = cfg["class_stability"]
    out = Parallel(n_jobs=jobs, verbose=5)(
        delayed(class_stability_unit)(kind, d, cfg["n_stages"], s)
        for kind in s["kinds"] for d in range(s["devices"]))
    return {"class_stability_raw.csv": [r for rows in out for r in rows]}


# --------------------------------------------------------------------------
# stability attack on the masked k-XOR arbiter PUF
# --------------------------------------------------------------------------
def xor_stability_unit(k: int, device: int, n: int, s: dict) -> list[dict]:
    budgets, n_test, nu, T = s["budgets"], s["n_test"], s["noisiness"], s["repeats"]
    max_n = max(budgets)
    C = unique_challenges(max_n + n_test, n, rng_for("xstab", "challenges", k, device))
    name = "mask_apuf" if k == 1 else f"mask_xor{k}"
    masked, base, _ = build(name, n, device, nu)
    z = unstable_indicator(masked, C[:max_n], T, rng_for("xstab", "noise", k, device))
    z_base = unstable_indicator(base, C[:max_n], T, rng_for("xstab", "noise", k, device))
    phi = parity_features(C)
    hidden = base.eval(C[max_n:])                         # noise-free PUF bit, never seen
    mask_te = masked.mask(C[max_n:])
    masked_truth = hidden ^ mask_te                       # noise-free masked response
    # a handful of the adversary's own noisy masked answers, used only after
    # a key compromise to fix the one sign bit that stability leaves open
    sign_n = s["sign_crps"]
    own = masked.eval(C[:sign_n], rng_for("xstab", "own", k, device))
    own_unmasked = own ^ masked.mask(C[:sign_n])
    rows = []
    for N in budgets:
        t0 = time.perf_counter()
        W, info = fit_xor_stability(phi[:N], z[:N], k, int_seed("xstab", "fit", k, device, N))
        seconds = time.perf_counter() - t0
        pred = xor_model_bits(W, phi[max_n:])
        flip = np.mean(xor_model_bits(W, phi[:sign_n]) == own_unmasked) < 0.5
        with_key = (pred ^ np.uint8(flip)) ^ mask_te
        rows.append({
            "k": k, "device": device, "budget": N, "repeats": T, "queries": N * T,
            "noisiness": nu, "unstable_fraction": round(float(z[:N].mean()), 6),
            "mask_cancels": bool(np.array_equal(z, z_base)),
            # (1) recovery of the PUF model up to one global sign
            "model_agreement_sign_free": round(sign_free_agreement(pred, hidden), 6),
            "min_matched_cosine": round(min(matched_cosines(W, base.weights)), 6),
            # (2) prediction of masked responses without the key
            "masked_response_acc_no_key": round(float(np.mean(pred == masked_truth)), 6),
            # (3) prediction of masked responses after a key compromise
            "masked_response_acc_with_key": round(float(np.mean(with_key == masked_truth)), 6),
            "sign_crps": sign_n, "val_loss": round(info["val_loss"], 6),
            "epochs": info["epochs"], "restarts": info["restarts"],
            "seconds": round(seconds, 2)})
    return rows


def stage_xor_stability(cfg, jobs):
    s = cfg["xor_stability"]
    out = Parallel(n_jobs=jobs, verbose=5)(
        delayed(xor_stability_unit)(k, d, cfg["n_stages"], s)
        for k in s["k"] for d in range(s["devices"]))
    return {"xor_stability_raw.csv": [r for rows in out for r in rows]}


# --------------------------------------------------------------------------
# convergence and cost of the modelling attacks (seeds of the main run)
# --------------------------------------------------------------------------
def convergence_unit(name: str, device: int, n: int, a: dict, v: dict) -> list[dict]:
    budgets, n_test = a["budgets"], a["n_test"]
    max_n = max(budgets)
    C = unique_challenges(max_n + n_test, n,
                          rng_for("challenges", device, "uniform", max_n, n_test))
    puf, base, k = build(name, n, device, 0.0)
    y = puf.eval(C)
    phi = parity_features(C)
    k_model = k if k > 0 else 2
    rows = []
    for N in budgets:
        n_val = max(1, int(round(a["val_fraction"] * N)))
        n_tr = N - n_val
        for atk in v["attacks"]:
            seed = int_seed("fit", "attack", name, device, "uniform", "0.0", N, atk)
            res = fit_predict(atk, phi[:n_tr], y[:n_tr], phi[n_tr:N], y[n_tr:N], phi[max_n:],
                              seed, k_model)
            rows.append({"construction": name, "device": device, "budget": N, "n_train": n_tr,
                         "attack": atk, "val_acc": round(res["val_acc"], 6),
                         "test_acc": round(float(np.mean(res["test_pred"] == y[max_n:])), 6),
                         "epochs": res["epochs"], "iterations": res["iterations"],
                         "restarts": res["restarts"],
                         "hit_epoch_limit": res["hit_epoch_limit"],
                         "seconds": round(res["seconds"], 3)})
    return rows


def stage_convergence(cfg, jobs):
    a, v = cfg["attack"], cfg["convergence"]
    out = Parallel(n_jobs=jobs, verbose=5)(
        delayed(convergence_unit)(name, d, cfg["n_stages"], a, v)
        for name in v["constructions"] for d in range(v["devices"]))
    return {"convergence_raw.csv": [r for rows in out for r in rows]}


# --------------------------------------------------------------------------
# conditional min-entropy of the key given the helper data
# --------------------------------------------------------------------------
def stage_entropy(cfg, jobs):
    rows = []
    for code in cfg["ecc"]["codes"]:
        for p in cfg["entropy"]["bias_p"]:
            b = key_entropy_bounds(code["n"], code["k"], code["t"], code["bits_per_symbol"],
                                   code["blocks"], p)
            rows.append({"code": code["name"], "bias_p": p, "nominal_key_bits": 256,
                         "secret_bits": b["secret_bits"],
                         "entropy_lower": round(b["entropy_lower"], 4),
                         "entropy_upper": round(b["entropy_upper"], 4),
                         "naive_lower": round(b["naive_lower"], 4),
                         "reading_min_entropy": round(b["reading_min_entropy"], 4),
                         "helper_redundancy_bits": b["helper_redundancy_bits"]})
    return {"entropy_bounds.csv": rows}


# --------------------------------------------------------------------------
# ablation on measured FPGA instances
# --------------------------------------------------------------------------
def _load_bits(file: str, pool: int):
    z = np.load(RAW / file, allow_pickle=False)
    c = (z["challenges"][:pool] < 0).astype(np.uint8)
    r = (z["responses"][:pool].reshape(pool, -1)[:, 0] < 0).astype(np.uint8)
    return c, r


def fpga_instances(f: dict) -> list[dict]:
    inst = [{"name": "top", "kind": "arbiter", "files": [f["top"]], "k": 1}]
    inst += [{"name": f"b{i}", "kind": "arbiter", "files": [x], "k": 1}
             for i, x in enumerate(f["chains"])]
    inst += [{"name": f"b{i}^b{j}", "kind": "xor2", "files": [f["chains"][i], f["chains"][j]], "k": 2}
             for i, j in combinations(range(len(f["chains"])), 2)]
    return inst[:f["max_instances"]] if f.get("max_instances") else inst


def fpga_unit(inst: dict, f: dict) -> list[dict]:
    pool, n_test, budgets = f["pool"], f["n_test"], f["budgets"]
    c, r = _load_bits(inst["files"][0], pool)
    for extra in inst["files"][1:]:
        c2, r2 = _load_bits(extra, pool)
        assert np.array_equal(c, c2), "chains of a software XOR must share their challenges"
        r = r ^ r2
    _, first = np.unique(np.packbits(c, axis=1), axis=0, return_index=True)
    keep = np.sort(first)
    c, r = c[keep], r[keep]
    perm = rng_for("fpga", inst["name"], "split").permutation(len(c))
    max_n = max(budgets)
    idx_te, idx_pool = perm[:n_test], perm[n_test:n_test + max_n]
    key = new_device_key(rng_for("fpga", inst["name"], "key"))
    mask = hmac_mask(key, c, 1, hashlib.sha256(inst["name"].encode()).digest()[:16])[:, 0]
    targets = {"measured": r, "mask_only": mask, "masked": r ^ mask}
    phi = parity_features(c)
    rows = []

    def record(variant, atk, N, res, yte):
        rows.append({"instance": inst["name"], "kind": inst["kind"], "stages": c.shape[1],
                     "variant": variant, "budget": N, "attack": atk, "n_test": n_test,
                     "val_acc": round(res["val_acc"], 6),
                     "test_acc": round(float(np.mean(res["test_pred"] == yte)), 6),
                     "fraction_of_ones": round(float(r.mean()), 6),
                     "duplicates_dropped": int(pool - keep.size),
                     "seconds": round(res["seconds"], 2)})

    for variant, y in targets.items():
        for N in budgets:
            n_val = int(round(0.2 * N))
            tr, va = idx_pool[:N - n_val], idx_pool[N - n_val:N]
            for atk in f["attacks"]:
                res = fit_predict(atk, phi[tr], y[tr], phi[va], y[va], phi[idx_te],
                                  int_seed("fpga", inst["name"], variant, N, atk), inst["k"])
                record(variant, atk, N, res, y[idx_te])
            if variant == "masked":
                y_or = y ^ mask                       # key known: the measured responses again
                assert np.array_equal(y_or, r)
                atk = "lr" if inst["k"] == 1 else "lr_xor"
                res = fit_predict(atk, phi[tr], y_or[tr], phi[va], y_or[va], phi[idx_te],
                                  int_seed("fpga", inst["name"], "oracle", N), inst["k"])
                record("key_known", atk, N, res, y_or[idx_te])
    return rows


def stage_fpga(cfg, jobs):
    f = cfg["fpga"]
    missing = [x for x in [f["top"], *f["chains"]] if not (RAW / x).exists()]
    if missing:
        raise SystemExit("missing public data files: " + ", ".join(missing)
                         + "\nrun  python scripts/download_public_data.py  first")
    insts = fpga_instances(f)
    out = Parallel(n_jobs=jobs, verbose=5)(delayed(fpga_unit)(i, f) for i in insts)
    meta = {"source": "Zenodo record 10.5281/zenodo.5221305 (pypuf data), measured by Aghaie and Moradi",
            "instances": [{k: v for k, v in i.items()} for i in insts],
            "files_sha256": {x: hashlib.sha256((RAW / x).read_bytes()).hexdigest()
                             for x in [f["top"], *f["chains"]]},
            "measurements_per_challenge": 1,
            "note": "2-XOR instances are XORs, computed in software, of the measured bits of two "
                    "chains that answered the same challenges. One measurement per challenge, "
                    "so no stability attack is possible on these data."}
    return {"fpga_raw.csv": [r for rows in out for r in rows], "fpga_meta.json": meta}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/extensions_full.json")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ext = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    base_cfg = json.loads((ROOT / ext["base_config"]).read_text(encoding="utf-8"))
    cfg = {**base_cfg, **ext}                 # extension stages also read attack and ecc settings
    out = Path(args.out) if args.out else ROOT / "experiments" / "results" / cfg["profile"]
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest_extensions.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {"stages": {}})
    manifest.update({"master_seed": MASTER_SEED, "config_file": args.config, "config": ext,
                     "attack_config": ATTACK_CONFIG, "reliability_config": RELIABILITY_CONFIG,
                     "joint_stability_config": JOINT_CONFIG, "versions": versions(),
                     "jobs": args.jobs})
    fns = {"classes": stage_classes, "class_stability": stage_class_stability,
           "xor_stability": stage_xor_stability, "convergence": stage_convergence,
           "entropy": stage_entropy, "fpga": stage_fpga}
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
