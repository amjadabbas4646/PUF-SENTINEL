"""E11: the masking ablation on public FPGA measurements.

Data (see DATA_PROVENANCE.md, fetched by scripts/download_public_data.py)
    fpga_apuf   AM21_top.npz, one 64-stage arbiter PUF on FPGA, 1,000,000 CRPs
    fpga_xor4   MTZAA20_4XOR_..., one 64-stage 4-XOR arbiter PUF on FPGA,
                1,000,000 CRPs
Each file holds ONE device and ONE measurement per challenge. The replicates
below are therefore random splits of the same device's CRPs. They show the
variation caused by the choice of training and test CRPs, not the variation
between devices, and no device-level inference is drawn from them.

For each data set the same attacks are run on
    measured    the measured responses
    masked      measured responses XOR the HMAC mask under a simulated key
    oracle      the masked responses after removing the mask with the key
                (bit-identical to the measured responses by construction; it
                is run, not assumed)

Encoding: the files store challenges and responses in {-1, +1}. They are
mapped to bits by bit = 1 if value < 0 else 0, the pypuf convention.

Usage:  python experiments/run_hardware.py [--jobs 6] [--smoke] [--out DIR]
Output: experiments/results/<profile>/hardware_raw.csv, hardware_meta.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from joblib import Parallel, delayed

from attacks.models import fit_predict
from common.seeding import int_seed, rng_for
from puf_models.arbiter import parity_features
from puf_models.mask import hmac_mask, new_device_key

RAW = ROOT / "data" / "raw"
DATASETS = {
    "fpga_apuf": {"file": "AM21_top.npz", "k": 1, "attacks": ["lr", "mlp", "lr_xor"],
                  "budgets": [1000, 10000, 100000]},
    "fpga_xor4": {"file": "MTZAA20_4XOR_64bit_LUT_2239B_attacking_1M.txt.npz", "k": 4,
                  "attacks": ["lr", "mlp", "lr_xor"], "budgets": [10000, 100000, 300000]},
}
SMOKE_BUDGETS = {"fpga_apuf": [1000, 5000], "fpga_xor4": [5000, 20000]}
N_TEST, SPLITS, POOL = 10000, 5, 320000


def load(name: str):
    z = np.load(RAW / DATASETS[name]["file"], allow_pickle=False)
    c = (z["challenges"][:POOL] < 0).astype(np.uint8)
    r = (z["responses"][:POOL].reshape(POOL, -1)[:, 0] < 0).astype(np.uint8)
    packed = np.packbits(c, axis=1)
    _, first = np.unique(packed, axis=0, return_index=True)
    keep = np.sort(first)                       # drop repeated challenges
    return c[keep], r[keep], POOL - keep.size


def unit(name: str, split: int, budgets: list[int]) -> list[dict]:
    spec = DATASETS[name]
    c, r, _ = load(name)
    perm = rng_for("hardware", name, "split", split).permutation(len(c))
    max_n = max(budgets)
    idx_te, idx_pool = perm[:N_TEST], perm[N_TEST:N_TEST + max_n]
    key = new_device_key(rng_for("hardware", name, "key"))
    mask = hmac_mask(key, c, 1, hashlib.sha256(name.encode()).digest()[:16])[:, 0]
    targets = {"measured": r, "masked": r ^ mask}
    targets["oracle"] = targets["masked"] ^ mask
    assert np.array_equal(targets["oracle"], r)
    phi = parity_features(c)
    rows = []
    for variant, y in targets.items():
        yte = y[idx_te]
        for N in budgets:
            n_val = int(round(0.2 * N))
            tr, va = idx_pool[:N - n_val], idx_pool[N - n_val:N]
            for atk in spec["attacks"]:
                if variant == "oracle" and atk != "lr_xor":
                    continue
                res = fit_predict(atk, phi[tr], y[tr], phi[va], y[va], phi[idx_te],
                                  int_seed("hardware", name, variant, split, N, atk),
                                  spec["k"])
                rows.append({"dataset": name, "variant": variant, "split": split,
                             "budget": N, "attack": atk, "n_test": N_TEST,
                             "val_acc": round(res["val_acc"], 6),
                             "test_acc": round(float(np.mean(res["test_pred"] == yte)), 6),
                             "majority_baseline": round(float(max(yte.mean(), 1 - yte.mean())), 6),
                             "seconds": round(res["seconds"], 2)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    profile = "smoke" if args.smoke else "full"
    out = Path(args.out) if args.out else ROOT / "experiments" / "results" / profile
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    splits = 2 if args.smoke else SPLITS
    budgets = SMOKE_BUDGETS if args.smoke else {k: v["budgets"] for k, v in DATASETS.items()}
    res = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(unit)(name, s, budgets[name]) for name in DATASETS for s in range(splits))
    rows = [r for rr in res for r in rr]
    with open(out / "hardware_raw.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    meta = {}
    for name, spec in DATASETS.items():
        c, r, dropped = load(name)
        meta[name] = {"file": spec["file"],
                      "sha256": hashlib.sha256((RAW / spec["file"]).read_bytes()).hexdigest(),
                      "crps_read": POOL, "duplicate_challenges_dropped": int(dropped),
                      "crps_used_pool": int(len(c)), "fraction_of_ones": float(r.mean()),
                      "devices": 1, "measurements_per_challenge": 1,
                      "splits": splits, "n_test": N_TEST, "budgets": budgets[name]}
    (out / "hardware_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
