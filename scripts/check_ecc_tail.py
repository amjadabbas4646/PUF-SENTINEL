"""Independent re-check of one rare-event point of E7.

In the main run RS(48,16,16) at a bit error rate of 1 % showed 1 failure in
200,000 error patterns while the exact formula gives 5.1e-8 (expected count
0.01). An observation of at least one failure then has probability 0.01. This
script draws a fresh, much larger sample with a different seed to decide
whether the formula or the sampler is wrong, or whether the main-run event was
a chance occurrence.

Usage:  python scripts/check_ecc_tail.py
Output: experiments/results/ecc_tail_check.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy import stats

from common.seeding import rng_for
from error_correction.error_models import analytic_failure_iid, sample_errors_batch

n, t, u, p = 48, 16, 8, 0.01
exact = analytic_failure_iid("RS", n, t, u, 1, p)
rng = rng_for("ecc", "tail-check", "RS(48,16,16)", "0.01")
trials, fails = 0, 0
for _ in range(200):                                  # 200 x 50,000 = 10,000,000 bit-level patterns
    E = sample_errors_batch("iid", p, n * u, 50000, rng)
    w = E.reshape(50000, n, u).any(axis=2).sum(axis=1)
    fails += int((w > t).sum())
    trials += 50000
expected = exact * trials
p_two_sided = float(stats.binomtest(fails, trials, exact).pvalue)
out = {"code": "RS(48,16,16)", "ber": p, "exact_failure": exact, "trials": trials,
       "failures": fails, "expected_failures": expected, "binomial_test_p": p_two_sided,
       "main_run": {"trials": 200000, "failures": 1,
                    "p_at_least_one": float(1 - (1 - exact) ** 200000)}}
(ROOT / "experiments" / "results" / "ecc_tail_check.json").write_text(json.dumps(out, indent=1),
                                                                      encoding="utf-8")
print(json.dumps(out, indent=1))
