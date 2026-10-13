"""Turn raw experiment outputs into summaries, LaTeX tables and the numeric
macros used in the manuscript text.

Nothing here is typed by hand: every number written to tables/ comes from a
file under experiments/results/<profile>/.

Usage:  python experiments/analyze.py --profile full
Output: experiments/results/<profile>/summary/*.csv and *.json
        tables/*.tex (only for the full profile, or with --tables-dir)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from common.seeding import rng_for
from common.stats import (mean_t_interval, one_sample_t, tost_one_sample,
                          wilson_interval)


def floor_pct(x: float, d: int = 2) -> str:
    """Percent value rounded DOWN, for thresholds that must not be overstated."""
    import math
    return f"{math.floor(100 * x * 10 ** d + 1e-9) / 10 ** d:.{d}f}"

LABEL = {"apuf": "Arbiter", "xor2": "2-XOR arbiter", "mask_only": "Mask only",
         "mask_apuf": "Masked arbiter", "mask_xor2": "Masked 2-XOR (full)"}
ATTACK_LABEL = {"lr": "LR", "svm": "SVM", "rf": "RF", "gb": "GB", "mlp": "MLP",
                "lr_xor": "Product LR", "best_val": "Selected"}
COND_LABEL = {("uniform", 0.0): "uniform, noise free",
              ("markov", 0.0): "correlated, noise free",
              ("uniform", 0.1): "uniform, noisy"}


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k in ("device", "budget", "n_train", "n_val", "n_test", "split", "queries",
                     "repeats", "decoder_trials", "decoder_success", "weight_trials",
                     "weight_failures", "decoder_agrees_with_radius"):
                r[k] = int(v)
            elif k in ("noisiness", "val_acc", "test_acc", "test_acc_ideal", "seconds",
                       "ber", "unstable_fraction", "hidden_model_acc", "abs_cosine",
                       "majority_baseline", "failure"):
                r[k] = float(v)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}"


def summarise_attacks(rows: list[dict], attack_order: list[str]) -> list[dict]:
    """Per cell mean over devices with a t interval, plus the validation
    selected attack ('best_val')."""
    cells = defaultdict(dict)                 # (constr, dist, nu, budget, attack) -> {device: row}
    for r in rows:
        cells[(r["construction"], r["dist"], r["noisiness"], r["budget"], r["attack"])][r["device"]] = r
    out = []
    groups = defaultdict(list)
    for (c, d, nu, b, a) in cells:
        groups[(c, d, nu, b)].append(a)
    for (c, d, nu, b), attacks in groups.items():
        devices = sorted(cells[(c, d, nu, b, attacks[0])])
        chosen, chosen_ideal, picks = [], [], defaultdict(int)
        for dev in devices:
            cand = [a for a in attack_order if a in attacks]
            best = max(cand, key=lambda a: (cells[(c, d, nu, b, a)][dev]["val_acc"],
                                            -cand.index(a)))
            picks[best] += 1
            chosen.append(cells[(c, d, nu, b, best)][dev]["test_acc"])
            chosen_ideal.append(cells[(c, d, nu, b, best)][dev]["test_acc_ideal"])
        series = {a: ([cells[(c, d, nu, b, a)][dev]["test_acc"] for dev in devices],
                      [cells[(c, d, nu, b, a)][dev]["test_acc_ideal"] for dev in devices])
                  for a in attacks}
        series["best_val"] = (chosen, chosen_ideal)
        for a, (vals, ideal) in series.items():
            s = mean_t_interval(vals)
            n_test = cells[(c, d, nu, b, attacks[0])][devices[0]]["n_test"]
            out.append({"construction": c, "dist": d, "noisiness": nu, "budget": b,
                        "attack": a, "devices": s["n"], "mean": s["mean"], "sd": s["sd"],
                        "ci_lo": s["lo"], "ci_hi": s["hi"], "min": min(vals), "max": max(vals),
                        "mean_vs_ideal": float(np.mean(ideal)), "n_test": n_test,
                        "selected_counts": json.dumps(dict(picks)) if a == "best_val" else "",
                        "per_device": json.dumps(vals)})
    return out


def chance_tests(summary: list[dict], margin: float) -> list[dict]:
    out = []
    for s in summary:
        if not s["construction"].startswith("mask_") or s["attack"].startswith("oracle"):
            continue
        vals = json.loads(s["per_device"])
        t = one_sample_t(vals, 0.5)
        eq = tost_one_sample(vals, 0.5, margin)
        correct = int(round(sum(vals) * s["n_test"]))
        total = len(vals) * s["n_test"]
        w_lo, w_hi = wilson_interval(correct, total)
        x = np.asarray(vals)
        brng = rng_for("analysis", "boot", s["construction"], s["dist"], str(s["noisiness"]),
                       s["budget"], s["attack"])
        boots = x[brng.integers(0, x.size, size=(10000, x.size))].mean(axis=1)
        b_lo, b_hi = np.quantile(boots, [0.025, 0.975])
        ci90 = mean_t_interval(vals, conf=0.90)
        binom_var = 0.25 / s["n_test"]
        out.append({"construction": s["construction"], "dist": s["dist"],
                    "noisiness": s["noisiness"], "budget": s["budget"], "attack": s["attack"],
                    "devices": s["devices"], "mean": s["mean"], "ci_lo": s["ci_lo"],
                    "ci_hi": s["ci_hi"], "t_p_two_sided": t["p"], "cohen_d": t["cohen_d"],
                    "tost_margin": margin, "tost_p": eq["p_tost"],
                    "equivalent_at_0.05": bool(eq["p_tost"] < 0.05),
                    "pooled_correct": correct, "pooled_total": total,
                    "pooled_acc": correct / total, "pooled_wilson_lo": w_lo,
                    "pooled_wilson_hi": w_hi,
                    "device_boot_lo": float(b_lo), "device_boot_hi": float(b_hi),
                    "device_sd": float(x.std(ddof=1)),
                    "variance_ratio_vs_binomial": float(x.var(ddof=1) / binom_var),
                    "smallest_margin_supported": float(max(abs(ci90["lo"] - 0.5),
                                                           abs(ci90["hi"] - 0.5)))})
    return out


def find(summary, **kw):
    hits = [s for s in summary if all(s[k] == v for k, v in kw.items())]
    if len(hits) != 1:
        raise KeyError(f"{kw} matched {len(hits)} rows")
    return hits[0]


def ci_cell(s, d: int = 1) -> str:
    """mean with the half width of the 95 % interval, in percent."""
    half = (s["ci_hi"] - s["ci_lo"]) / 2
    return f"{pct(s['mean'], d)} $\\pm$ {pct(half, d)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full")
    ap.add_argument("--tables-dir", default=None)
    args = ap.parse_args()
    res = ROOT / "experiments" / "results" / args.profile
    summ = res / "summary"
    summ.mkdir(exist_ok=True)
    tables = Path(args.tables_dir) if args.tables_dir else (
        ROOT / "tables" if args.profile == "full" else res / "tables")
    tables.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((res / "manifest.json").read_text(encoding="utf-8"))["config"]
    A = cfg["attack"]
    order = A["attacks"]
    margin = A["equivalence_margin"]
    bmax = max(A["budgets"])
    N = {}                                    # numbers for the manuscript text
    macros = {}

    # ------------------------------------------------------------ attacks
    raw = read_csv(res / "attack_raw.csv")
    summary = summarise_attacks(raw, order)
    write_csv(summ / "attack_summary.csv", summary)
    chance = chance_tests(summary, margin)
    write_csv(summ / "chance_tests.csv", chance)

    conds = [(c["dist"], c["noisiness"]) for c in A["conditions"]]
    u0 = ("uniform", 0.0)
    # Table: attack matrix at the largest budget, uniform noise-free
    cols = order + ["best_val"]
    lines = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             "Construction & " + " & ".join(ATTACK_LABEL[a] for a in cols) + " \\\\",
             "\\midrule"]
    for c in A["constructions"]:
        cells = [ci_cell(find(summary, construction=c, dist=u0[0], noisiness=u0[1],
                              budget=bmax, attack=a)) for a in cols]
        lines.append(f"{LABEL[c]} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    for c in ("mask_apuf", "mask_xor2"):
        a = "oracle_lr" if c == "mask_apuf" else "oracle_lr_xor"
        s = find(summary, construction=c, dist=u0[0], noisiness=u0[1], budget=bmax, attack=a)
        col = "lr" if c == "mask_apuf" else "lr_xor"
        cells = [ci_cell(s) if x == col else "" for x in cols]
        lines.append(f"{LABEL[c]}, key known & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tables / "tab_attack_matrix.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Table: ablation across the three conditions with the selected attack
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "Construction & Uniform & Correlated & Uniform, noisy \\\\", "\\midrule"]
    for c in A["constructions"]:
        cells = [ci_cell(find(summary, construction=c, dist=d, noisiness=nu, budget=bmax,
                              attack="best_val")) for d, nu in conds]
        lines.append(f"{LABEL[c]} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    for c in ("mask_apuf", "mask_xor2"):
        a = "oracle_lr" if c == "mask_apuf" else "oracle_lr_xor"
        cells = [ci_cell(find(summary, construction=c, dist=d, noisiness=nu, budget=bmax,
                              attack=a)) for d, nu in conds]
        lines.append(f"{LABEL[c]}, key known & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tables / "tab_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Table: equivalence tests for the masked constructions (selected attack)
    lines = ["\\begin{tabular}{llccccc}", "\\toprule",
             "Construction & Condition & Mean & 95\\% CI ($t$) & 95\\% CI (bootstrap) & $p_{\\mathrm{TOST}}$ & VR \\\\",
             "\\midrule"]
    all_equiv = True
    for c in ("mask_only", "mask_apuf", "mask_xor2"):
        for d, nu in conds:
            t = [x for x in chance if x["construction"] == c and x["dist"] == d
                 and x["noisiness"] == nu and x["budget"] == bmax and x["attack"] == "best_val"][0]
            all_equiv &= t["equivalent_at_0.05"]
            p = "$<$0.001" if t["tost_p"] < 0.001 else f"{t['tost_p']:.3f}"
            lines.append(f"{LABEL[c]} & {COND_LABEL[(d, nu)]} & {pct(t['mean'], 2)} & "
                         f"[{pct(t['ci_lo'], 2)}, {pct(t['ci_hi'], 2)}] & "
                         f"[{pct(t['device_boot_lo'], 2)}, {pct(t['device_boot_hi'], 2)}] & {p} & "
                         f"{t['variance_ratio_vs_binomial']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tables / "tab_equivalence.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    masked_cells = [s for s in summary if s["construction"].startswith("mask_")
                    and not s["attack"].startswith("oracle") and s["attack"] != "best_val"]
    dev_vals = [v for s in masked_cells for v in json.loads(s["per_device"])]
    eq_cells = [x for x in chance if x["attack"] != "best_val" and x["budget"] == bmax]
    N["masked"] = {
        "cells": len(masked_cells), "fits": len(dev_vals),
        "cell_mean_min": min(s["mean"] for s in masked_cells),
        "cell_mean_max": max(s["mean"] for s in masked_cells),
        "device_min": min(dev_vals), "device_max": max(dev_vals),
        "binomial_sd_one_fit": 0.5 / np.sqrt(A["n_test"]),
        "selected_all_equivalent": bool(all_equiv),
        "single_attack_cells_at_max_budget": len(eq_cells),
        "single_attack_cells_equivalent": int(sum(x["equivalent_at_0.05"] for x in eq_cells)),
        "single_attack_cells_t_p_below_0.05": int(sum(x["t_p_two_sided"] < 0.05 for x in eq_cells)),
    }

    def g(c, a, b=bmax, cond=u0):
        return find(summary, construction=c, dist=cond[0], noisiness=cond[1], budget=b, attack=a)

    def first_budget(c, a, thr, cond=u0):
        for b in sorted(A["budgets"]):
            if g(c, a, b, cond)["mean"] >= thr:
                return b
        return None

    from scipy import stats as _sst
    prim = [x for x in chance if x["attack"] == "best_val" and x["budget"] == bmax]
    ps = sorted(x["t_p_two_sided"] for x in eq_cells)
    holm_sig = 0
    for i, pv in enumerate(ps):                 # Holm step-down at level 0.05
        if pv <= 0.05 / (len(ps) - i):
            holm_sig += 1
        else:
            break
    sd_max = max(x["device_sd"] for x in prim)
    n_dev = prim[0]["devices"]
    prng = rng_for("analysis", "tost-power")
    sims = prng.normal(0.5, sd_max, size=(200000, n_dev))
    tcrit = _sst.t.ppf(0.95, n_dev - 1)
    half = tcrit * sims.std(axis=1, ddof=1) / np.sqrt(n_dev)
    power0 = float(np.mean(np.abs(sims.mean(axis=1) - 0.5) + half < margin))
    sims5 = sims + 0.005
    power5 = float(np.mean(np.abs(sims5.mean(axis=1) - 0.5) + half < margin))
    N["statistics"] = {
        "primary_cells": len(prim),
        "primary_max_tost_p": max(x["tost_p"] for x in prim),
        "primary_device_sd_min": min(x["device_sd"] for x in prim),
        "primary_device_sd_max": sd_max,
        "binomial_sd_one_device": float(np.sqrt(0.25 / A["n_test"])),
        "variance_ratio_min": min(x["variance_ratio_vs_binomial"] for x in prim),
        "variance_ratio_max": max(x["variance_ratio_vs_binomial"] for x in prim),
        "smallest_margin_supported_max": max(x["smallest_margin_supported"] for x in prim),
        "single_attack_cells": len(eq_cells),
        "single_attack_uncorrected_significant": int(sum(pv < 0.05 for pv in ps)),
        "single_attack_holm_significant": holm_sig,
        "single_attack_min_p": ps[0],
        "tost_power_true_diff_0": power0, "tost_power_true_diff_0.5pp": power5,
        "power_note": "simulation, normal device accuracies, sd = largest observed primary sd",
    }

    N["apuf"] = {"lr_max": g("apuf", "lr")["mean"], "lr_1000": g("apuf", "lr", 1000)["mean"]
                 if 1000 in A["budgets"] else None,
                 "lr_first_95": first_budget("apuf", "lr", 0.95),
                 "lr_first_99": first_budget("apuf", "lr", 0.99),
                 "rf_max": g("apuf", "rf")["mean"], "gb_max": g("apuf", "gb")["mean"],
                 "mlp_max": g("apuf", "mlp")["mean"],
                 "lr_markov_max": g("apuf", "lr", cond=("markov", 0.0))["mean"],
                 "lr_noisy_max": g("apuf", "lr", cond=("uniform", 0.1))["mean"],
                 "lr_noisy_vs_ideal_max": g("apuf", "best_val", cond=("uniform", 0.1))["mean_vs_ideal"]}
    N["xor2"] = {a + "_max": g("xor2", a)["mean"] for a in order}
    N["xor2"].update({"best_max": g("xor2", "best_val")["mean"],
                      "lr_xor_first_95": first_budget("xor2", "lr_xor", 0.95),
                      "mlp_first_95": first_budget("xor2", "mlp", 0.95),
                      "best_markov_max": g("xor2", "best_val", cond=("markov", 0.0))["mean"],
                      "best_noisy_max": g("xor2", "best_val", cond=("uniform", 0.1))["mean"],
                      "best_noisy_vs_ideal_max": g("xor2", "best_val", cond=("uniform", 0.1))["mean_vs_ideal"],
                      "selected_counts": g("xor2", "best_val")["selected_counts"]})
    N["full"] = {"best_max": g("mask_xor2", "best_val")["mean"],
                 "best_lo": g("mask_xor2", "best_val")["ci_lo"],
                 "best_hi": g("mask_xor2", "best_val")["ci_hi"],
                 "oracle_max": g("mask_xor2", "oracle_lr_xor")["mean"],
                 "oracle_apuf_max": g("mask_apuf", "oracle_lr")["mean"]}
    # identical data, identical fit: oracle accuracy must equal the unmasked attack
    N["oracle_check"] = {
        "mask_apuf_oracle_minus_apuf_lr": g("mask_apuf", "oracle_lr")["mean"] - g("apuf", "lr")["mean"],
        "note": "oracle fits use their own seed, so product LR may differ slightly from the unmasked run"}

    # ------------------------------------------------------------ legacy
    if (res / "legacy_raw.csv").exists():
        leg = summarise_attacks(read_csv(res / "legacy_raw.csv"), order)
        write_csv(summ / "legacy_summary.csv", leg)
        lb = max(s["budget"] for s in leg)
        sel = [s for s in leg if s["attack"] == "best_val" and s["budget"] == lb]
        N["legacy"] = {"budget": lb, "devices": sel[0]["devices"],
                       **{s["construction"]: {"mean": s["mean"], "lo": s["ci_lo"],
                                              "hi": s["ci_hi"]} for s in sel}}

    # ------------------------------------------------------------ XOR scaling
    if (res / "xor_scaling_raw.csv").exists():
        xs = summarise_attacks(read_csv(res / "xor_scaling_raw.csv"),
                               cfg["xor_scaling"]["attacks"])
        write_csv(summ / "xor_scaling_summary.csv", xs)
        budgets = sorted(cfg["xor_scaling"]["budgets"])
        lines = ["\\begin{tabular}{c" + "cc" * len(budgets) + "}", "\\toprule",
                 " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{b:,} CRPs}}".replace(",", "\\,")
                                    for b in budgets) + " \\\\",
                 "$k$ & " + " & ".join("MLP & Product LR" for _ in budgets) + " \\\\", "\\midrule"]
        N["xor_scaling"] = {}
        for k in cfg["xor_scaling"]["k"]:
            name = "apuf" if k == 1 else f"xor{k}"
            cells = []
            for b in budgets:
                for a in ("mlp", "lr_xor"):
                    s = find(xs, construction=name, budget=b, attack=a)
                    cells.append(ci_cell(s))
                    N["xor_scaling"][f"k{k}_{a}_{b}"] = {"mean": s["mean"], "min": s["min"],
                                                         "max": s["max"]}
            lines.append(f"{k} & " + " & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (tables / "tab_xor_scaling.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------ reliability
    if (res / "reliability_raw.csv").exists():
        rel = read_csv(res / "reliability_raw.csv")
        by = defaultdict(list)
        for r in rel:
            by[r["budget"]].append(r)
        rs = []
        for b in sorted(by):
            acc = [r["hidden_model_acc"] for r in by[b]]
            s = mean_t_interval(acc)
            rs.append({"budget": b, "queries": by[b][0]["queries"], "devices": s["n"],
                       "mean": s["mean"], "sd": s["sd"], "ci_lo": s["lo"], "ci_hi": s["hi"],
                       "min": min(acc), "max": max(acc),
                       "mean_abs_cosine": float(np.mean([r["abs_cosine"] for r in by[b]])),
                       "unstable_fraction": float(np.mean([r["unstable_fraction"] for r in by[b]])),
                       "per_device": json.dumps(acc)})
        write_csv(summ / "reliability_summary.csv", rs)
        N["reliability"] = {"mask_cancels_all": all(r["mask_cancels"] == "True" for r in rel),
                            "repeats": rel[0]["repeats"], "noisiness": rel[0]["noisiness"],
                            "rows": {str(r["budget"]): {k: r[k] for k in
                                                        ("mean", "min", "max", "ci_lo", "ci_hi",
                                                         "unstable_fraction", "queries")}
                                     for r in rs}}

    # ------------------------------------------------------------ quality
    q = json.loads((res / "quality_summary.json").read_text(encoding="utf-8"))
    lines = ["\\begin{tabular}{lcccccccc}", "\\toprule",
             "Construction & Uniformity & Uniqueness & Aliasing SD & $H_1$ & $H_\\infty$ & "
             "Flip rate & NIST f, r & Attack acc. \\\\", "\\midrule"]
    for c in cfg["quality"]["constructions"]:
        s = q[c]
        best = g(c, "best_val")["mean"] if c in A["constructions"] else float("nan")
        f = s["flip_rate"]
        lines.append(
            f"{LABEL[c]} & {s['uniformity']['mean']:.3f} $\\pm$ {s['uniformity']['sd']:.3f} & "
            f"{s['uniqueness']['value']:.3f} [{s['uniqueness']['boot_lo']:.3f}, "
            f"{s['uniqueness']['boot_hi']:.3f}] & {s['bit_aliasing_sd']:.3f} & "
            f"{s['shannon_marginal']['mean']:.3f} & {s['min_entropy_mcv']['mean']:.3f} & "
            f"{f['mean']:.2f} [{f['min']:.2f}, {f['max']:.2f}] & "
            f"{s['nist_frequency']['pass_proportion']:.2f}, {s['nist_runs']['pass_proportion']:.2f} & "
            f"{pct(best)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tables / "tab_quality.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    N["quality"] = {c: {"uniformity_mean": s["uniformity"]["mean"],
                        "uniformity_sd": s["uniformity"]["sd"],
                        "uniformity_min": s["uniformity"]["min"],
                        "uniformity_max": s["uniformity"]["max"],
                        "uniqueness": s["uniqueness"]["value"],
                        "uniq_lo": s["uniqueness"]["boot_lo"], "uniq_hi": s["uniqueness"]["boot_hi"],
                        "aliasing_sd": s["bit_aliasing_sd"],
                        "aliasing_sd_ideal": s["bit_aliasing_sd_expected_ideal"],
                        "shannon": s["shannon_marginal"]["mean"],
                        "min_entropy": s["min_entropy_mcv"]["mean"],
                        "flip_mean": s["flip_rate"]["mean"], "flip_min": s["flip_rate"]["min"],
                        "flip_max": s["flip_rate"]["max"],
                        "nist_freq": s["nist_frequency"], "nist_runs": s["nist_runs"],
                        "intra_ber": s["intra_ber"]["mean"]} for c, s in q.items()}
    N["quality"]["devices"] = cfg["quality"]["devices"]
    N["quality"]["challenges"] = cfg["quality"]["n_challenges"]
    N["quality"]["nist_sequences"] = q[cfg["quality"]["constructions"][0]]["nist_frequency"]["sequences"]

    # ------------------------------------------------------------ ECC
    ecc = read_csv(res / "ecc_raw.csv")
    design = json.loads((res / "ecc_design.json").read_text(encoding="utf-8"))
    es = []
    for r in ecc:
        f = r["weight_failures"] / r["weight_trials"]
        lo, hi = wilson_interval(r["weight_failures"], r["weight_trials"])
        dfail = 1 - r["decoder_success"] / r["decoder_trials"]
        dlo, dhi = wilson_interval(r["decoder_trials"] - r["decoder_success"], r["decoder_trials"])
        es.append({"code": r["code"], "model": r["model"], "ber": r["ber"],
                   "failure_weight": f, "weight_lo": lo, "weight_hi": hi,
                   "weight_trials": r["weight_trials"],
                   "failure_decoder": dfail, "decoder_lo": dlo, "decoder_hi": dhi,
                   "decoder_trials": r["decoder_trials"],
                   "decoder_agreement": r["decoder_agrees_with_radius"] / r["decoder_trials"],
                   "analytic_iid": r["analytic_failure_iid"]})
    write_csv(summ / "ecc_summary.csv", es)

    def fail(code, model, ber):
        return [e for e in es if e["code"] == code and e["model"] == model
                and abs(e["ber"] - ber) < 1e-9][0]

    def sci(x):
        if x == 0:
            return "0"
        if x >= 0.01:
            return f"{x:.3f}"
        m, e = f"{x:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"

    show = [b for b in (0.05, 0.10) if b in cfg["ecc"]["bers"]] or cfg["ecc"]["bers"][:2]
    lines = ["\\begin{tabular}{lcccccc" + "ccc" * len(show) + "}", "\\toprule",
             " & & & & & & & " + " & ".join(
                 f"\\multicolumn{{3}}{{c}}{{Failure at BER {int(round(100 * b))}\\%}}" for b in show) + " \\\\",
             "Code & $d$ & $t$ & Cells & Secret & Helper & BER$_{10^{-6}}$ & "
             + " & ".join("i.i.d. & Het. & Com." for _ in show) + " \\\\", "\\midrule"]
    N["ecc"] = {"decoder_agreement_min": min(e["decoder_agreement"] for e in es),
                "decoder_trials_total": int(sum(e["decoder_trials"] for e in es)), "codes": {}}
    for name, dsg in design.items():
        cells = [name.replace(" x", " $\\times$"), str(dsg["d"]), str(dsg["t"]),
                 str(dsg["puf_bits"]), str(dsg["secret_bits"]), str(dsg["helper_bits"]),
                 f"{floor_pct(dsg['max_ber_fail_1e-6'])}\\%"]
        N["ecc"]["codes"][name] = dict(dsg)
        for b in show:
            for m in cfg["ecc"]["models"]:
                e = fail(name, m, b)
                cells.append(sci(e["failure_weight"]))
                N["ecc"]["codes"][name][f"fail_{m}_{b}"] = e["failure_weight"]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tables / "tab_ecc.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Agreement of the simulated i.i.d. failure counts with the exact formula.
    # A z score is meaningful only where both expected counts are large, so
    # rare-event points are judged by an exact binomial test instead, with a
    # Bonferroni threshold over all i.i.d. points.
    from scipy import stats as _st
    iid = [e for e in es if e["model"] == "iid"]
    zs, ps = [], []
    for e in iid:
        a, n_tr = float(e["analytic_iid"]), e["weight_trials"]
        k_obs = int(round(e["failure_weight"] * n_tr))
        ps.append(float(_st.binomtest(k_obs, n_tr, min(max(a, 0.0), 1.0)).pvalue))
        if a * n_tr >= 10 and (1 - a) * n_tr >= 10:
            zs.append(abs(k_obs - a * n_tr) / np.sqrt(a * (1 - a) * n_tr))
    N["ecc"]["iid_points"] = len(iid)
    N["ecc"]["min_exact_binomial_p"] = min(ps)
    N["ecc"]["bonferroni_threshold"] = 0.05 / len(iid)
    N["ecc"]["max_abs_z_where_counts_are_large"] = float(max(zs)) if zs else float("nan")
    tail_path = res.parent / "ecc_tail_check.json"
    if tail_path.exists():
        N["ecc"]["tail_check"] = json.loads(tail_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------ overhead
    ov = json.loads((res / "overhead.json").read_text(encoding="utf-8"))
    N["overhead"] = {"hmac_us": ov["hmac_mask_us_per_challenge"]["median"],
                     "pbkdf2_us": ov["pbkdf2_mask_us_per_challenge"]["median"],
                     "ratio": ov["pbkdf2_over_hmac_ratio"],
                     "rep_ms": {k: v["rep_ms_median"] for k, v in ov["codes"].items()}}
    secs = defaultdict(list)
    for r in raw:
        if r["budget"] == bmax and r["construction"] in ("apuf", "xor2"):
            secs[r["attack"]].append(r["seconds"])
    N["overhead"]["attack_seconds_median_at_max_budget"] = {a: float(np.median(v))
                                                            for a, v in secs.items()}

    # ------------------------------------------------------------ hardware
    if (res / "hardware_raw.csv").exists():
        hw = read_csv(res / "hardware_raw.csv")
        meta = json.loads((res / "hardware_meta.json").read_text(encoding="utf-8"))
        by = defaultdict(list)
        for r in hw:
            by[(r["dataset"], r["variant"], r["budget"], r["attack"])].append(r)
        hs = []
        for (dset, var, b, a), rr in by.items():
            acc = [r["test_acc"] for r in rr]
            hs.append({"dataset": dset, "variant": var, "budget": b, "attack": a,
                       "splits": len(acc), "mean": float(np.mean(acc)),
                       "sd": float(np.std(acc, ddof=1)) if len(acc) > 1 else float("nan"),
                       "min": min(acc), "max": max(acc),
                       "majority_baseline": float(np.mean([r["majority_baseline"] for r in rr]))})
        write_csv(summ / "hardware_summary.csv", hs)
        hl = {"fpga_apuf": "FPGA arbiter", "fpga_xor4": "FPGA 4-XOR"}
        vl = {"measured": "measured", "masked": "masked", "oracle": "masked, key known"}
        lines = ["\\begin{tabular}{llrccc}", "\\toprule",
                 "Data set & Responses & CRPs & LR & MLP & Product LR \\\\", "\\midrule"]
        N["hardware"] = {"meta": meta, "rows": {}}
        for dset in meta:
            for var in ("measured", "masked", "oracle"):
                for b in meta[dset]["budgets"]:
                    cells = []
                    for a in ("lr", "mlp", "lr_xor"):
                        hit = [h for h in hs if (h["dataset"], h["variant"], h["budget"],
                                                 h["attack"]) == (dset, var, b, a)]
                        if hit:
                            cells.append(f"{pct(hit[0]['mean'])} ({pct(hit[0]['min'])}, {pct(hit[0]['max'])})")
                            N["hardware"]["rows"][f"{dset}|{var}|{b}|{a}"] = {
                                "mean": hit[0]["mean"], "min": hit[0]["min"], "max": hit[0]["max"]}
                        else:
                            cells.append("")
                    lines.append(f"{hl[dset]} & {vl[var]} & " + f"{b:,}".replace(",", "\\,")
                                 + " & " + " & ".join(cells) + " \\\\")
            lines.append("\\midrule")
        lines[-1] = "\\bottomrule"
        lines.append("\\end{tabular}")
        (tables / "tab_hardware.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------ macros
    def m(name, value):
        macros[name] = value

    m("NumDevices", str(A["devices"]))
    m("NumTest", f"{A['n_test']:,}".replace(",", "\\,"))
    m("BudgetMax", f"{bmax:,}".replace(",", "\\,"))
    m("EqMargin", pct(margin, 0))
    m("ApufLrMax", pct(N["apuf"]["lr_max"]))
    m("ApufLrFirstNinetyFive", f"{N['apuf']['lr_first_95']:,}".replace(",", "\\,") if N["apuf"]["lr_first_95"] else "n.a.")
    m("ApufLrFirstNinetyNine", f"{N['apuf']['lr_first_99']:,}".replace(",", "\\,") if N["apuf"]["lr_first_99"] else "n.a.")
    m("ApufRfMax", pct(N["apuf"]["rf_max"]))
    m("ApufGbMax", pct(N["apuf"]["gb_max"]))
    m("ApufMlpMax", pct(N["apuf"]["mlp_max"]))
    m("ApufLrMarkovMax", pct(N["apuf"]["lr_markov_max"]))
    m("ApufLrNoisyMax", pct(N["apuf"]["lr_noisy_max"]))
    m("ApufLrNoisyIdealMax", pct(N["apuf"]["lr_noisy_vs_ideal_max"]))
    m("XorLrMax", pct(N["xor2"]["lr_max"]))
    m("XorSvmMax", pct(N["xor2"]["svm_max"]))
    m("XorRfMax", pct(N["xor2"]["rf_max"]))
    m("XorGbMax", pct(N["xor2"]["gb_max"]))
    m("XorMlpMax", pct(N["xor2"]["mlp_max"]))
    m("XorProdMax", pct(N["xor2"]["lr_xor_max"]))
    m("XorBestMax", pct(N["xor2"]["best_max"]))
    m("XorBestMarkovMax", pct(N["xor2"]["best_markov_max"]))
    m("XorBestNoisyMax", pct(N["xor2"]["best_noisy_max"]))
    m("XorBestNoisyIdealMax", pct(N["xor2"]["best_noisy_vs_ideal_max"]))
    fb = N["xor2"]["lr_xor_first_95"]
    m("XorProdFirstNinetyFive", f"{fb:,}".replace(",", "\\,") if fb else "n.a.")
    m("FullBestMax", pct(N["full"]["best_max"], 2))
    m("FullBestLo", pct(N["full"]["best_lo"], 2))
    m("FullBestHi", pct(N["full"]["best_hi"], 2))
    m("FullOracleMax", pct(N["full"]["oracle_max"]))
    m("MaskApufOracleMax", pct(N["full"]["oracle_apuf_max"]))
    m("MaskedCells", str(N["masked"]["cells"]))
    m("MaskedFits", f"{N['masked']['fits']:,}".replace(",", "\\,"))
    m("MaskedCellMin", pct(N["masked"]["cell_mean_min"], 2))
    m("MaskedCellMax", pct(N["masked"]["cell_mean_max"], 2))
    m("MaskedDevMin", pct(N["masked"]["device_min"], 2))
    m("MaskedDevMax", pct(N["masked"]["device_max"], 2))
    m("MaskedSingleCells", str(N["masked"]["single_attack_cells_at_max_budget"]))
    m("MaskedSingleEquiv", str(N["masked"]["single_attack_cells_equivalent"]))
    m("MaskedSingleSig", str(N["masked"]["single_attack_cells_t_p_below_0.05"]))
    if "reliability" in N:
        rows = N["reliability"]["rows"]
        bs = sorted(int(b) for b in rows)
        m("RelRepeats", str(N["reliability"]["repeats"]))
        m("RelBudgetMin", f"{bs[0]:,}".replace(",", "\\,"))
        m("RelBudgetMax", f"{bs[-1]:,}".replace(",", "\\,"))
        m("RelAccMin", pct(rows[str(bs[0])]["mean"]))
        m("RelAccMax", pct(rows[str(bs[-1])]["mean"]))
        m("RelAccMaxWorst", pct(rows[str(bs[-1])]["min"]))
        m("RelUnstable", pct(rows[str(bs[-1])]["unstable_fraction"]))
    if "legacy" in N:
        m("LegacyDevices", str(N["legacy"]["devices"]))
        m("LegacyBudget", f"{N['legacy']['budget']:,}".replace(",", "\\,"))
        m("LegacyFullMean", pct(N["legacy"]["mask_xor2"]["mean"], 2))
        m("LegacyFullLo", pct(N["legacy"]["mask_xor2"]["lo"], 2))
        m("LegacyFullHi", pct(N["legacy"]["mask_xor2"]["hi"], 2))
    Qn = N["quality"]
    m("QualDevices", str(Qn["devices"]))
    m("QualChallenges", f"{Qn['challenges']:,}".replace(",", "\\,"))
    m("QualNistSeqs", str(Qn["nist_sequences"]))
    for c, tag in (("apuf", "Apuf"), ("xor2", "Xor"), ("mask_xor2", "Full")):
        if c in Qn:
            m(f"Qual{tag}Unif", f"{Qn[c]['uniformity_mean']:.3f}")
            m(f"Qual{tag}UnifMin", f"{Qn[c]['uniformity_min']:.3f}")
            m(f"Qual{tag}UnifMax", f"{Qn[c]['uniformity_max']:.3f}")
            m(f"Qual{tag}Uniq", f"{Qn[c]['uniqueness']:.3f}")
            m(f"Qual{tag}Shannon", f"{Qn[c]['shannon']:.3f}")
            m(f"Qual{tag}FlipMin", f"{Qn[c]['flip_min']:.2f}")
            m(f"Qual{tag}FlipMax", f"{Qn[c]['flip_max']:.2f}")
            m(f"Qual{tag}NistFreq", f"{Qn[c]['nist_freq']['pass_proportion']:.2f}")
            m(f"Qual{tag}Ber", pct(Qn[c]["intra_ber"]))
    m("EccDecoderTrials", f"{N['ecc']['decoder_trials_total']:,}".replace(",", "\\,"))
    m("EccAgreement", pct(N["ecc"]["decoder_agreement_min"], 1))
    m("EccWeightTrials", f"{cfg['ecc']['weight_trials']:,}".replace(",", "\\,"))
    m("OvHmacUs", f"{N['overhead']['hmac_us']:.1f}")
    m("OvPbkdfUs", f"{N['overhead']['pbkdf2_us']:.0f}")
    m("OvRatio", f"{N['overhead']['ratio']:.0f}")
    if "hardware" in N:
        hr = N["hardware"]["rows"]

        def hv(key, field="mean"):
            return pct(hr[key][field]) if key in hr else "n.a."
        ab = max(N["hardware"]["meta"]["fpga_apuf"]["budgets"])
        xb = max(N["hardware"]["meta"]["fpga_xor4"]["budgets"])
        m("HwApufBudget", f"{ab:,}".replace(",", "\\,"))
        m("HwXorBudget", f"{xb:,}".replace(",", "\\,"))
        m("HwApufLr", hv(f"fpga_apuf|measured|{ab}|lr"))
        m("HwApufMaskedLr", hv(f"fpga_apuf|masked|{ab}|lr"))
        m("HwApufOracle", hv(f"fpga_apuf|oracle|{ab}|lr_xor"))
        m("HwXorProd", hv(f"fpga_xor4|measured|{xb}|lr_xor"))
        m("HwXorMlp", hv(f"fpga_xor4|measured|{xb}|mlp"))
        m("HwXorLr", hv(f"fpga_xor4|measured|{xb}|lr"))
        m("HwXorMaskedProd", hv(f"fpga_xor4|masked|{xb}|lr_xor"))
        m("HwApufOnes", pct(N["hardware"]["meta"]["fpga_apuf"]["fraction_of_ones"]))
        m("HwXorOnes", pct(N["hardware"]["meta"]["fpga_xor4"]["fraction_of_ones"]))

    # further macros used in the running text
    if "xor_scaling" in N:
        xb = sorted(cfg["xor_scaling"]["budgets"])
        kmax = max(cfg["xor_scaling"]["k"])
        xsn = N["xor_scaling"]
        m("XsKmax", str(kmax))
        m("XsBudgetLow", f"{xb[0]:,}".replace(",", "\\,"))
        m("XsBudgetHigh", f"{xb[-1]:,}".replace(",", "\\,"))
        m("XsMaxMlpLow", pct(xsn[f"k{kmax}_mlp_{xb[0]}"]["mean"]))
        m("XsMaxMlpHigh", pct(xsn[f"k{kmax}_mlp_{xb[-1]}"]["mean"]))
        m("XsMaxProdHighMin", pct(xsn[f"k{kmax}_lr_xor_{xb[-1]}"]["min"]))
        m("XsMaxProdHighMax", pct(xsn[f"k{kmax}_lr_xor_{xb[-1]}"]["max"]))
    codes_n = N["ecc"]["codes"]
    best = max(codes_n, key=lambda c: codes_n[c]["max_ber_fail_1e-6"])
    worst_c = min(codes_n, key=lambda c: codes_n[c]["max_ber_fail_1e-6"])

    def plain(x):
        return f"{x:.4f}" if x < 0.01 else f"{x:.3f}"
    m("EccBestName", best.replace(" x", " $\\times$"))
    m("EccBestBer", floor_pct(codes_n[best]["max_ber_fail_1e-6"]))
    m("EccWorstBer", floor_pct(codes_n[worst_c]["max_ber_fail_1e-6"]))
    m("EccBestBerExact", f"{100 * codes_n[best]['max_ber_fail_1e-6']:.4f}")
    zero_pts = [e for e in es if e["failure_weight"] == 0]
    N["ecc"]["zero_failure_points"] = len(zero_pts)
    N["ecc"]["zero_failure_upper_95"] = 1 - 0.05 ** (1 / cfg["ecc"]["weight_trials"])
    m("EccZeroPoints", str(len(zero_pts)))
    m("EccZeroUpper", "$1.5\\times10^{-5}$" if abs(N["ecc"]["zero_failure_upper_95"] - 1.498e-5) < 1e-7
      else f"{N['ecc']['zero_failure_upper_95']:.1e}")
    cv_path = res.parent / "code_verification.json"
    if cv_path.exists():
        cv = json.loads(cv_path.read_text(encoding="utf-8"))
        N["ecc"]["independent_verification"] = {
            "passed": cv["passed"], "checks": len(cv["checks"]),
            "threshold_percent_exact": cv["bch_127_36_15_x4"]["threshold_percent"]}
        m("EccIndepChecks", str(len(cv["checks"])))
    for b, tag in ((0.05, "Five"), (0.1, "Ten")):
        if f"fail_iid_{b}" in codes_n[best]:
            m(f"EccBestFailIid{tag}", plain(codes_n[best][f"fail_iid_{b}"]))
            m(f"EccBestFailHet{tag}", plain(codes_n[best][f"fail_heterogeneous_{b}"]))
            m(f"EccBestFailCom{tag}", plain(codes_n[best][f"fail_common_mode_{b}"]))
    m("EccRepMsMax", f"{max(N['overhead']['rep_ms'].values()):.1f}")
    if "tail_check" in N["ecc"]:
        tc = N["ecc"]["tail_check"]
        m("EccTailTrials", f"{tc['trials']:,}".replace(",", "\\,"))
        m("EccTailFails", str(tc["failures"]))
        m("EccTailExpected", f"{tc['expected_failures']:.1f}")
        m("EccTailMainP", f"{tc['main_run']['p_at_least_one']:.2f}")
    m("QualApufUnifSd", f"{Qn['apuf']['uniformity_sd']:.3f}")
    m("QualFullUnifSd", f"{Qn['mask_xor2']['uniformity_sd']:.3f}")
    m("QualApufNistRuns", f"{Qn['apuf']['nist_runs']['pass_proportion']:.2f}")
    m("QualFullNistRuns", f"{Qn['mask_xor2']['nist_runs']['pass_proportion']:.2f}")
    m("QualNistBandLo", f"{Qn['apuf']['nist_freq']['accept_lo']:.3f}")
    m("QualApufMinEnt", f"{Qn['apuf']['min_entropy']:.3f}")
    m("QualFullMinEnt", f"{Qn['mask_xor2']['min_entropy']:.3f}")
    m("MaskedBinomSd", pct(N["masked"]["binomial_sd_one_fit"], 1))
    if "reliability" in N and len(bs) > 1:
        m("RelBudgetSecond", f"{bs[1]:,}".replace(",", "\\,"))
        m("RelAccSecond", pct(rows[str(bs[1])]["mean"]))
        m("RelAccSecondWorst", pct(rows[str(bs[1])]["min"]))
        m("RelAccMinWorst", pct(rows[str(bs[0])]["min"]))
        m("RelAccMinBest", pct(rows[str(bs[0])]["max"]))

    S = N["statistics"]
    m("StatMaxTostP", f"{S['primary_max_tost_p']:.3f}")
    m("StatSdMin", pct(S["primary_device_sd_min"], 2))
    m("StatSdMax", pct(S["primary_device_sd_max"], 2))
    m("StatVrMin", f"{S['variance_ratio_min']:.2f}")
    m("StatVrMax", f"{S['variance_ratio_max']:.2f}")
    m("StatMarginSupported", pct(S["smallest_margin_supported_max"], 2))
    m("StatHolmSig", str(S["single_attack_holm_significant"]))
    m("StatPowerZero", f"{S['tost_power_true_diff_0']:.3f}" if S["tost_power_true_diff_0"] < 0.9995 else "above 0.999")
    m("StatPowerHalf", f"{S['tost_power_true_diff_0.5pp']:.2f}")

    # ------------------------------------------------ stability models (E5b)
    sm_path = res / "stability_models_raw.csv"
    if sm_path.exists():
        with open(sm_path, newline="", encoding="utf-8") as f:
            sm = list(csv.DictReader(f))
        for r in sm:
            for k_ in ("noisiness", "unstable_fraction", "unstable_fraction_all",
                       "predicted_unstable_fraction", "hidden_model_acc", "abs_cosine",
                       "norm_ratio", "val_loss"):
                r[k_] = float(r[k_])
            for k_ in ("device", "repeats", "budget"):
                r[k_] = int(r[k_])
        sc = cfg["stability_models"]
        gb = sc["grid_budget"]
        cells = defaultdict(list)
        for r in sm:
            cells[(r["noisiness"], r["repeats"], r["budget"], r["model"])].append(r)
        ss = []
        for (nu, T, b, model), rr in sorted(cells.items()):
            acc = [r["hidden_model_acc"] for r in rr]
            s_ = mean_t_interval(acc)
            ss.append({"noisiness": nu, "repeats": T, "budget": b, "model": model,
                       "devices": s_["n"], "mean": s_["mean"], "sd": s_["sd"],
                       "ci_lo": s_["lo"], "ci_hi": s_["hi"], "min": min(acc), "max": max(acc),
                       "devices_above_95": int(sum(a >= 0.95 for a in acc)),
                       "norm_ratio_mean": float(np.mean([r["norm_ratio"] for r in rr])),
                       "val_loss_mean": float(np.mean([r["val_loss"] for r in rr])),
                       "unstable_observed": float(np.mean([r["unstable_fraction_all"] for r in rr])),
                       "unstable_predicted": float(np.mean([r["predicted_unstable_fraction"] for r in rr])),
                       "per_device": json.dumps(acc)})
        write_csv(summ / "stability_models_summary.csv", ss)

        def sfind(nu, T, b, model):
            return [x for x in ss if abs(x["noisiness"] - nu) < 1e-12 and x["repeats"] == T
                    and x["budget"] == b and x["model"] == model][0]
        mlabel = {"logistic": "Surrogate", "exact_cold": "Exact, random",
                  "exact_warm": "Exact, warm"}
        lines = ["\\begin{tabular}{cl" + "c" * len(sc["repeats"]) + "}", "\\toprule",
                 "$\\nu$ & Model & " + " & ".join(f"$T={T}$" for T in sc["repeats"]) + " \\\\",
                 "\\midrule"]
        for nu in sc["noisiness"]:
            for i, model in enumerate(sc["models"]):
                cells_ = []
                for T in sc["repeats"]:
                    x = sfind(nu, T, gb, model)
                    cells_.append(f"{pct(x['mean'])} ({pct(x['min'])})")
                lines.append((f"{nu:g}" if i == 0 else "") + f" & {mlabel[model]} & "
                             + " & ".join(cells_) + " \\\\")
            lines.append("\\midrule")
        lines[-1] = "\\bottomrule"
        lines.append("\\end{tabular}")
        (tables / "tab_stability.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

        bc = sc["baseline"]
        bl = {model: {b: sfind(bc["noisiness"], bc["repeats"], b, model) for b in bc["budgets"]}
              for model in sc["models"]}
        bmaxs = max(bc["budgets"])
        gap = max(abs(x["unstable_observed"] - x["unstable_predicted"]) for x in ss)
        paired = defaultdict(dict)
        for r in sm:
            if r["budget"] == gb:
                paired[(r["noisiness"], r["repeats"], r["device"])][r["model"]] = r["hidden_model_acc"]
        diffs = [v["exact_warm"] - v["logistic"] for v in paired.values()]
        cold_fail = sum(1 for v in paired.values() if v["exact_cold"] < 0.95)
        # the 16 cells of one device are not independent replicates, so the
        # paired difference is averaged per device before the interval is formed
        per_dev = defaultdict(list)
        for (nu_, T_, d_), v in paired.items():
            per_dev[d_].append(v["exact_warm"] - v["logistic"])
        dstat = mean_t_interval([float(np.mean(v)) for v in per_dev.values()])
        N["stability_models"] = {
            "grid_budget": gb, "cells": len(sc["noisiness"]) * len(sc["repeats"]),
            "baseline": {model: {str(b): {k_: bl[model][b][k_] for k_ in
                                          ("mean", "min", "max", "ci_lo", "ci_hi",
                                           "devices_above_95", "norm_ratio_mean", "val_loss_mean")}
                                 for b in bc["budgets"]} for model in sc["models"]},
            "max_gap_observed_vs_predicted_unstable_fraction": gap,
            "paired_diff_exact_warm_minus_logistic": {**dstat, "min": min(diffs), "max": max(diffs)},
            "exact_cold_fits_below_95": cold_fail, "fits_per_model_on_grid": len(paired),
            "grid_min_mean": {model: min(sfind(nu, T, gb, model)["mean"]
                                         for nu in sc["noisiness"] for T in sc["repeats"])
                              for model in sc["models"]},
            "grid_min_device": {model: min(sfind(nu, T, gb, model)["min"]
                                           for nu in sc["noisiness"] for T in sc["repeats"])
                                for model in sc["models"]},
            "grid_norm_ratio_exact_warm": [min(sfind(nu, T, gb, "exact_warm")["norm_ratio_mean"]
                                               for nu in sc["noisiness"] for T in sc["repeats"]),
                                           max(sfind(nu, T, gb, "exact_warm")["norm_ratio_mean"]
                                               for nu in sc["noisiness"] for T in sc["repeats"])],
        }
        SM = N["stability_models"]
        m("StabGridBudget", f"{gb:,}".replace(",", "\\,"))
        m("StabCells", str(SM["cells"]))
        m("StabFits", str(SM["fits_per_model_on_grid"]))
        m("StabBaseSurr", pct(bl["logistic"][bmaxs]["mean"]))
        m("StabBaseSurrWorst", pct(bl["logistic"][bmaxs]["min"]))
        m("StabBaseExact", pct(bl["exact_warm"][bmaxs]["mean"]))
        m("StabBaseExactWorst", pct(bl["exact_warm"][bmaxs]["min"]))
        b3 = sorted(bc["budgets"])[1]
        m("StabBaseBudgetSecond", f"{b3:,}".replace(",", "\\,"))
        m("StabBaseExactSecond", pct(bl["exact_warm"][b3]["mean"]))
        m("StabBaseSurrSecond", pct(bl["logistic"][b3]["mean"]))
        b1 = sorted(bc["budgets"])[0]
        m("StabBaseExactFirst", pct(bl["exact_warm"][b1]["mean"]))
        m("StabBaseExactFirstWorst", pct(bl["exact_warm"][b1]["min"]))
        m("StabBaseSurrFirst", pct(bl["logistic"][b1]["mean"]))
        m("StabBaseSurrFirstWorst", pct(bl["logistic"][b1]["min"]))
        m("StabGap", pct(gap, 2))
        m("StabDiffMean", f"{100 * dstat['mean']:+.2f}")
        m("StabDiffLo", f"{100 * dstat['lo']:+.2f}")
        m("StabDiffHi", f"{100 * dstat['hi']:+.2f}")
        m("StabColdFail", str(cold_fail))
        m("StabGridMinSurr", pct(SM["grid_min_mean"]["logistic"]))
        m("StabGridMinExact", pct(SM["grid_min_mean"]["exact_warm"]))
        m("StabGridMinDevSurr", pct(SM["grid_min_device"]["logistic"]))
        m("StabGridMinDevExact", pct(SM["grid_min_device"]["exact_warm"]))
        m("StabNormLo", f"{SM['grid_norm_ratio_exact_warm'][0]:.2f}")
        m("StabNormHi", f"{SM['grid_norm_ratio_exact_warm'][1]:.2f}")

    # ------------------------------------------------ product LR restarts (E5d)
    pr_path = res / "prodlr_restarts_raw.csv"
    if pr_path.exists():
        with open(pr_path, newline="", encoding="utf-8") as f:
            pr = list(csv.DictReader(f))
        acc = np.array([float(r["test_acc"]) for r in pr])
        val = np.array([float(r["val_acc"]) for r in pr])
        dev = np.array([int(r["device"]) for r in pr])
        conv = acc >= 0.80
        best_by_val = [float(acc[dev == d][np.argmax(val[dev == d])]) for d in sorted(set(dev))]
        first5 = [float(acc[dev == d][:5][np.argmax(val[dev == d][:5])]) for d in sorted(set(dev))]
        N["prodlr_restarts"] = {
            "k": int(pr[0]["k"]), "budget": int(pr[0]["budget"]), "restarts_total": int(acc.size),
            "devices": len(set(dev)), "converged_above_80": int(conv.sum()),
            "fraction_converged": float(conv.mean()),
            "converged_mean_acc": float(acc[conv].mean()) if conv.any() else float("nan"),
            "converged_min_acc": float(acc[conv].min()) if conv.any() else float("nan"),
            "converged_max_acc": float(acc[conv].max()) if conv.any() else float("nan"),
            "not_converged_max_acc": float(acc[~conv].max()) if (~conv).any() else float("nan"),
            "in_between_55_and_80": int(((acc > 0.55) & (acc < 0.80)).sum()),
            "best_by_validation_per_device": best_by_val,
            "best_of_first_five_per_device": first5,
            "devices_without_any_converged_restart": int(sum(not conv[dev == d].any()
                                                             for d in set(dev)))}
        P = N["prodlr_restarts"]
        lines = ["\\begin{tabular}{cccc}", "\\toprule",
                 "Device & Converged restarts & Accuracy of converged restarts & Selected on validation \\\\",
                 "\\midrule"]
        for i, d in enumerate(sorted(set(dev))):
            a_d, c_d = acc[dev == d], conv[dev == d]
            rng_txt = (f"{pct(a_d[c_d].min())} to {pct(a_d[c_d].max())}" if c_d.any() else "none")
            lines.append(f"{d + 1} & {int(c_d.sum())} of {a_d.size} & {rng_txt} & {pct(best_by_val[i])} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (tables / "tab_restarts.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        m("PrK", str(P["k"]))
        m("PrRestarts", str(P["restarts_total"] // P["devices"]))
        m("PrTotal", str(P["restarts_total"]))
        m("PrConverged", str(P["converged_above_80"]))
        m("PrConvMean", pct(P["converged_mean_acc"]))
        m("PrConvMin", pct(P["converged_min_acc"]))
        m("PrConvMax", pct(P["converged_max_acc"]))
        m("PrFailMax", pct(P["not_converged_max_acc"]))
        m("PrBestMin", pct(min(best_by_val)))
        m("PrBestMax", pct(max(best_by_val)))
        m("PrDevNoConv", str(P["devices_without_any_converged_restart"]))

    # ------------------------------------------------ reproduction facts
    import re as _re
    import subprocess as _sp
    try:
        out_ = _sp.run([sys.executable, "-m", "pytest", str(ROOT / "tests"), "--collect-only", "-q"],
                       capture_output=True, text=True, timeout=300).stdout
        n_tests = int(_re.search(r"(\d+) tests? collected", out_).group(1))
    except Exception:
        n_tests = None
    cmp_path = res.parent / "clean_env" / "comparison_with_stored.json"
    cmp_ = json.loads(cmp_path.read_text(encoding="utf-8")) if cmp_path.exists() else None
    N["reproduction"] = {"unit_tests_collected": n_tests, "clean_env_comparison": cmp_}
    m("ReproTests", str(n_tests) if n_tests else "n.a.")
    m("ReproFiles", str(cmp_["files_identical"]) if cmp_ else "n.a.")

    (summ / "numbers.json").write_text(json.dumps(N, indent=1), encoding="utf-8")
    (summ / "macros.json").write_text(json.dumps(macros, indent=1), encoding="utf-8")
    tex = ["% Generated by experiments/analyze.py. Do not edit."]
    tex += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (tables / "numbers.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"analysis written to {summ} and {tables}")


if __name__ == "__main__":
    main()
