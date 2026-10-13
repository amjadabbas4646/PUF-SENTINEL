"""Summaries, LaTeX tables and text macros for the experiments of
run_extensions.py, plus the additional statistical reporting of the second
revision. Run after experiments/analyze.py.

Usage:  python experiments/analyze_extensions.py --profile full
Output: experiments/results/<profile>/summary/ext_*.csv, numbers_ext.json
        tables/tab_*.tex and tables/numbers_ext.tex (full profile)

Every number comes from a raw file. Nothing is typed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy import stats as sst

from common.stats import mean_t_interval, tost_one_sample

ATTACKS = ["lr", "svm", "rf", "gb", "mlp", "lr_xor"]
ATTACK_LABEL = {"lr": "LR", "svm": "SVM", "rf": "RF", "gb": "GB", "mlp": "MLP",
                "lr_xor": "Product LR"}
CLASS_LABEL = {"R": "R: keyed response mask", "X": "X: challenge XOR key",
               "L": "L: secret linear challenge map", "P": "P: keyed challenge PRF",
               "H": "H: keyless hashed output"}
CLASS_SECRET = {"R": "key", "X": "key", "L": "matrix", "P": "key", "H": "none"}


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if v in ("True", "False"):
                r[k] = v == "True"
                continue
            try:
                r[k] = int(v)
            except ValueError:
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}"


def num(x: int) -> str:
    return f"{x:,}".replace(",", "\\,")


def ci(vals, d: int = 1) -> str:
    s = mean_t_interval(vals)
    return f"{pct(s['mean'], d)} $\\pm$ {pct((s['hi'] - s['lo']) / 2, d)}"


def selected(cells: dict, units: list, attacks: list[str]) -> list[float]:
    """Test accuracy of the attack with the best validation accuracy, per
    unit (device or instance). cells[(unit, attack)] is a raw row."""
    out = []
    for u in units:
        cand = [a for a in attacks if (u, a) in cells]
        best = max(cand, key=lambda a: (cells[(u, a)]["val_acc"], -cand.index(a)))
        out.append(cells[(u, best)]["test_acc"])
    return out


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
    man = json.loads((res / "manifest_extensions.json").read_text(encoding="utf-8"))
    ext = man["config"]
    cfg = json.loads((res / "manifest.json").read_text(encoding="utf-8"))["config"]
    A = cfg["attack"]
    margin = A["equivalence_margin"]
    bmax = max(A["budgets"])
    N, macros = {}, {}

    def m(name, value):
        macros[name] = value

    main_summary = read_csv(summ / "attack_summary.csv")

    def main_cell(construction, attack, budget=bmax):
        hit = [s for s in main_summary if s["construction"] == construction
               and s["dist"] == "uniform" and float(s["noisiness"]) == 0.0
               and s["budget"] == budget and s["attack"] == attack]
        assert len(hit) == 1, (construction, attack, budget, len(hit))
        return json.loads(hit[0]["per_device"])

    # ------------------------------------------------------- XOR stability
    xs = read_csv(res / "xor_stability_raw.csv")
    S = ext["xor_stability"]
    cells = defaultdict(list)
    for r in xs:
        cells[(r["k"], r["budget"])].append(r)
    xsum = []
    for (k, b), rr in sorted(cells.items()):
        ag = [r["model_agreement_sign_free"] for r in rr]
        nk = [r["masked_response_acc_no_key"] for r in rr]
        wk = [r["masked_response_acc_with_key"] for r in rr]
        s_ag, s_nk, s_wk = mean_t_interval(ag), mean_t_interval(nk), mean_t_interval(wk)
        xsum.append({"k": k, "budget": b, "queries": rr[0]["queries"], "devices": len(rr),
                     "unstable_fraction": float(np.mean([r["unstable_fraction"] for r in rr])),
                     "mask_cancels_all": all(r["mask_cancels"] for r in rr),
                     "agreement_mean": s_ag["mean"], "agreement_min": min(ag),
                     "agreement_max": max(ag), "agreement_lo": s_ag["lo"],
                     "agreement_hi": s_ag["hi"],
                     "devices_above_95": int(sum(a >= 0.95 for a in ag)),
                     "min_cosine": float(min(r["min_matched_cosine"] for r in rr)),
                     "no_key_mean": s_nk["mean"], "no_key_lo": s_nk["lo"], "no_key_hi": s_nk["hi"],
                     "no_key_min": min(nk), "no_key_max": max(nk),
                     "no_key_tost_p": tost_one_sample(nk, 0.5, margin)["p_tost"],
                     "with_key_mean": s_wk["mean"], "with_key_min": min(wk),
                     "epochs_median": float(np.median([r["epochs"] for r in rr])),
                     "seconds_median": float(np.median([r["seconds"] for r in rr])),
                     "per_device_agreement": json.dumps(ag)})
    write_csv(summ / "ext_xor_stability_summary.csv", xsum)

    def xf(k, b):
        return [x for x in xsum if x["k"] == k and x["budget"] == b][0]

    lines = ["\\begin{tabular}{crrcccc}", "\\toprule",
             " & & & Unstable & PUF model, & \\multicolumn{2}{c}{Masked response} \\\\",
             "$k$ & Challenges & Queries & (\\%) & up to sign & no key & key known \\\\",
             "\\midrule"]
    for k in S["k"]:
        for b in S["budgets"]:
            x = xf(k, b)
            lines.append(f"{k} & {num(b)} & {num(x['queries'])} & {pct(x['unstable_fraction'])} & "
                         f"{pct(x['agreement_mean'])} ({pct(x['agreement_min'])}) & "
                         f"{pct(x['no_key_mean'])} ({pct(x['no_key_max'])}) & "
                         f"{pct(x['with_key_mean'])} ({pct(x['with_key_min'])}) \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    (tables / "tab_xor_stability.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    xb = sorted(S["budgets"])
    N["xor_stability"] = {f"k{x['k']}_{x['budget']}": {kk: x[kk] for kk in x
                                                        if kk != "per_device_agreement"}
                          for x in xsum}
    m("XstRepeats", str(S["repeats"]))
    m("XstDevices", str(S["devices"]))
    m("XstBudgetMax", num(xb[-1]))
    m("XstBudgetMid", num(xb[len(xb) // 2]))
    m("XstBudgetMin", num(xb[0]))
    m("XstQueriesMax", num(xb[-1] * S["repeats"]))
    m("XstQueriesMid", num(xb[len(xb) // 2] * S["repeats"]))
    m("XstSignCrps", str(S["sign_crps"]))
    m("XstMaskCancels", "all" if all(x["mask_cancels_all"] for x in xsum) else "NOT ALL")
    words = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}
    for k in S["k"]:
        hi, mid = xf(k, xb[-1]), xf(k, xb[len(xb) // 2])
        t = words[k]
        m(f"Xst{t}Agree", pct(hi["agreement_mean"]))
        m(f"Xst{t}AgreeWorst", pct(hi["agreement_min"]))
        m(f"Xst{t}AgreeMid", pct(mid["agreement_mean"]))
        m(f"Xst{t}AgreeMidWorst", pct(mid["agreement_min"]))
        m(f"Xst{t}NoKey", pct(hi["no_key_mean"], 2))
        m(f"Xst{t}WithKey", pct(hi["with_key_mean"]))
        m(f"Xst{t}Unstable", pct(hi["unstable_fraction"]))
    nk_all = [x for x in xsum]
    m("XstNoKeyMin", pct(min(x["no_key_min"] for x in nk_all), 2))
    m("XstNoKeyMax", pct(max(x["no_key_max"] for x in nk_all), 2))
    m("XstNoKeyTostMax", f"{max(x['no_key_tost_p'] for x in nk_all):.3f}")
    m("XstNoKeyCells", str(len(nk_all)))
    m("XstNoKeyEquiv", str(sum(x["no_key_tost_p"] < 0.05 for x in nk_all)))

    # ------------------------------------------------------- classes
    cr = read_csv(res / "classes_raw.csv")
    cs = read_csv(res / "class_stability_raw.csv")
    C = ext["classes"]
    cb = max(C["budgets"])
    devs = list(range(C["devices"]))
    csum, row_tex = [], []

    def add(kind, label, vals):
        s = mean_t_interval(vals)
        eq = tost_one_sample(vals, 0.5, margin)["p_tost"]
        csum.append({"class": kind, "quantity": label, "devices": len(vals), "mean": s["mean"],
                     "ci_lo": s["lo"], "ci_hi": s["hi"], "min": min(vals), "max": max(vals),
                     "tost_p_vs_chance": eq, "per_device": json.dumps(vals)})
        return s, eq

    stab_r = [r["masked_response_acc_no_key"] for r in xs
              if r["k"] == 1 and r["budget"] == ext["class_stability"]["budget"]]
    N["classes"] = {}
    for kind in ["R"] + C["kinds"]:
        if kind == "R":
            direct = main_cell("mask_xor2", "best_val")
            subst = main_cell("mask_only", "best_val")
            known = main_cell("mask_xor2", "oracle_lr_xor")
            stab = stab_r
            direct_eval = None
        else:
            rows = [r for r in cr if r["class"] == kind and r["budget"] == cb]
            full = {(r["device"], r["attack"]): r for r in rows if r["variant"] == "full"}
            sub = {(r["device"], r["attack"]): r for r in rows if r["variant"] == "substituted"}
            direct = selected(full, devs, ATTACKS)
            subst = selected(sub, devs, ATTACKS)
            known = ([full[(d, "known_secret")]["test_acc"] for d in devs]
                     if (0, "known_secret") in full else None)
            direct_eval = ([sub[(d, "direct_evaluation")]["test_acc"] for d in devs]
                           if (0, "direct_evaluation") in sub else None)
            stab = [r["response_acc"] for r in cs if r["class"] == kind]
            base_d = [full[(d, ATTACKS[0])]["majority_baseline"] for d in devs]
            excess = [a - b for a, b in zip(direct, base_d)]
            se, pe = add(kind, "direct_minus_majority_baseline",
                         [0.5 + e for e in excess])       # shifted so that the test is against 0
            N.setdefault("class_bias", {})[kind] = {
                "majority_baseline_mean": float(np.mean(base_d)),
                "majority_baseline_max": max(base_d),
                "excess_mean": float(np.mean(excess)), "excess_min": min(excess),
                "excess_max": max(excess), "excess_tost_p": pe}
            m(f"Cls{kind}Majority", pct(float(np.mean(base_d))))
            m(f"Cls{kind}Excess", f"{100 * float(np.mean(excess)):+.2f}")
            m(f"Cls{kind}ExcessTost", "$<$0.001" if pe < 0.001 else f"{pe:.3f}")
        sd, pd_ = add(kind, "direct", direct)
        ss, ps_ = add(kind, "substituted", subst)
        cells_ = [CLASS_LABEL[kind], CLASS_SECRET[kind], ci(direct), ci(subst)]
        if direct_eval is not None:
            add(kind, "substituted_direct_evaluation", direct_eval)
            cells_[-1] += f" ({pct(float(np.mean(direct_eval)), 0)})"
        if known is not None:
            add(kind, "known_secret", known)
            cells_.append(ci(known))
        else:
            cells_.append("no secret")
        if stab:
            add(kind, "stability_response_prediction", stab)
            cells_.append(f"{pct(float(np.mean(stab)))} ({pct(min(stab))}, {pct(max(stab))})")
        else:
            cells_.append("")
        row_tex.append(" & ".join(cells_) + " \\\\")
        N["classes"][kind] = {"direct_mean": sd["mean"], "direct_tost_p": pd_,
                              "substituted_mean": ss["mean"], "substituted_tost_p": ps_,
                              "known_mean": float(np.mean(known)) if known else None,
                              "direct_evaluation": float(np.mean(direct_eval)) if direct_eval else None,
                              "stability_mean": float(np.mean(stab)) if stab else None,
                              "stability_min": min(stab) if stab else None}
        m(f"Cls{kind}Direct", pct(sd["mean"]))
        m(f"Cls{kind}Subst", pct(ss["mean"]))
        if known:
            m(f"Cls{kind}Known", pct(float(np.mean(known))))
        if stab:
            m(f"Cls{kind}Stab", pct(float(np.mean(stab))))
            m(f"Cls{kind}StabWorst", pct(min(stab)))
        m(f"Cls{kind}DirectTost", "$<$0.001" if pd_ < 0.001 else f"{pd_:.3f}")
    write_csv(summ / "ext_classes_summary.csv", csum)
    lines = ["\\begin{tabular}{llcccc}", "\\toprule",
             " & & Direct attack, & PUF replaced by & Secret known, & Stability only, \\\\",
             "Class & Secret & selected & a public function & matched attack & response prediction \\\\",
             "\\midrule"] + row_tex + ["\\bottomrule", "\\end{tabular}"]
    (tables / "tab_classes.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    m("ClsBudget", num(cb))
    m("ClsDevices", str(C["devices"]))
    m("ClsHashBits", str(C["hash_bits"]))
    m("ClsStabBudget", num(ext["class_stability"]["budget"]))
    m("ClsStabRepeats", str(ext["class_stability"]["repeats"]))
    m("ClsStabQueries", num(ext["class_stability"]["budget"] * ext["class_stability"]["repeats"]))
    hrows = [r for r in cs if r["class"] == "H"]
    if hrows:
        m("ClsHCos", f"{float(np.mean([r['abs_cosine'] for r in hrows])):.3f}")
        m("ClsHUnstable", pct(float(np.mean([r["unstable_fraction"] for r in hrows]))))
    # which attack reaches what on class X and L (for the text)
    for kind in ("X", "L", "P", "H"):
        for atk in ("lr", "mlp", "lr_xor"):
            v = [r["test_acc"] for r in cr if r["class"] == kind and r["variant"] == "full"
                 and r["budget"] == cb and r["attack"] == atk]
            if v:
                N["classes"][kind][f"{atk}_mean"] = float(np.mean(v))
    m("ClsXProd", pct(N["classes"]["X"]["lr_xor_mean"]))
    m("ClsXLr", pct(N["classes"]["X"]["lr_mean"]))

    # ------------------------------------------------------- convergence and cost
    cv = read_csv(res / "convergence_raw.csv")
    V = ext["convergence"]
    raw_main = read_csv(res / "attack_raw.csv")
    stored = {(r["construction"], r["device"], r["budget"], r["attack"]): r["test_acc"]
              for r in raw_main if r["dist"] == "uniform" and float(r["noisiness"]) == 0.0}
    same = [abs(r["test_acc"] - stored[(r["construction"], r["device"], r["budget"], r["attack"])]) < 1e-9
            for r in cv]
    N["convergence"] = {"fits": len(cv), "accuracy_equal_to_stored_run": int(sum(same)),
                        "all_equal": bool(all(same))}
    cells = defaultdict(list)
    for r in cv:
        cells[(r["construction"], r["budget"], r["attack"])].append(r)
    vsum = []
    for (c, b, a), rr in sorted(cells.items()):
        effort = [r["epochs"] if a in ("mlp", "lr_xor") else r["iterations"] for r in rr]
        vsum.append({"construction": c, "budget": b, "attack": a, "devices": len(rr),
                     "effort_unit": "epochs" if a in ("mlp", "lr_xor") else
                     ("trees" if a == "rf" else "iterations"),
                     "effort_median": float(np.median(effort)), "effort_min": min(effort),
                     "effort_max": max(effort),
                     "restarts_median": float(np.median([r["restarts"] for r in rr])),
                     "restarts_max": max(r["restarts"] for r in rr),
                     "runs_at_epoch_limit": int(sum(r["hit_epoch_limit"] for r in rr)),
                     "seconds_median": float(np.median([r["seconds"] for r in rr])),
                     "seconds_max": max(r["seconds"] for r in rr),
                     "test_acc_mean": float(np.mean([r["test_acc"] for r in rr]))})
    write_csv(summ / "ext_convergence_summary.csv", vsum)

    def vf(c, a, b=bmax):
        return [x for x in vsum if x["construction"] == c and x["attack"] == a and x["budget"] == b][0]

    cons = V["constructions"]
    clabel = {"apuf": "Arbiter", "xor2": "2-XOR", "mask_xor2": "Masked 2-XOR"}
    lines = ["\\begin{tabular}{ll" + "rr" * len(cons) + "}", "\\toprule",
             " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{clabel[c]}}}" for c in cons) + " \\\\",
             "Attack & Unit of effort & " + " & ".join("Effort & Time (s)" for _ in cons) + " \\\\",
             "\\midrule"]
    for a in V["attacks"]:
        cells_ = []
        for c in cons:
            x = vf(c, a)
            eff = f"{x['effort_median']:.0f}"
            if a == "lr_xor":
                eff += f" ({x['restarts_median']:.0f})"
            cells_ += [eff, f"{x['seconds_median']:.1f}"]
        unit = vf(cons[0], a)["effort_unit"] + (" (restarts)" if a == "lr_xor" else "")
        lines.append(f"{ATTACK_LABEL[a]} & {unit} & " + " & ".join(cells_) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tables / "tab_cost.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cfg_att = man["attack_config"]
    m("CostFits", num(len(cv)))
    m("CostSame", "all" if all(same) else "NOT ALL")
    m("CostMlpMaxEpochs", str(cfg_att["mlp"]["max_epochs"]))
    m("CostProdMaxEpochs", str(cfg_att["lr_xor"]["max_epochs"]))
    m("CostProdRestarts", str(cfg_att["lr_xor"]["restarts"]))
    m("CostLrMaxIter", num(cfg_att["lr"]["max_iter"]))
    for c, tag in (("apuf", "Apuf"), ("xor2", "Xor"), ("mask_xor2", "Full")):
        if c in cons:
            m(f"Cost{tag}MlpEpochs", f"{vf(c, 'mlp')['effort_median']:.0f}")
            m(f"Cost{tag}ProdEpochs", f"{vf(c, 'lr_xor')['effort_median']:.0f}")
            m(f"Cost{tag}ProdRestarts", f"{vf(c, 'lr_xor')['restarts_median']:.0f}")
            m(f"Cost{tag}ProdRestartsMax", str(vf(c, 'lr_xor')['restarts_max']))
            m(f"Cost{tag}LrIter", f"{vf(c, 'lr')['effort_median']:.0f}")
    limit_hits = sum(x["runs_at_epoch_limit"] for x in vsum if x["attack"] in ("mlp", "lr_xor"))
    grad_runs = sum(r["restarts"] if r["attack"] == "lr_xor" else 1 for r in cv
                    if r["attack"] in ("mlp", "lr_xor"))
    m("CostLimitHits", str(limit_hits))
    if "apuf" in cons:
        apuf_prod_hits = sum(x["runs_at_epoch_limit"] for x in vsum
                             if x["construction"] == "apuf" and x["attack"] == "lr_xor")
        m("CostLimitHitsApufProd", str(apuf_prod_hits))
        m("CostLimitHitsOther", str(limit_hits - apuf_prod_hits))
    m("CostGradRuns", num(grad_runs))
    N["convergence"].update({"gradient_runs": grad_runs, "runs_ending_at_epoch_limit": limit_hits})
    lr_iter_max = max(r["iterations"] for r in cv if r["attack"] == "lr")
    m("CostLrIterMax", num(lr_iter_max))

    # total cost of the whole study, from the stage timings and the raw files
    man_main = json.loads((res / "manifest.json").read_text(encoding="utf-8"))
    wall = {k_: v["seconds"] for k_, v in man_main["stages"].items()}
    wall_ext = {k_: v["seconds"] for k_, v in man["stages"].items()}
    fit_seconds = sum(r["seconds"] for r in raw_main)
    n_fits_main = len(raw_main)
    N["cost"] = {"wall_seconds_main": wall, "wall_seconds_extensions": wall_ext,
                 "jobs_main": man_main.get("jobs"), "jobs_extensions": man.get("jobs"),
                 "main_attack_fits": n_fits_main, "main_attack_fit_cpu_seconds": fit_seconds,
                 "cpu": man["versions"].get("cpu_count"), "platform": man["versions"].get("platform")}
    m("CostWallMainMin", f"{sum(wall.values()) / 60:.0f}")
    m("CostWallExtMin", f"{sum(wall_ext.values()) / 60:.0f}")
    m("CostMainFits", num(n_fits_main))
    m("CostMainFitCpuH", f"{fit_seconds / 3600:.1f}")
    m("CostJobs", str(man.get("jobs")))
    m("CostCrpsPerDevice", num(bmax + A["n_test"]))
    R = cfg["reliability_attack"]
    m("CostStabQueriesMax", num(max(R["budgets"]) * R["repeats"]))

    # ------------------------------------------------------- equivalence margin
    ch = read_csv(summ / "chance_tests.csv")
    prim = [x for x in ch if x["attack"] == "best_val" and x["budget"] == bmax]
    assert len(prim) == 9 or args.profile != "full", len(prim)
    sens = []
    summary_idx = {(s["construction"], s["dist"], float(s["noisiness"])): json.loads(s["per_device"])
                   for s in main_summary if s["attack"] == "best_val" and s["budget"] == bmax
                   and s["construction"].startswith("mask_")}
    sw = []
    for key, vals in summary_idx.items():
        sw.append(float(sst.shapiro(vals).pvalue) if len(vals) >= 3 else float("nan"))
    for d_pp in (0.25, 0.5, 0.75, 1.0, 2.0):
        ps = [tost_one_sample(v, 0.5, d_pp / 100)["p_tost"] for v in summary_idx.values()]
        sens.append({"margin_pp": d_pp, "comparisons": len(ps),
                     "equivalent_at_0.05": int(sum(p < 0.05 for p in ps)), "max_p": max(ps)})
    write_csv(summ / "ext_margin_sensitivity.csv", sens)
    N["margin"] = {"sensitivity": sens, "shapiro_wilk_min_p": float(np.nanmin(sw)),
                   "shapiro_wilk_below_0.05": int(np.sum(np.array(sw) < 0.05)),
                   "exact_match_64_bits_log2_at_margin": 64 * math.log2(0.5 + margin),
                   "binomial_sd_one_device": 0.5 / math.sqrt(A["n_test"])}
    for srow, tag in zip(sens, ("Quarter", "Half", "ThreeQuarter", "One", "Two")):
        m(f"Mg{tag}Equiv", str(srow["equivalent_at_0.05"]))
    m("MgComparisons", str(sens[0]["comparisons"]))
    m("MgShapiroMin", f"{N['margin']['shapiro_wilk_min_p']:.2f}")
    m("MgShapiroSig", str(N["margin"]["shapiro_wilk_below_0.05"]))
    m("MgExactMatchLog", f"{-N['margin']['exact_match_64_bits_log2_at_margin']:.1f}")
    m("MgSdRatio", f"{margin / N['margin']['binomial_sd_one_device']:.0f}")
    # smallest margin from which every larger tested margin gives equivalence
    # in all comparisons
    smallest = None
    for s_ in reversed(sens):
        if s_["equivalent_at_0.05"] == s_["comparisons"]:
            smallest = s_["margin_pp"]
        else:
            break
    m("MgSmallestAll", f"{smallest:g}" if smallest is not None else "n.a.")
    N["margin"]["smallest_margin_with_all_equivalent"] = smallest
    lines = ["\\begin{tabular}{ccc}", "\\toprule",
             "Margin (\\pp{}s) & Equivalent at the 5\\% level & Largest $p_{\\mathrm{TOST}}$ \\\\",
             "\\midrule"]
    for s_ in sens:
        p_ = "$<$0.001" if s_["max_p"] < 0.001 else f"{s_['max_p']:.3f}"
        lines.append(f"{s_['margin_pp']:g} & {s_['equivalent_at_0.05']} of {s_['comparisons']} & {p_} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tables / "tab_margin.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------- entropy
    en = read_csv(res / "entropy_bounds.csv")
    write_csv(summ / "ext_entropy_summary.csv", en)
    ps_ = [p for p in ext["entropy"]["bias_p"]]
    show_p = [p for p in (0.5, 0.48, 0.45, 0.4, 0.3) if p in ps_] or ps_
    codes = [c["name"] for c in cfg["ecc"]["codes"]]

    def ef(code, p):
        return [e for e in en if e["code"] == code and abs(e["bias_p"] - p) < 1e-12][0]

    lines = ["\\begin{tabular}{lccc" + "c" * len(show_p) + "}", "\\toprule",
             " & Nominal & Message & Helper & \\multicolumn{" + str(len(show_p))
             + "}{c}{Lower bound on $\\tilde{H}_\\infty(S\\mid H)$ in bits for $p_1=$} \\\\",
             "Code & key bits & bits & redundancy & " + " & ".join(f"{p:g}" for p in show_p) + " \\\\",
             "\\midrule"]
    N["entropy"] = {}
    for code in codes:
        e0 = ef(code, show_p[0])
        cells_ = [code.replace(" x", " $\\times$"), str(e0["nominal_key_bits"]),
                  str(e0["secret_bits"]), str(e0["helper_redundancy_bits"])]
        N["entropy"][code] = {}
        for p in ps_:
            e = ef(code, p)
            N["entropy"][code][str(p)] = {"lower": e["entropy_lower"], "upper": e["entropy_upper"],
                                          "naive": e["naive_lower"]}
        for p in show_p:
            cells_.append(f"{math.floor(ef(code, p)['entropy_lower'] + 1e-9):d}")
        lines.append(" & ".join(cells_) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tables / "tab_entropy.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # largest bias at which each code still keeps 128 bits (lower bound), on a fine grid
    from error_correction.entropy import key_entropy_bounds
    keep = {}
    for c in cfg["ecc"]["codes"]:
        best_p = None
        for p in np.arange(0.5, 0.2995, -0.001):
            b = key_entropy_bounds(c["n"], c["k"], c["t"], c["bits_per_symbol"], c["blocks"],
                                   float(round(p, 3)))
            if b["entropy_lower"] >= 128:
                best_p = float(round(p, 3))
            else:
                break
        keep[c["name"]] = best_p
    N["entropy"]["smallest_p_one_with_128_bits"] = keep
    best_code = "BCH(127,36,15) x4"
    if best_code in keep:
        m("EntBestCode", best_code.replace(" x", " $\\times$"))
        m("EntBestKeepP", f"{keep[best_code]:.3f}" if keep[best_code] else "n.a.")
        m("EntBestAtFortyFive", f"{math.floor(ef(best_code, 0.45)['entropy_lower']):d}"
          if 0.45 in ps_ else "n.a.")
        m("EntBestAtForty", f"{math.floor(ef(best_code, 0.4)['entropy_lower']):d}"
          if 0.4 in ps_ else "n.a.")
    k78 = "BCH(127,78,7) x2"
    if k78 in keep:
        m("EntSeventyEightKeepP", f"{keep[k78]:.3f}" if keep[k78] else "n.a.")
    exact_128 = [c for c, v in keep.items() if v is not None and abs(v - 0.5) < 1e-9]
    m("EntNoSlackCodes", str(len(exact_128)))
    m("EntCodes", str(len(codes)))
    m("EntMsgMax", str(max(int(ef(c, ps_[0])["secret_bits"]) for c in codes)))
    m("EntMsgMin", str(min(int(ef(c, ps_[0])["secret_bits"]) for c in codes)))
    if 0.45 in ps_:
        low45 = [math.floor(ef(c, 0.45)["entropy_lower"] + 1e-9) for c in codes]
        m("EntLowFortyFiveMin", str(min(low45)))
        m("EntLowFortyFiveMax", str(max(low45)))
        m("EntLowFortyFiveBelow", str(sum(v < 128 for v in low45)))
    # full table of both bounds for the supplement
    lines = ["\\begin{tabular}{l" + "c" * len(ps_) + "}", "\\toprule",
             "Code & " + " & ".join(f"$p_1={p:g}$" for p in ps_) + " \\\\", "\\midrule"]
    for code in codes:
        cells_ = [f"{math.floor(ef(code, p)['entropy_lower'] + 1e-9):d} to "
                  f"{math.ceil(ef(code, p)['entropy_upper'] - 1e-9):d}" for p in ps_]
        lines.append(code.replace(" x", " $\\times$") + " & " + " & ".join(cells_) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tables / "tab_entropy_full.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    gaps = [e["entropy_upper"] - e["entropy_lower"] for e in en]
    m("EntMaxGap", f"{max(gaps):.1f}")
    naive_zero = sum(1 for e in en if e["naive_lower"] == 0 and e["entropy_lower"] > 0)
    N["entropy"]["max_gap_between_bounds_bits"] = max(gaps)
    N["entropy"]["cells_where_naive_bound_is_zero_but_refined_is_positive"] = naive_zero

    # ------------------------------------------------------- FPGA
    fpath = res / "fpga_raw.csv"
    if fpath.exists():
        fr = read_csv(fpath)
        F = ext["fpga"]
        fsum = []
        N["fpga"] = {}
        glabel = {"arbiter": "FPGA arbiter chains", "xor2": "FPGA 2-XOR (software XOR)"}
        vlabel = {"measured": "measured", "mask_only": "mask only", "masked": "masked",
                  "key_known": "masked, key known"}
        lines = ["\\begin{tabular}{llc" + "c" * len(F["budgets"]) + "}", "\\toprule",
                 "Instances & Responses & $n$ & " + " & ".join(f"{num(b)} CRPs" for b in F["budgets"])
                 + " \\\\", "\\midrule"]
        for kind in ("arbiter", "xor2"):
            insts = sorted({r["instance"] for r in fr if r["kind"] == kind})
            if not insts:
                continue
            for var in ("measured", "mask_only", "masked", "key_known"):
                cells_ = []
                for b in F["budgets"]:
                    rows = {(r["instance"], r["attack"]): r for r in fr if r["kind"] == kind
                            and r["variant"] == var and r["budget"] == b}
                    if var == "key_known":
                        vals = [r["test_acc"] for r in rows.values()]
                    else:
                        vals = selected(rows, insts, ATTACKS)
                    s = mean_t_interval(vals)
                    eq = tost_one_sample(vals, 0.5, margin)["p_tost"]
                    fsum.append({"kind": kind, "variant": var, "budget": b, "instances": len(vals),
                                 "mean": s["mean"], "ci_lo": s["lo"], "ci_hi": s["hi"],
                                 "min": min(vals), "max": max(vals), "tost_p_vs_chance": eq,
                                 "per_instance": json.dumps(vals)})
                    cells_.append(f"{pct(s['mean'])} ({pct(min(vals))}, {pct(max(vals))})")
                    N["fpga"][f"{kind}|{var}|{b}"] = {"mean": s["mean"], "min": min(vals),
                                                      "max": max(vals), "ci_lo": s["lo"],
                                                      "ci_hi": s["hi"], "tost_p": eq,
                                                      "instances": len(vals)}
                lead = f"{len(insts)} {glabel[kind]}" if var == "measured" else ""
                lines.append(f"{lead} & {vlabel[var]} & " + ("64, 65" if kind == "arbiter" else "65")
                             + " & " + " & ".join(cells_) + " \\\\")
            lines.append("\\midrule")
        lines[-1] = "\\bottomrule"
        lines.append("\\end{tabular}")
        (tables / "tab_fpga.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_csv(summ / "ext_fpga_summary.csv", fsum)
        fb = max(F["budgets"])
        ones = sorted({(r["instance"], r["fraction_of_ones"]) for r in fr if r["kind"] == "arbiter"})
        m("FpgaArbiters", str(len(ones)))
        m("FpgaXors", str(len({r["instance"] for r in fr if r["kind"] == "xor2"})))
        n_inst = len({r["instance"] for r in fr})
        words_ = {16: "Sixteen", 15: "Fifteen", 12: "Twelve", 10: "Ten", 8: "Eight", 6: "Six"}
        m("FpgaInstances", str(n_inst))
        m("FpgaInstancesWord", words_.get(n_inst, str(n_inst)))
        m("FpgaBudget", num(fb))
        m("FpgaTest", num(F["n_test"]))
        m("FpgaOnesMin", pct(min(o for _, o in ones)))
        m("FpgaOnesMax", pct(max(o for _, o in ones)))
        for kind, tag in (("arbiter", "Arb"), ("xor2", "Xor")):
            for var, vt in (("measured", "Meas"), ("mask_only", "MaskOnly"), ("masked", "Masked"),
                            ("key_known", "Key")):
                key = f"{kind}|{var}|{fb}"
                if key in N["fpga"]:
                    x = N["fpga"][key]
                    m(f"Fpga{tag}{vt}", pct(x["mean"]))
                    m(f"Fpga{tag}{vt}Min", pct(x["min"]))
                    m(f"Fpga{tag}{vt}Max", pct(x["max"]))
        for kind, tag in (("arbiter", "Arb"), ("xor2", "Xor")):
            for var, vt in (("mask_only", "MaskOnly"), ("masked", "Masked")):
                key = f"{kind}|{var}|{fb}"
                if key in N["fpga"]:
                    p = N["fpga"][key]["tost_p"]
                    m(f"Fpga{tag}{vt}Tost", "$<$0.001" if p < 0.001 else f"{p:.3f}")
                    m(f"Fpga{tag}{vt}Lo", pct(N["fpga"][key]["ci_lo"], 2))
                    m(f"Fpga{tag}{vt}Hi", pct(N["fpga"][key]["ci_hi"], 2))
        # which attack wins on the measured 2-XOR instances
        for atk in ATTACKS:
            v = [r["test_acc"] for r in fr if r["kind"] == "xor2" and r["variant"] == "measured"
                 and r["budget"] == fb and r["attack"] == atk]
            if v:
                N["fpga"][f"xor2_measured_{atk}"] = {"mean": float(np.mean(v)), "min": min(v), "max": max(v)}
        for atk, tag in (("lr_xor", "Prod"), ("mlp", "Mlp"), ("lr", "Lr")):
            if f"xor2_measured_{atk}" in N["fpga"]:
                m(f"FpgaXor{tag}", pct(N["fpga"][f"xor2_measured_{atk}"]["mean"]))

    (summ / "numbers_ext.json").write_text(json.dumps(N, indent=1), encoding="utf-8")
    (summ / "macros_ext.json").write_text(json.dumps(macros, indent=1), encoding="utf-8")
    tex = ["% Generated by experiments/analyze_extensions.py. Do not edit."]
    tex += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (tables / "numbers_ext.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"extension analysis written to {summ} and {tables}")


if __name__ == "__main__":
    main()
