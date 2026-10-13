"""Consistency checks between raw results, generated tables, the manuscript
and (optionally) a second independent run.

Checks
  1  every expected output file exists
  2  raw files match the SHA-256 hashes recorded in the run manifest
  3  tables regenerate byte for byte from the raw files
  4  every numeric macro used in the manuscript is defined, and every defined
     macro has the value stored in macros.json
  5  headline claims of the manuscript hold in numbers.json
  6  the hyperparameters printed in the manuscript equal those in the code
  7  with --compare DIR: raw outputs of a second run equal the first run,
     ignoring wall-clock columns

Usage:  python scripts/validate_outputs.py [--compare experiments/results/repro_check]
Exit code 0 when every check passes. A report is written to
experiments/results/validation_report.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
FULL = ROOT / "experiments" / "results" / "full"
TIMING_COLUMNS = {"seconds"}
RAW_CSV = ["attack_raw.csv", "legacy_raw.csv", "xor_scaling_raw.csv", "reliability_raw.csv",
           "stability_models_raw.csv", "prodlr_restarts_raw.csv",
           "quality_per_device.csv", "ecc_raw.csv", "ecc_analytic.csv", "hardware_raw.csv"]
RAW_JSON = ["quality_summary.json", "ecc_design.json"]
EXT_CSV = ["classes_raw.csv", "class_stability_raw.csv", "xor_stability_raw.csv",
           "convergence_raw.csv", "entropy_bounds.csv", "fpga_raw.csv"]
SUB = ROOT / "CEE_SUBMISSION"
PAPER = SUB.exists()     # the public code repository has no article folder
EXPECTED = ([FULL / f for f in RAW_CSV + RAW_JSON + EXT_CSV
             + ["overhead.json", "manifest.json", "manifest_extensions.json",
                "hardware_meta.json", "fpga_meta.json"]]
            + [ROOT / "tables" / f for f in
               ["numbers.tex", "numbers_ext.tex", "tab_attack_matrix.tex", "tab_ablation.tex",
                "tab_equivalence.tex", "tab_quality.tex", "tab_ecc.tex", "tab_xor_scaling.tex",
                "tab_hardware.tex", "tab_stability.tex", "tab_restarts.tex", "tab_classes.tex",
                "tab_xor_stability.tex", "tab_cost.tex", "tab_entropy.tex", "tab_entropy_full.tex",
                "tab_fpga.tex", "tab_margin.tex"]]
            + [ROOT / "figures" / f for f in
               ["fig_attack_curves.pdf", "fig_stability.pdf", "fig_fpga.pdf", "fig_ecc.pdf",
                "fig_entropy.pdf", "fig_flip_profile.pdf"]]
            + ([SUB / f for f in
                ["manuscript.tex", "manuscript.pdf", "supplement.tex", "supplement.pdf",
                 "highlights.tex", "highlights.pdf", "bibliography.bib",
                 "figures/framework_tikz.tex"]] if PAPER else []))

results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append({"check": name, "passed": bool(ok), "detail": detail})
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))


def rows_without_timing(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: v for k, v in r.items() if k not in TIMING_COLUMNS} for r in csv.DictReader(f)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", default=None)
    args = ap.parse_args()

    # 1
    missing = [str(p.relative_to(ROOT)) for p in EXPECTED if not p.exists()]
    check("expected files exist", not missing, ", ".join(missing))

    # 2
    manifest = json.loads((FULL / "manifest.json").read_text(encoding="utf-8"))
    bad = [f for st in manifest["stages"].values() for f, h in st["files"].items()
           if hashlib.sha256((FULL / f).read_bytes()).hexdigest() != h]
    check("raw files match manifest hashes", not bad, ", ".join(bad))
    man_ext = json.loads((FULL / "manifest_extensions.json").read_text(encoding="utf-8"))
    bad = [f for st in man_ext["stages"].values() for f, h in st["files"].items()
           if hashlib.sha256((FULL / f).read_bytes()).hexdigest() != h]
    check("extension raw files match manifest hashes", not bad, ", ".join(bad))

    # 3
    tmp = ROOT / "build" / "validate_tables"
    tmp.mkdir(parents=True, exist_ok=True)
    rcs = [subprocess.run([sys.executable, str(ROOT / "experiments" / script),
                           "--profile", "full", "--tables-dir", str(tmp)],
                          capture_output=True, text=True)
           for script in ("analyze.py", "analyze_extensions.py")]
    diff = [p.name for p in (ROOT / "tables").glob("*.tex")
            if not (tmp / p.name).exists() or (tmp / p.name).read_bytes() != p.read_bytes()]
    check("tables regenerate identically from raw files",
          all(r.returncode == 0 for r in rcs) and not diff,
          ", ".join(diff) or " ".join(r.stderr[-200:] for r in rcs))
    if PAPER:
        copied = [p.name for p in (ROOT / "tables").glob("*.tex")
                  if not (SUB / "tables" / p.name).exists()
                  or (SUB / "tables" / p.name).read_bytes() != p.read_bytes()]
        check("submission folder holds the current tables", not copied, ", ".join(copied))
    else:
        print("SKIP  checks on the article (no CEE_SUBMISSION folder in this copy)")

    # 4
    tex = (SUB / "manuscript.tex").read_text(encoding="utf-8") if PAPER else ""
    supp = (SUB / "supplement.tex").read_text(encoding="utf-8") if PAPER else ""
    numbers_tex = (ROOT / "tables" / "numbers.tex").read_text(encoding="utf-8")
    defined = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", numbers_tex))
    macros = json.loads((FULL / "summary" / "macros.json").read_text(encoding="utf-8"))
    check("numbers.tex equals macros.json", defined == macros)
    ext_tex = (ROOT / "tables" / "numbers_ext.tex").read_text(encoding="utf-8")
    defined_ext = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", ext_tex))
    macros_ext = json.loads((FULL / "summary" / "macros_ext.json").read_text(encoding="utf-8"))
    check("numbers_ext.tex equals macros_ext.json", defined_ext == macros_ext)
    clash = sorted(set(defined) & set(defined_ext))
    check("no macro is defined twice", not clash, ", ".join(clash))
    all_defined = {**defined, **defined_ext}
    used_any = set(re.findall(r"\\([A-Z][A-Za-z]+)\b", tex + supp))
    known_tex = {"PENDING", "CreditStatement", "CompetingInterests", "FundingStatement",
                 "AcknowledgmentText", "ArtifactLocation", "AIDeclaration", "FloatBarrier"}
    looks_numeric = {m for m in used_any if re.match(
        r"(Num|Budget|Eq|Apuf|Xor|Full|Mask|Masked|Rel|Legacy|Qual|Ecc|Ov|Hw|Xs|Stat|Stab|Pr[A-Z]|"
        r"Repro|Xst|Cls|Cost|Mg|Ent|Fpga)", m)}
    undefined_ext = sorted(looks_numeric - set(all_defined) - known_tex)
    if PAPER:
        check("every numeric macro of the submission is defined", not undefined_ext,
              ", ".join(undefined_ext))
        used = set(re.findall(r"\\([A-Z][A-Za-z]+)\b", tex + supp)) & set(all_defined)
        check("article and supplement use generated numbers", len(used) >= 60,
              f"{len(used)} macros used")

    # 5
    N = json.loads((FULL / "summary" / "numbers.json").read_text(encoding="utf-8"))
    check("claim: arbiter learned by LR above 99 %", N["apuf"]["lr_max"] > 0.99,
          f"{N['apuf']['lr_max']:.4f}")
    check("claim: 2-XOR learned by product LR above 95 %", N["xor2"]["lr_xor_max"] > 0.95,
          f"{N['xor2']['lr_xor_max']:.4f}")
    check("claim: linear LR near chance on 2-XOR (below 60 %)", N["xor2"]["lr_max"] < 0.60,
          f"{N['xor2']['lr_max']:.4f}")
    check("claim: all nine selected masked cells equivalent to chance",
          N["masked"]["selected_all_equivalent"])
    check("claim: masked cell means within 49 to 51 %",
          0.49 <= N["masked"]["cell_mean_min"] and N["masked"]["cell_mean_max"] <= 0.51,
          f"{N['masked']['cell_mean_min']:.4f} to {N['masked']['cell_mean_max']:.4f}")
    check("claim: known-key accuracy equals unmasked accuracy within 0.5 points",
          abs(N["full"]["oracle_max"] - N["xor2"]["lr_xor_max"]) < 0.005
          and abs(N["full"]["oracle_apuf_max"] - N["apuf"]["lr_max"]) < 0.005)
    check("claim: mask cancels in the stability indicator for all devices",
          N["reliability"]["mask_cancels_all"])
    last = max(N["reliability"]["rows"], key=int)
    check("claim: stability attack recovers hidden model above 95 %",
          N["reliability"]["rows"][last]["min"] > 0.95,
          f"worst device {N['reliability']['rows'][last]['min']:.4f}")
    check("claim: decoder success coincides with correction radius in every trial",
          N["ecc"]["decoder_agreement_min"] == 1.0)
    check("claim: simulated i.i.d. failure counts agree with the exact formula "
          "(exact binomial test, Bonferroni over all points)",
          N["ecc"]["min_exact_binomial_p"] > N["ecc"]["bonferroni_threshold"],
          f"smallest p = {N['ecc']['min_exact_binomial_p']:.4f}, threshold "
          f"{N['ecc']['bonferroni_threshold']:.5f}, {N['ecc']['iid_points']} points")
    check("claim: same agreement where both expected counts exceed 10 (|z| < 4)",
          N["ecc"]["max_abs_z_where_counts_are_large"] < 4,
          f"max |z| = {N['ecc']['max_abs_z_where_counts_are_large']:.2f}")
    hw = N["hardware"]["rows"]
    masked_hw = [v["mean"] for k, v in hw.items() if "|masked|" in k]
    check("claim: all masked FPGA cells within 49 to 51 %",
          all(0.49 <= v <= 0.51 for v in masked_hw), f"{min(masked_hw):.4f} to {max(masked_hw):.4f}")
    with open(FULL / "attack_raw.csv", newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    check("protocol: train, validation and test sizes add up and never overlap by construction",
          all(int(r["n_train"]) + int(r["n_val"]) == int(r["budget"]) for r in raw))

    # 5b: version 2 audits
    if PAPER:
        led = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_ledger.py")],
                             capture_output=True, text=True)
        check("ledger: every independently recomputed number equals the manuscript value",
              led.returncode == 0, led.stdout.strip().split("\n")[0][:150])
    cv = json.loads((ROOT / "experiments" / "results" / "code_verification.json").read_text(encoding="utf-8"))
    check("independent verification of codes, decoders and failure probabilities passed",
          cv["passed"], f"{len(cv['checks'])} checks")
    check("ECC threshold is rounded down (exact value not below the printed value)",
          float(defined["EccBestBer"]) <= float(defined["EccBestBerExact"]),
          f"printed {defined['EccBestBer']} %, exact {defined['EccBestBerExact']} %")
    SM = N["stability_models"]
    check("claim: surrogate and warm-started exact estimator recover the model in every grid cell (mean >= 95 %)",
          SM["grid_min_mean"]["logistic"] >= 0.95 and SM["grid_min_mean"]["exact_warm"] >= 0.95,
          f"{SM['grid_min_mean']['logistic']:.4f}, {SM['grid_min_mean']['exact_warm']:.4f}")
    check("claim: exact likelihood predicts the observed unstable fraction (gap below 0.5 point)",
          SM["max_gap_observed_vs_predicted_unstable_fraction"] < 0.005,
          f"{100 * SM['max_gap_observed_vs_predicted_unstable_fraction']:.3f} points")
    check("claim: no single-attack masked cell significant after Holm correction",
          N["statistics"]["single_attack_holm_significant"] == 0)
    P = N["prodlr_restarts"]
    check("claim: product LR restarts are bimodal (none between 55 and 80 %)",
          P["in_between_55_and_80"] == 0, f"{P['converged_above_80']} of {P['restarts_total']} converged")
    # 5c: extension claims
    NE = json.loads((FULL / "summary" / "numbers_ext.json").read_text(encoding="utf-8"))
    xs = NE["xor_stability"]
    top = {k: v for k, v in xs.items() if k.endswith(f"_{max(v_['budget'] for v_ in xs.values())}")}
    check("claim: stability recovers every masked k-XOR model above 95 % at the largest budget",
          all(v["agreement_min"] > 0.95 for v in top.values()),
          ", ".join(f"k={v['k']}: {v['agreement_min']:.4f}" for v in top.values()))
    check("claim: masked responses without key equivalent to chance in every stability cell",
          all(v["no_key_tost_p"] < 0.05 for v in xs.values()))
    check("claim: mask cancels in the stability indicator for every k and device",
          all(v["mask_cancels_all"] for v in xs.values()))
    C = NE["classes"]
    check("claim: class X learned without the secret (above 95 %)", C["X"]["direct_mean"] > 0.95)
    check("claim: classes R, L, P at chance with the PUF replaced (equivalence)",
          all(C[k]["substituted_tost_p"] < 0.05 for k in ("R", "L", "P")))
    check("claim: class H predicted from stability above 90 %, direct attack at chance",
          C["H"]["stability_min"] > 0.9 and C["H"]["direct_tost_p"] < 0.05)
    F = NE["fpga"]
    fb = max(int(k.split("|")[2]) for k in F if k.count("|") == 2)
    fm = [F[f"{kind}|{v}|{fb}"] for kind in ("arbiter", "xor2") for v in ("masked", "mask_only")]
    check("claim: masked FPGA instances equivalent to chance",
          all(x["tost_p"] < 0.05 for x in fm), ", ".join(f"{x['mean']:.4f}" for x in fm))
    check("claim: cost refits reproduce the stored accuracies", NE["convergence"]["all_equal"])
    E = NE["entropy"]
    low45 = [v["0.45"]["lower"] for k, v in E.items() if isinstance(v, dict) and "0.45" in v]
    check("claim: guaranteed key entropy below 128 bits for every code at p1 = 0.45",
          low45 and all(x < 128 for x in low45), ", ".join(f"{x:.1f}" for x in low45))

    # 5d: the built submission
    if PAPER:
        abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.S).group(1)
        words = abstract
        for name in sorted(all_defined, key=len, reverse=True):
            words = words.replace("\\" + name + "{}", all_defined[name])
            words = words.replace("\\" + name, all_defined[name])
        words = re.sub(r"\\[a-zA-Z]+\{?\}?", " ", words.replace("\\,", ""))
        words = re.sub(r"[${}%\\]", " ", words)
        n_words = len(words.split())
        check("abstract has at most 250 words (journal limit, macros expanded)",
              0 < n_words <= 250, f"{n_words} words")
        for doc in ("manuscript", "supplement", "highlights"):
            log = (SUB / f"{doc}.log").read_text(encoding="utf-8", errors="replace")
            src = (SUB / f"{doc}.tex").read_text(encoding="utf-8").splitlines()
            title_line = next((i + 1 for i, t in enumerate(src) if t.strip() == "\\maketitle"), None)
            class_box = f"detected at line {title_line}" if title_line else None
            bad = [ln for ln in log.splitlines() if re.search(r"undefined|Overfull", ln)
                   and not (class_box and ln.startswith("Overfull") and ln.rstrip().endswith(class_box))]
            check(f"{doc}: LaTeX log has no undefined reference or citation and no overfull box",
                  not bad, "; ".join(bad[:3]))
            pdf, srcs = SUB / f"{doc}.pdf", [SUB / f"{doc}.tex"] + list((SUB / "tables").glob("*.tex"))
            check(f"{doc}: PDF is newer than its sources",
                  pdf.exists() and pdf.stat().st_mtime >= max(s.stat().st_mtime for s in srcs))
        hl = re.findall(r"\\item\s+(.*)", (SUB / "highlights.tex").read_text(encoding="utf-8"))
        check("highlights: 3 to 5 items of at most 85 characters",
              3 <= len(hl) <= 5 and all(len(h.strip()) <= 85 for h in hl),
              ", ".join(str(len(h.strip())) for h in hl))
        del abstract

    # 6
    from attacks.models import ATTACK_CONFIG as AC
    want = [f"$C={AC['lr']['C']:g}$", f"at most {AC['lr']['max_iter']} iterations",
            f"$C={AC['svm']['C']:g}$", f"{AC['rf']['n_estimators']} trees",
            f"{AC['gb']['max_iter']} iterations", f"learning rate {AC['gb']['learning_rate']:g}",
            f"{AC['gb']['max_leaf_nodes']} leaves", f"of {AC['mlp']['hidden'][0]} units",
            f"rate {AC['mlp']['lr']:g}", f"batch {AC['mlp']['batch']}",
            f"patience {AC['mlp']['patience']}", f"learning rate {AC['lr_xor']['lr']:g}",
            f"{AC['lr_xor']['restarts']} restarts"]
    expanded = tex
    for name in sorted(all_defined, key=len, reverse=True):
        expanded = expanded.replace("\\" + name + "{}", all_defined[name])
        expanded = re.sub(r"\\" + name + r"(?![A-Za-z])", lambda _m, v=all_defined[name]: v, expanded)
    flat = " ".join(expanded.replace("\\,", "").split())
    absent = [w for w in want if w not in flat]
    if PAPER:
        check("hyperparameters in the manuscript equal the code", not absent, "; ".join(absent))

    # 7
    if args.compare:
        other = Path(args.compare)
        if not other.is_absolute():
            other = ROOT / other
        for f in RAW_CSV:
            if (other / f).exists():
                same = rows_without_timing(FULL / f) == rows_without_timing(other / f)
                check(f"second run identical: {f}", same)
            else:
                check(f"second run identical: {f}", False, "missing in second run")
        for f in RAW_JSON:
            same = (other / f).exists() and json.loads((other / f).read_text(encoding="utf-8")) == \
                json.loads((FULL / f).read_text(encoding="utf-8"))
            check(f"second run identical: {f}", same)

    report = {"passed": all(r["passed"] for r in results), "checks": results,
              "compared_with": args.compare}
    (ROOT / "experiments" / "results" / "validation_report.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n{sum(r['passed'] for r in results)} of {len(results)} checks passed")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
