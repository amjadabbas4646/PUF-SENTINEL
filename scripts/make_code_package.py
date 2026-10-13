"""Assemble the self-contained research artifact PUF-SENTINEL_code/ and its
zip file from the project folder.

Included: source, tests, configurations, experiment scripts, the raw and
summary results of the reported runs, generated tables and figures, the
article source with its PDFs, the Windows runner and a README. Excluded: the
virtual environment, the public data files (downloaded by script), scratch
builds and earlier versions of the paper.

Batch files are written with CRLF line endings, because cmd.exe misreads
labels and goto in batch files with LF endings.

Usage:  python scripts/make_code_package.py [--out DIR]
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = "PUF-SENTINEL_code"

TREES = ["src", "tests", "configs", "tables", "figures", "CEE_SUBMISSION"]
FILES = ["environment/requirements.txt", "environment/requirements-lock.txt",
         "data/README.md", "setup_and_run.bat",
         "experiments/run_all.py", "experiments/run_hardware.py",
         "experiments/run_extensions.py", "experiments/analyze.py",
         "experiments/analyze_extensions.py",
         "experiments/results/audit_original_code.json",
         "experiments/results/code_verification.json",
         "experiments/results/ecc_tail_check.json",
         "experiments/results/clean_env/comparison_with_stored.json"]
SCRIPTS = ["build_cee.py", "build_cee_bib.py", "build_ledger.py", "check_ecc_tail.py",
           "check_style.py", "check_submission.py", "compare_runs.py", "contact_sheet.py",
           "crossref_lookup.py", "crossref_to_bib.py", "download_public_data.py",
           "inspect_public_data.py", "make_figures.py", "validate_outputs.py",
           "verify_codes_independent.py", "reproduce_all.sh", "make_code_package.py"]
RESULTS = ["experiments/results/full", "experiments/results/smoke"]
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
SKIP_SUFFIX = {".pyc", ".aux", ".fls", ".fdb_latexmk", ".out", ".blg", ".synctex.gz"}


def copy_tree(src: Path, dst: Path) -> None:
    for p in src.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts) or p.suffix in SKIP_SUFFIX:
            continue
        if p.is_file():
            t = dst / p.relative_to(src)
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT))
    args = ap.parse_args()
    out = Path(args.out) / NAME
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for t in TREES + RESULTS:
        copy_tree(ROOT / t, out / t)
    for f in FILES:
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, out / f)
    for s in SCRIPTS:
        (out / "scripts").mkdir(exist_ok=True)
        shutil.copy2(ROOT / "scripts" / s, out / "scripts" / s)
    shutil.copy2(ROOT / "environment" / "README_CODE.md", out / "README.md")
    (out / ".vscode").mkdir(exist_ok=True)
    for f in (ROOT / "environment" / "vscode").glob("*.json"):
        shutil.copy2(f, out / ".vscode" / f.name)
    (out / "data" / "raw").mkdir(parents=True, exist_ok=True)
    for bat in out.rglob("*.bat"):
        text = bat.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        bat.write_bytes(text)
    z = Path(args.out) / f"{NAME}.zip"
    if z.exists():
        z.unlink()
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                zf.write(p, Path(NAME) / p.relative_to(out))
    n = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"{n} files in {out}, zip {z.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
