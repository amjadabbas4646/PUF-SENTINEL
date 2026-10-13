"""Assemble and compile the Computers and Electrical Engineering submission.

Steps
    1. rebuild CEE_SUBMISSION/bibliography.bib (scripts/build_cee_bib.py)
    2. copy the generated tables (tables/*.tex) and figures (figures/*.pdf)
       into CEE_SUBMISSION, so that the folder compiles on its own
    3. compile manuscript.tex, supplement.tex and highlights.tex with latexmk

A compile counts as successful only if latexmk returns 0 and the PDF is newer
than every source it depends on. latexmk refuses to rerun after a failed
invocation and leaves the previous PDF on disk, so on failure the auxiliary
files are removed with latexmk -C and the compile is retried once.

Usage:  python scripts/build_cee.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "CEE_SUBMISSION"
DOCS = ["supplement", "manuscript", "highlights"]   # the manuscript reads supplement.aux (xr)


def copy_inputs() -> None:
    (SUB / "tables").mkdir(exist_ok=True)
    (SUB / "figures").mkdir(exist_ok=True)
    for f in (ROOT / "tables").glob("*.tex"):
        shutil.copy2(f, SUB / "tables" / f.name)
    for f in (ROOT / "figures").glob("fig_*.pdf"):
        shutil.copy2(f, SUB / "figures" / f.name)


def sources() -> list[Path]:
    out = [p for p in SUB.glob("*.tex")] + [SUB / "bibliography.bib"]
    out += list((SUB / "tables").glob("*.tex")) + list((SUB / "figures").glob("*"))
    return out


def latexmk(doc: str, clean: bool = False) -> int:
    if clean:
        subprocess.run(["latexmk", "-C", f"{doc}.tex"], cwd=SUB, capture_output=True)
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                        f"{doc}.tex"], cwd=SUB, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.returncode


def fresh(doc: str) -> bool:
    pdf = SUB / f"{doc}.pdf"
    if not pdf.exists():
        return False
    newest = max(p.stat().st_mtime for p in sources() if p.exists())
    return pdf.stat().st_mtime >= newest


def main() -> None:
    # the bibliography is rebuilt only where its sources exist (the project
    # folder); the code package ships the finished bibliography.bib
    if (ROOT / "build" / "new_entries_r2.bib").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_cee_bib.py")], check=True)
    copy_inputs()
    failed = []
    for doc in DOCS:
        rc = latexmk(doc)
        if rc != 0 or not fresh(doc):
            rc = latexmk(doc, clean=True)
        ok = rc == 0 and fresh(doc)
        print(f"{doc}.pdf: {'ok' if ok else 'FAILED (see ' + doc + '.log)'}")
        if not ok:
            failed.append(doc)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
