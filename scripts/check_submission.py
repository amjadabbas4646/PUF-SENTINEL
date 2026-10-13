"""List what still blocks submission of CEE_SUBMISSION: every field that only
the authors can supply (marked PENDING in the sources) and the house style
rules of scripts/check_style.py.

Usage:  python scripts/check_submission.py      exit code 1 while anything is open
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "CEE_SUBMISSION"


def main() -> None:
    open_items = []
    for p in sorted(SUB.glob("*.tex")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("%") or "newcommand{\\PENDING}" in line:
                continue
            for m in re.finditer(r"\\PENDING\{([^}]*)\}|PENDING[-\w ]*", line):
                open_items.append(f"{p.name}:{i}: {m.group(1) or m.group(0)}")
    for item in open_items:
        print("OPEN  " + item)
    style = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_style.py"),
                            *[str(p) for p in SUB.glob("*.tex")]], capture_output=True, text=True)
    print(style.stdout.strip().splitlines()[-1])
    print(f"{len(open_items)} author field(s) open")
    sys.exit(1 if open_items or style.returncode else 0)


if __name__ == "__main__":
    main()
