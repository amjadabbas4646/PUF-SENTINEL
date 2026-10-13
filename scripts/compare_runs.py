"""Compare the raw outputs of two runs, ignoring wall-clock columns, and write a
JSON report next to the second run.
Usage: python scripts/compare_runs.py experiments/results/full experiments/results/clean_env"""
import csv
import hashlib
import json
import sys
from pathlib import Path

a, b = Path(sys.argv[1]), Path(sys.argv[2])
report = {"reference": str(a), "candidate": str(b), "files": {}}
ok = True
for fa in sorted(a.glob("*.csv")) + sorted(a.glob("*.json")):
    fb = b / fa.name
    if fa.name in ("manifest.json", "manifest_extensions.json", "overhead.json",
                   "comparison_with_stored.json"):
        continue                                   # versions, timings, this report
    if not fb.exists():
        print(f"MISSING  {fa.name}")
        report["files"][fa.name] = "missing"
        ok = False
        continue
    if fa.suffix == ".csv":
        ra = [{k: v for k, v in r.items() if k != "seconds"} for r in csv.DictReader(open(fa, newline="", encoding="utf-8"))]
        rb = [{k: v for k, v in r.items() if k != "seconds"} for r in csv.DictReader(open(fb, newline="", encoding="utf-8"))]
        same = ra == rb
        diff = sum(x != y for x, y in zip(ra, rb)) + abs(len(ra) - len(rb))
        print(f"{'SAME    ' if same else 'DIFFERS '} {fa.name}  rows={len(ra)} differing_rows={diff}")
        digest = hashlib.sha256(json.dumps(ra, sort_keys=True).encode()).hexdigest()
        report["files"][fa.name] = {"identical": same, "rows": len(ra), "differing_rows": diff,
                                    "sha256_of_content_without_timing": digest}
    else:
        same = json.loads(fa.read_text(encoding="utf-8")) == json.loads(fb.read_text(encoding="utf-8"))
        print(f"{'SAME    ' if same else 'DIFFERS '} {fa.name}")
        report["files"][fa.name] = {"identical": same}
    ok &= same
report["files_compared"] = len(report["files"])
report["files_identical"] = sum(1 for v in report["files"].values()
                                if isinstance(v, dict) and v["identical"])
report["all_identical"] = bool(ok)
(b / "comparison_with_stored.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
sys.exit(0 if ok else 1)
