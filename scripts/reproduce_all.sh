#!/usr/bin/env bash
# Reproduce every result, table, figure and the paper from scratch (POSIX
# shell: Linux, macOS or Git Bash on Windows). On Windows without a POSIX
# shell, use setup_and_run.bat in the project root instead.
#
#   bash scripts/reproduce_all.sh            full run (about 35 minutes on 16 cores)
#   bash scripts/reproduce_all.sh --check    run the experiments a second time
#                                            into experiments/results/rerun and
#                                            compare the two runs
#
# Requirements: Python 3.10 to 3.13, network access on the first run, and a
# TeX distribution with latexmk and elsarticle to recompile the paper (the
# paper step is skipped without it).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else
  python3 -m venv .venv 2>/dev/null || python -m venv .venv
  if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
fi
"$PY" -m pip install --disable-pip-version-check -q -r environment/requirements.txt
JOBS="${JOBS:-14}"

"$PY" -m pytest tests -q
"$PY" scripts/download_public_data.py
"$PY" experiments/run_all.py --config configs/full.json --jobs "$JOBS"
"$PY" experiments/run_hardware.py --jobs "$JOBS"
"$PY" experiments/run_extensions.py --config configs/extensions_full.json --jobs "$JOBS"
"$PY" scripts/check_ecc_tail.py
"$PY" experiments/analyze.py --profile full
"$PY" experiments/analyze_extensions.py --profile full
"$PY" scripts/make_figures.py --profile full
if command -v latexmk >/dev/null 2>&1; then "$PY" scripts/build_cee.py; fi
"$PY" scripts/check_style.py

if [ "${1:-}" = "--check" ]; then
  "$PY" experiments/run_all.py --config configs/full.json --jobs "$JOBS" --out experiments/results/rerun
  "$PY" experiments/run_hardware.py --jobs "$JOBS" --out experiments/results/rerun
  "$PY" experiments/run_extensions.py --config configs/extensions_full.json --jobs "$JOBS" \
        --out experiments/results/rerun
  "$PY" scripts/compare_runs.py experiments/results/full experiments/results/rerun
fi
"$PY" scripts/validate_outputs.py
