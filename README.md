# PUF-SENTINEL

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amjadabbas4646/PUF-SENTINEL/blob/main/run_in_colab.ipynb)
[![check](https://github.com/amjadabbas4646/PUF-SENTINEL/actions/workflows/check.yml/badge.svg)](https://github.com/amjadabbas4646/PUF-SENTINEL/actions/workflows/check.yml)

**One click:** press *Open in Colab* above, then choose *Runtime > Run all*. The notebook installs the
packages, runs the unit tests, runs every experiment with small settings, regenerates every table and
figure of the article from the stored raw results, shows the figures and runs the validation. Nothing has
to be installed locally. The *check* badge shows the result of the same steps run automatically by GitHub
on every change.


This folder is the research artifact of the article "Controls that separate keyed output functions from modelling resistance in strong PUFs: a benchmark with published obfuscation classes and measured FPGA data". It contains the source code, the configurations, the raw outputs of every reported run, the scripts that regenerate every table, figure and number of the article and the article source.

## Quick start on Windows

1. Install Python 3.12 from https://www.python.org (tick "Add python.exe to PATH"). Python 3.10, 3.11 and 3.13 also work. Python 3.14 does not, because PyTorch 2.7.1 has no build for it.
2. Unpack the zip file into a short folder, for example `D:\PUF-SENTINEL_code`. The path may contain spaces but must stay under 90 characters.
3. Double-click `setup_and_run.bat`.

## Running it in Visual Studio Code

1. Open VS Code, choose File > Open Folder and select this folder. If VS Code asks whether you trust the authors of the files, answer yes, because tasks run only in trusted folders.
2. Choose Terminal > Run Task... and pick a task:
   * `1. Setup and quick run`: run this once first. It builds `.venv`, runs the tests, downloads the data and runs every experiment with tiny settings.
   * `2. Run the unit tests`
   * `3. Regenerate all tables and figures of the article`: writes `tables/` and `figures/` from the stored results.
   * `4. Check every number, table and claim`: the validation, 48 checks.
   * `5. Compile the article PDF`: needs a TeX distribution.
   * `6. Run all experiments again`: about 35 minutes, writes to `experiments\results\rerun`.
3. The output appears in the terminal panel at the bottom. Ctrl+Shift+B starts task 1.

With the Python extension of VS Code installed, any script can also be run or debugged with F5; the interpreter in `.venv` is selected automatically.

The script creates a virtual environment in `.venv`, installs the pinned packages, runs the unit tests, downloads and verifies the public FPGA data, runs every experiment with tiny settings, regenerates every table and figure of the article from the stored results and checks them. It then asks whether to run the full experiments again (about 35 minutes on 16 cores). Each step prints its name and a failure stops the script with the name of the failed step.

From a Command Prompt in this folder:

```bat
setup_and_run.bat smoke
```

runs everything except the full experiments and

```bat
setup_and_run.bat full
```

also runs the full experiments into `experiments\results\rerun` and compares them with the stored results. The stored results are never overwritten.

## Linux, macOS or Git Bash

```bash
bash scripts/reproduce_all.sh
```

## Single commands

All commands start in this folder. On Linux and macOS write `.venv/bin/python` in place of `.venv\Scripts\python.exe`.

```bat
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe scripts\download_public_data.py
.venv\Scripts\python.exe experiments\run_all.py --config configs/full.json --jobs 14
.venv\Scripts\python.exe experiments\run_hardware.py --jobs 14
.venv\Scripts\python.exe experiments\run_extensions.py --config configs/extensions_full.json --jobs 14
.venv\Scripts\python.exe experiments\analyze.py --profile full
.venv\Scripts\python.exe experiments\analyze_extensions.py --profile full
.venv\Scripts\python.exe scripts\make_figures.py --profile full
.venv\Scripts\python.exe scripts\validate_outputs.py
.venv\Scripts\python.exe scripts\build_cee.py
```

The last command recompiles the article and needs a TeX distribution with `latexmk`; the Elsevier class files `cas-dc.cls` and `cas-common.sty` are included in `CEE_SUBMISSION`. Use `python -m pytest` and `python -m pip` rather than `pytest.exe` or `pip.exe`: the launchers in `.venv\Scripts` stop working when a virtual environment is moved to another folder, while `python -m` always works.

## Contents

| Path | Content |
|---|---|
| `src/` | PUF models, obfuscation classes, keyed mask, attacks, error correction, entropy bounds, metrics |
| `experiments/` | `run_all.py` (main study), `run_hardware.py` (measured 4-XOR and arbiter data), `run_extensions.py` (classes, stability, cost, entropy, FPGA instances), `analyze*.py` |
| `experiments/results/full/` | raw outputs of every reported run, manifests with SHA-256 hashes and package versions and `summary/` |
| `configs/` | `full.json` and `extensions_full.json` (article), `smoke.json` and `extensions_smoke.json` (quick check) |
| `tables/`, `figures/` | generated LaTeX tables, number macros and PDF figures |
| `tests/` | unit tests |
| `scripts/` | data download, figures, validation, style and reference checks, article build |
| `environment/` | pinned requirements and the full lock file |
| `data/` | the public FPGA data are downloaded into `data/raw/` and are not redistributed |

## Data

The measured FPGA responses come from Zenodo record 10.5281/zenodo.5221305 (pypuf data, CC BY 4.0): one 64-stage arbiter PUF and five 65-stage arbiter chains measured by Aghaie and Moradi and one 64-stage 4-XOR arbiter PUF measured by Mursi et al. `scripts/download_public_data.py` downloads the seven files and verifies their published MD5 checksums. All other data are simulated from a master seed and a path of labels, so every run is deterministic.

## Expected results

The smoke run uses tiny settings and its numbers are not those of the article. The article numbers come from `experiments/results/full`. `scripts/validate_outputs.py` regenerates every table from the raw files and checks that the result is byte for byte identical, that every number in the article is a generated macro, that the headline claims hold in the stored numbers and that the article PDFs are newer than their sources. The full rerun was found identical to the stored results on the machine of the authors, apart from wall-clock timings. Reproduction on another machine may differ in the last digits of floating-point results, because the linear algebra libraries differ.
