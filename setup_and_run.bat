@echo off
rem ==========================================================================
rem  PUF-SENTINEL: set up a Python environment and run the artifact on Windows
rem
rem  Double-click this file, or run it from a Command Prompt:
rem
rem      setup_and_run.bat            asks whether to run the full experiments
rem      setup_and_run.bat smoke      tests, smoke run, tables, figures, checks
rem      setup_and_run.bat full       the same, then every experiment again
rem
rem  Add "nopause" as a second argument to close the window without a pause.
rem  Needs Python 3.10, 3.11, 3.12 or 3.13 and internet access on the first
rem  run (Python packages and seven public data files, about 75 MB).
rem ==========================================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set "MODE=%~1"
set "NOPAUSE=%~2"
set "STEP=start"

set /a JOBS=%NUMBER_OF_PROCESSORS%-2
if %JOBS% LSS 1 set JOBS=1

rem Windows limits most paths to 260 characters. Some installed packages nest
rem files about 150 characters deep inside .venv, so a long folder path makes
rem the installation fail with "WinError 206". Stop early with advice instead.
set "STEP=check the folder path"
set "HERE=%~dp0"
if not "!HERE:~90!"=="" (
    echo.
    echo This folder path is too long for Windows:
    echo   %~dp0
    echo Move the folder to a short path, for example C:\PUF-SENTINEL_code,
    echo and start setup_and_run.bat again.
    goto :fail
)

rem --------------------------------------------------------------- 1. Python
set "STEP=find Python"
set "PYBOOT="
for %%V in (3.12 3.11 3.13 3.10) do (
    if not defined PYBOOT (
        py -%%V -c "import sys" >nul 2>&1 && set "PYBOOT=py -%%V"
    )
)
if not defined PYBOOT (
    python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>&1 && set "PYBOOT=python"
)
if not defined PYBOOT (
    echo.
    echo No suitable Python found. Install Python 3.12 from https://www.python.org
    echo and tick "Add python.exe to PATH" during installation.
    goto :fail
)
echo [1/7] Python: %PYBOOT%

rem ---------------------------------------------------- 2. virtual environment
rem A virtual environment stops working when its folder is moved or when the
rem Python it was made from is removed. Test it and rebuild it if needed.
set "STEP=create the virtual environment"
set "VPY=%~dp0.venv\Scripts\python.exe"
if exist "%VPY%" (
    "%VPY%" -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo Existing .venv does not work here. Rebuilding it.
        rmdir /s /q "%~dp0.venv"
    )
)
if not exist "%VPY%" (
    echo [2/7] Creating .venv
    %PYBOOT% -m venv "%~dp0.venv" || goto :fail
) else (
    echo [2/7] Using existing .venv
)

set "STEP=install the pinned packages"
echo [3/7] Installing pinned packages (the first time this downloads about 300 MB)
"%VPY%" -m pip install --disable-pip-version-check -q -r environment\requirements.txt || goto :fail

rem ----------------------------------------------------------------- 3. tests
set "STEP=unit tests"
echo [4/7] Unit tests
"%VPY%" -m pytest tests -q || goto :fail

rem ------------------------------------------------------------ 4. public data
set "STEP=download and verify the public FPGA data"
echo [5/7] Public FPGA data (Zenodo 10.5281/zenodo.5221305)
"%VPY%" scripts\download_public_data.py || goto :fail

rem ------------------------------------------------------------- 5. smoke run
set "STEP=smoke run"
echo [6/7] Smoke run of every experiment with tiny settings (a few minutes)
"%VPY%" experiments\run_all.py --config configs/smoke.json --jobs %JOBS% || goto :fail
"%VPY%" experiments\run_hardware.py --smoke --jobs %JOBS% || goto :fail
"%VPY%" experiments\run_extensions.py --config configs/extensions_smoke.json --jobs %JOBS% || goto :fail
"%VPY%" experiments\analyze.py --profile smoke || goto :fail
"%VPY%" experiments\analyze_extensions.py --profile smoke || goto :fail
"%VPY%" scripts\make_figures.py --profile smoke || goto :fail

rem ------------------------------------- 6. paper tables, figures and checks
rem Regenerates every table, number and figure of the paper from the stored
rem raw results, recompiles the paper if a TeX installation is present and
rem runs the consistency checks.
set "STEP=regenerate tables and figures of the paper"
echo [7/7] Tables, figures and checks from the stored results
"%VPY%" experiments\analyze.py --profile full || goto :fail
"%VPY%" experiments\analyze_extensions.py --profile full || goto :fail
"%VPY%" scripts\make_figures.py --profile full || goto :fail
where latexmk >nul 2>&1
if errorlevel 1 (
    echo latexmk not found: the paper is not recompiled. The PDFs in CEE_SUBMISSION are used.
) else (
    set "STEP=compile the paper"
    "%VPY%" scripts\build_cee.py || goto :fail
)
set "STEP=consistency checks"
"%VPY%" scripts\validate_outputs.py || goto :fail

rem ------------------------------------------------------- 7. full experiments
if /i "%MODE%"=="smoke" goto :done
if /i not "%MODE%"=="full" (
    echo.
    echo The full experiments take about 35 minutes on 16 cores and write to
    echo experiments\results\rerun. The stored results are not changed.
    set "ANSWER=n"
    set /p "ANSWER=Run the full experiments now? [y/N] "
    if /i not "!ANSWER!"=="y" goto :done
)
set "STEP=full experiments"
echo Full experiments into experiments\results\rerun
"%VPY%" experiments\run_all.py --config configs/full.json --jobs %JOBS% --out experiments/results/rerun || goto :fail
"%VPY%" experiments\run_hardware.py --jobs %JOBS% --out experiments/results/rerun || goto :fail
"%VPY%" experiments\run_extensions.py --config configs/extensions_full.json --jobs %JOBS% --out experiments/results/rerun || goto :fail
set "STEP=compare the rerun with the stored results"
"%VPY%" scripts\compare_runs.py experiments/results/full experiments/results/rerun || goto :fail

:done
echo.
echo All steps finished without error.
if /i not "%NOPAUSE%"=="nopause" pause
endlocal
exit /b 0

:fail
echo.
echo FAILED at step: %STEP%
echo Please send the messages above this line to the authors.
if /i not "%NOPAUSE%"=="nopause" pause
endlocal
exit /b 1
