"""Download the public FPGA challenge-response sets used in experiment E5
and verify them against the MD5 checksums published by Zenodo.

Source   Zenodo record 10.5281/zenodo.5221305, "pypuf data", N. Wisiol, 2021
License  Creative Commons Attribution 4.0 International (CC BY 4.0)
Files    AM21_top.npz
             64-stage arbiter PUF on FPGA, 1,000,000 CRPs, one measurement
             each. Measured by Aghaie and Moradi, TCHES 2021(3),
             doi 10.46586/tches.v2021.i3.520-551.
         MTZAA20_4XOR_64bit_LUT_2239B_attacking_1M.txt.npz
             64-stage 4-XOR arbiter PUF on FPGA, 1,000,000 CRPs, one
             measurement each. Measured by Mursi et al., Electronics 9(10),
             1715, 2020, doi 10.3390/electronics9101715.
         AM21_b0.npz to AM21_b4.npz
             five 65-stage arbiter chains on FPGA that answered one common
             challenge set, 1,000,000 CRPs each, one measurement each.
             Measured by Aghaie and Moradi (as above). Used for the measured
             arbiter instances and the software-composed 2-XOR instances.

The files are written to data/raw/, which is not redistributed with this
project. Re-run this script to obtain them.

Usage:  python scripts/download_public_data.py
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
BASE = "https://zenodo.org/records/5221305/files/"
FILES = {
    "AM21_top.npz": {"md5": "1da6ca3aaf6c17f61a2e39093d196487", "bytes": 10431915},
    "MTZAA20_4XOR_64bit_LUT_2239B_attacking_1M.txt.npz":
        {"md5": "d2132e2a6b99cbb6aa98425bcc35aff4", "bytes": 10339164},
    "AM21_b0.npz": {"md5": "5d52e67d6372fd88edd33a5c9fca63a2", "bytes": 10596494},
    "AM21_b1.npz": {"md5": "df382569c5c7bd517b59c3b2cb0df680", "bytes": 10593719},
    "AM21_b2.npz": {"md5": "b428b123f49ca7f1fd1898ceb1b1634c", "bytes": 10590830},
    "AM21_b3.npz": {"md5": "c9b72f254c7b6013c789938cc6b5692f", "bytes": 10608185},
    "AM21_b4.npz": {"md5": "32d6f09f393ef356558beabf411fa5eb", "bytes": 10604676},
}


WAITS = (0, 15, 45, 90, 180)        # seconds before each attempt; Zenodo sometimes answers 5xx


def fetch(name: str, path: Path) -> None:
    """Download one file with retries. The data go to a .part file that is
    renamed only when complete, so an interrupted download never leaves a
    truncated file behind."""
    url = BASE + name + "?download=1"
    part = path.with_suffix(path.suffix + ".part")
    last = None
    for i, wait in enumerate(WAITS, 1):
        if wait:
            print(f"  retry {i} of {len(WAITS)} for {name} in {wait} s ({last})")
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "puf-sentinel/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r:
                part.write_bytes(r.read())
            part.replace(path)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e
    raise SystemExit(f"could not download {name} after {len(WAITS)} attempts ({last}).\n"
                     f"Zenodo may be busy: run the script again later, or download\n  {url}\n"
                     f"in a browser and save it as {path}")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    log = {"record_doi": "10.5281/zenodo.5221305", "license": "CC-BY-4.0", "files": {}}
    for name, meta in FILES.items():
        path = RAW / name
        if not path.exists():
            fetch(name, path)
        data = path.read_bytes()
        md5 = hashlib.md5(data).hexdigest()
        ok = md5 == meta["md5"] and len(data) == meta["bytes"]
        log["files"][name] = {"bytes": len(data), "md5": md5,
                              "sha256": hashlib.sha256(data).hexdigest(),
                              "matches_zenodo_checksum": ok,
                              "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                             time.gmtime(path.stat().st_mtime))}
        print(name, len(data), md5, "OK" if ok else "CHECKSUM MISMATCH")
        if not ok:
            raise SystemExit(f"checksum mismatch for {name}")
    (RAW / "download_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
