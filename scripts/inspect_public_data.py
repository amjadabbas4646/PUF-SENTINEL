"""Print the structure of the downloaded npz files (keys, shapes, dtypes,
value sets, duplicate challenges, response bias). Loads with allow_pickle
disabled, so the files are treated as plain arrays."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

for path in sorted(RAW.glob("*.npz")):
    z = np.load(path, allow_pickle=False)
    print("==", path.name)
    for key in z.files:
        a = z[key]
        vals = np.unique(a[:20000])
        print(f"  {key}: shape={a.shape} dtype={a.dtype} values={vals[:6]}")
    c = z["challenges"]
    r = z["responses"] if "responses" in z.files else z["information"]
    packed = np.packbits(c[:200000] > 0, axis=1)
    print("  duplicate challenges in first 200000:",
          200000 - np.unique(packed, axis=0).shape[0])
    print("  mean response:", float(np.mean(r)), " mean challenge:", float(np.mean(c)))
sys.stdout.flush()
