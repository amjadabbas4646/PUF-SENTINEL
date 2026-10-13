# Data folder

| Folder | Content | Status |
|---|---|---|
| `raw/` | seven public FPGA CRP files and `download_log.json` | created by `scripts/download_public_data.py`. Not redistributed |
| `processed/` | does not exist | not needed. The hardware experiment reads the raw files directly and applies a documented mapping in memory |
| `synthetic/` | does not exist | not needed. Synthetic CRPs are generated at run time from seeds and are never stored |

There are no hidden data files. Every number in the paper comes from `experiments/results/full/`, which holds the per-run outputs of the experiments.

The public files come from Zenodo record 10.5281/zenodo.5221305 under the CC BY 4.0 licence. Checksums are listed in `scripts/download_public_data.py` and the article describes the data and their limits.

## Redistributed public data

`raw/` contains the seven files of Zenodo record 10.5281/zenodo.5221305 (pypuf data, N. Wisiol, 2021),
redistributed unchanged under its Creative Commons Attribution 4.0 licence: `AM21_top.npz` and
`AM21_b0.npz` to `AM21_b4.npz` measured by Aghaie and Moradi (TCHES 2021(3), doi
10.46586/tches.v2021.i3.520-551) and `MTZAA20_4XOR_64bit_LUT_2239B_attacking_1M.txt.npz` measured by
Mursi et al. (Electronics 9(10) 1715, 2020, doi 10.3390/electronics9101715).
