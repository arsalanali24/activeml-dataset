# DMRG Benchmark Scripts

Scripts for comparing active space selection methods on 10 benchmark transition metal complexes.

## Methods compared
- CASSCF(10,10) — existing labels
- ASF v2.0.2 + block2 v0.5.3 — DMRG-CASSCF via HQS ActiveSpaceFinder
- DMRG-CI direct — block2 Python API with UHF energy window
- DMRG-CI v2 — block2 Python API with MP2 natural orbital window

## Results (ASF, job 33323926)
| System | CASSCF | ASF |
|--------|--------|-----|
| CrCl6_3m_oct | 4 | 7 |
| RhCl6_3m_oct | 5 | 4 |
| MoCl6_3m_oct | 3 | 5 |
| ReCl6_2m_oct | 3 | 3 |
| CoNH3_6_3p_oct | 6 | 6 |
| CrCl6_3m_CSD | 3 | 7 |
| FeCl4_1m_CSD | 5 | 6 |
| MnCl4_2m_CSD | 5 | 7 |
| OsCl6_2m_CSD | 4 | 6 |
| FeNH3_6_2p_CSD | 6 | 12 |

## Usage
1. `python create_benchmark_jsons.py` — generate JSON input files
2. `sbatch submit_asf_all.sh` — run ASF benchmark
3. `python compile_benchmark_table.py` — compile comparison table
