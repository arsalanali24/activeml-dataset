#!/usr/bin/env python3
"""
compile_benchmark_table.py
===========================
After the DMRG benchmark jobs finish, compile the full comparison table:

  System | Published | ML | CASSCF(10,10) | ASF/DMRG | DMRG-direct(s) | DMRG-direct(N) | ASF vs Direct

Usage:
    python3 compile_benchmark_table.py
"""
import json, glob, numpy as np

SCRATCH = "/scratch/hpc-prf-qehpc/hpcmual/dmrg_labels"

# Ground truth from DMRG_Benchmark_Results.md
KNOWN = {
    "CrCl6_3m_oct":    {"pub": 4, "ml": 4, "casscf": 4, "asf": 7},
    "RhCl6_3m_oct":    {"pub": 6, "ml": 4, "casscf": 5, "asf": 4},
    "MoCl6_3m_oct":    {"pub": 3, "ml": 4, "casscf": 3, "asf": 5},
    "ReCl6_2m_oct":    {"pub": 3, "ml": 4, "casscf": 3, "asf": 3},
    "CoNH3_6_3p_oct":  {"pub": 6, "ml": 7, "casscf": 6, "asf": 6},
    "CrCl6_3m_CSD":    {"pub": 3, "ml": 3, "casscf": 3, "asf": 7},
    "FeCl4_1m_CSD":    {"pub": 5, "ml": 4, "casscf": 5, "asf": 6},
    "MnCl4_2m_CSD":    {"pub": 5, "ml": 4, "casscf": 5, "asf": 7},
    "OsCl6_2m_CSD":    {"pub": 4, "ml": 3, "casscf": 4, "asf": 6},
    "FeNH3_6_2p_CSD":  {"pub": 6, "ml": 7, "casscf": 6, "asf": 12},
}

# Also check for the already-completed CrCl6 from first test run
ALIASES = {
    "CrCl6_3m_oct_d3": "CrCl6_3m_oct",
    "CrCl6_3m_CSD":    "CrCl6_3m_CSD",
}

print("=" * 95)
print(f"{'System':<22} {'Pub':>4} {'ML':>4} {'CASSCF':>7} {'ASF':>5} "
      f"{'DMRG-ε':>7} {'DMRG-N':>7} {'ASF=ε?':>7} {'Δ(CASSCF-ε)':>12}")
print("-" * 95)

rows = []
for label, ref in KNOWN.items():
    # Find DMRG output
    pattern = f"{SCRATCH}/{label}/dmrg_features.json"
    files = glob.glob(pattern)

    # Try alias (e.g. CrCl6_3m_oct_d3 → CrCl6_3m_oct)
    if not files:
        for alias, canonical in ALIASES.items():
            if canonical == label:
                files = glob.glob(f"{SCRATCH}/{alias}/dmrg_features.json")

    if not files:
        dmrg_e = "—"
        dmrg_n = "—"
        match = "—"
        delta = "—"
    else:
        d = json.load(open(files[0]))
        noon = np.array(d["noon"])
        s1   = np.array(d["s1"])
        dmrg_e = int((s1 > 0.138).sum())
        dmrg_n = int(sum(1 for n in noon if 0.02 < n < 1.98))
        match  = "✓" if dmrg_e == ref["asf"] else "✗"
        delta  = dmrg_e - ref["casscf"]

    rows.append((label, ref["pub"], ref["ml"], ref["casscf"],
                 ref["asf"], dmrg_e, dmrg_n, match, delta))
    print(f"{label:<22} {ref['pub']:>4} {ref['ml']:>4} {ref['casscf']:>7} "
          f"{ref['asf']:>5} {str(dmrg_e):>7} {str(dmrg_n):>7} "
          f"{str(match):>7} {str(delta):>12}")

print("=" * 95)

# Summary
done = [r for r in rows if r[4] != "—"]
if done:
    matches = sum(1 for r in done if r[7] == "✓")
    deltas  = [r[8] for r in done if isinstance(r[8], int)]
    print(f"\nDMRG-direct matches ASF:  {matches}/{len(done)}")
    if deltas:
        print(f"Mean DMRG-ε vs CASSCF:   {np.mean(deltas):+.1f} orbitals")
        print(f"DMRG-ε > CASSCF:          {sum(d < 0 for d in deltas)}/{len(deltas)} systems")
        print(f"DMRG-ε = CASSCF:          {sum(d == 0 for d in deltas)}/{len(deltas)} systems")
        print(f"DMRG-ε < CASSCF:          {sum(d > 0 for d in deltas)}/{len(deltas)} systems")

remaining = [r[0] for r in rows if r[4] == "—"]
if remaining:
    print(f"\nStill pending ({len(remaining)}): {', '.join(remaining)}")
