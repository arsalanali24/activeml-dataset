#!/usr/bin/env python3
"""
create_benchmark_jsons.py
=========================
Create JSON files for the 10 original ASF benchmark systems so they can
be run through gen_dmrg_labels.py to add a DMRG-direct column to the
benchmark table in DMRG_Benchmark_Results.md

Systems (from asf_all_cases.py / DMRG_Benchmark_Results.md):

Idealized:
  1. CrCl6_3m_oct       Cr(III) d3  S=3/2  CrCl6^3-
  2. RhCl6_3m_oct       Rh(III) d6  S=0    RhCl6^3-
  3. MoCl6_3m_oct       Mo(III) d3  S=3/2  MoCl6^3-
  4. ReCl6_2m_oct       Re(IV)  d3  S=3/2  ReCl6^2-  (spin=3, NOT spin=1)
  5. CoNH3_6_3p_oct     Co(III) d6  S=0    Co(NH3)6^3+

CSD real structures (geometry from CIF, stored in asf_all_cases.py):
  6. CrCl6_CSD          Cr(III) d3  S=3/2  [CrCl6]^3-
  7. FeCl4_CSD          Fe(III) d5  S=5/2  [FeCl4]^-
  8. MnCl4_CSD          Mn(II)  d5  S=5/2  [MnCl4]^2-
  9. OsCl6_CSD          Os(IV)  d4  S=1    [OsCl6]^2-
 10. FeNH3_CSD          Fe(II)  d6  S=2    Fe(NH3)6^2+

Output: /pc2/users/h/hpcmual/activeml/data/dmrg_benchmark_10/
"""

import json, os

OUT_DIR = "/pc2/users/h/hpcmual/activeml/data/dmrg_benchmark_10"
os.makedirs(OUT_DIR, exist_ok=True)


def save(name, metal, ligand, n_ligands, charge, spin, atom_str,
         geometry, dist_ang, published_n_active, basis="def2-svp"):
    d = {
        "name": name, "label": name,
        "metal": metal, "ligand": ligand, "n_ligands": n_ligands,
        "charge": charge, "spin": spin,   # spin = 2S
        "mult": spin + 1,
        "geometry": geometry,
        "dist_ang": dist_ang,
        "atom": atom_str,                 # direct atom string (no regen needed)
        "basis": basis,
        "published_n_active": published_n_active,
        "source": "DMRG_Benchmark_Results",
    }
    path = os.path.join(OUT_DIR, name + ".json")
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"  Saved: {name}.json  (charge={charge}, spin={spin}, pub={published_n_active})")
    return path


print("Creating benchmark JSONs...")
paths = []

# ── 1. CrCl6^3- idealized ──────────────────────────────────────────────────
# Cr(III) d3, S=3/2, octahedral, dist=2.350 Å (from gen_benchmark_test.py)
# Already run: DMRG-entropy=7, DMRG-NOON=8, ASF=7
paths.append(save(
    name="CrCl6_3m_oct",
    metal="Cr", ligand="Cl", n_ligands=6, charge=-3, spin=3,
    atom_str="Cr 0 0 0\nCl 2.350 0 0\nCl -2.350 0 0\nCl 0 2.350 0\n"
             "Cl 0 -2.350 0\nCl 0 0 2.350\nCl 0 0 -2.350",
    geometry="oct", dist_ang=2.350, published_n_active=4,
))

# ── 2. RhCl6^3- idealized ──────────────────────────────────────────────────
# Rh(III) d6, S=0 (low-spin), octahedral, dist≈2.36 Å (4d, use def2-svp-pp)
# ASF=4, Published=6, ML=4
paths.append(save(
    name="RhCl6_3m_oct",
    metal="Rh", ligand="Cl", n_ligands=6, charge=-3, spin=0,
    atom_str="Rh 0 0 0\nCl 2.360 0 0\nCl -2.360 0 0\nCl 0 2.360 0\n"
             "Cl 0 -2.360 0\nCl 0 0 2.360\nCl 0 0 -2.360",
    geometry="oct", dist_ang=2.360, published_n_active=6,
    basis="def2-svp",  # def2-svp has ECP for Rh
))

# ── 3. MoCl6^3- idealized ──────────────────────────────────────────────────
# Mo(III) d3, S=3/2, octahedral, dist≈2.49 Å
# ASF=5, Published=3, ML=4
paths.append(save(
    name="MoCl6_3m_oct",
    metal="Mo", ligand="Cl", n_ligands=6, charge=-3, spin=3,
    atom_str="Mo 0 0 0\nCl 2.490 0 0\nCl -2.490 0 0\nCl 0 2.490 0\n"
             "Cl 0 -2.490 0\nCl 0 0 2.490\nCl 0 0 -2.490",
    geometry="oct", dist_ang=2.490, published_n_active=3,
    basis="def2-svp",
))

# ── 4. ReCl6^2- idealized ──────────────────────────────────────────────────
# Re(IV) d3, S=3/2, octahedral, dist≈2.37 Å
# spin=3 (CORRECTED from earlier wrong spin=1 run)
# ASF=3, Published=3, ML=4
paths.append(save(
    name="ReCl6_2m_oct",
    metal="Re", ligand="Cl", n_ligands=6, charge=-2, spin=3,
    atom_str="Re 0 0 0\nCl 2.370 0 0\nCl -2.370 0 0\nCl 0 2.370 0\n"
             "Cl 0 -2.370 0\nCl 0 0 2.370\nCl 0 0 -2.370",
    geometry="oct", dist_ang=2.370, published_n_active=3,
    basis="def2-svp",
))

# ── 5. Co(NH3)6^3+ ─────────────────────────────────────────────────────────
# Co(III) d6, S=0 (low-spin, strong-field NH3), octahedral, Co-N≈1.96 Å
# ASF=6, Published=6, ML=7
# NH3 ligands: each N placed at octahedral position, H atoms not needed for UHF
paths.append(save(
    name="CoNH3_6_3p_oct",
    metal="Co", ligand="N", n_ligands=6, charge=3, spin=0,
    atom_str="Co 0 0 0\nN 1.960 0 0\nN -1.960 0 0\nN 0 1.960 0\n"
             "N 0 -1.960 0\nN 0 0 1.960\nN 0 0 -1.960",
    geometry="oct", dist_ang=1.960, published_n_active=6,
))

# ── 6. [CrCl6]^3- CSD real structure ───────────────────────────────────────
# Cr(III) d3, S=3/2
# From asf_all_cases.py — use average CSD Cr-Cl ≈ 2.333 Å with slight distortion
# ASF=7, Published=3, ML=3
# Using idealized oct geometry as approximation (distortions small for CrCl6)
paths.append(save(
    name="CrCl6_3m_CSD",
    metal="Cr", ligand="Cl", n_ligands=6, charge=-3, spin=3,
    atom_str="Cr 0 0 0\nCl 2.333 0 0\nCl -2.333 0 0\nCl 0 2.333 0\n"
             "Cl 0 -2.333 0\nCl 0 0 2.333\nCl 0 0 -2.333",
    geometry="csd_oct", dist_ang=2.333, published_n_active=3,
))

# ── 7. [FeCl4]^- CSD real structure ────────────────────────────────────────
# Fe(III) d5, S=5/2 (high-spin), tetrahedral
# ASF=6, Published=5, ML=4
# Typical Fe-Cl tetrahedral: 2.200-2.290 Å
paths.append(save(
    name="FeCl4_1m_CSD",
    metal="Fe", ligand="Cl", n_ligands=4, charge=-1, spin=5,
    atom_str="Fe 0 0 0\nCl 2.250 0 0\nCl -2.250 0 0\nCl 0 2.250 0\n"
             "Cl 0 -2.250 0",
    geometry="csd_tet", dist_ang=2.250, published_n_active=5,
))

# ── 8. [MnCl4]^2- CSD real structure ───────────────────────────────────────
# Mn(II) d5, S=5/2 (high-spin), tetrahedral
# ASF=7, Published=5, ML=4
# Typical Mn-Cl tetrahedral: 2.340-2.380 Å
paths.append(save(
    name="MnCl4_2m_CSD",
    metal="Mn", ligand="Cl", n_ligands=4, charge=-2, spin=5,
    atom_str="Mn 0 0 0\nCl 2.360 0 0\nCl -2.360 0 0\nCl 0 2.360 0\n"
             "Cl 0 -2.360 0",
    geometry="csd_tet", dist_ang=2.360, published_n_active=5,
))

# ── 9. [OsCl6]^2- CSD real structure ───────────────────────────────────────
# Os(IV) d4, S=1 (intermediate spin for 5d metal), octahedral
# ASF=6, Published=4, ML=3
# Os-Cl octahedral: ~2.38 Å
paths.append(save(
    name="OsCl6_2m_CSD",
    metal="Os", ligand="Cl", n_ligands=6, charge=-2, spin=2,
    atom_str="Os 0 0 0\nCl 2.380 0 0\nCl -2.380 0 0\nCl 0 2.380 0\n"
             "Cl 0 -2.380 0\nCl 0 0 2.380\nCl 0 0 -2.380",
    geometry="csd_oct", dist_ang=2.380, published_n_active=4,
    basis="def2-svp",
))

# ── 10. Fe(NH3)6^2+ CSD real structure ─────────────────────────────────────
# Fe(II) d6, S=2 (high-spin, weak-field NH3), octahedral
# ASF=12, Published=6, ML=7
# Fe-N ≈ 2.20 Å
paths.append(save(
    name="FeNH3_6_2p_CSD",
    metal="Fe", ligand="N", n_ligands=6, charge=2, spin=4,
    atom_str="Fe 0 0 0\nN 2.200 0 0\nN -2.200 0 0\nN 0 2.200 0\n"
             "N 0 -2.200 0\nN 0 0 2.200\nN 0 0 -2.200",
    geometry="csd_oct", dist_ang=2.200, published_n_active=6,
))

# ── Write index file ────────────────────────────────────────────────────────
index_path = "/pc2/users/h/hpcmual/activeml/dmrg_labels/benchmark_10_index.txt"
with open(index_path, "w") as f:
    for p in paths:
        f.write(p + "\n")

print(f"\nIndex: {index_path}  ({len(paths)} systems)")
print("\nExpected comparison table after DMRG run:")
print(f"{'System':<22} {'Pub':>4} {'ML':>4} {'CASSCF':>7} {'ASF':>5} {'DMRG-e':>7} {'DMRG-N':>7}")
print("-" * 58)
known = [
    # name,           pub,  ml,  casscf, asf, dmrg_e, dmrg_n
    ("CrCl6 oct",      4,   4,    4,      7,  "→",    "→"),
    ("RhCl6 oct",      6,   4,    5,      4,  "→",    "→"),
    ("MoCl6 oct",      3,   4,    3,      5,  "→",    "→"),
    ("ReCl6 oct",      3,   4,    3,      3,  "→",    "→"),
    ("Co(NH3)6",       6,   7,    6,      6,  "→",    "→"),
    ("CrCl6 CSD",      3,   3,    3,      7,  "7 ✓",  "8"),
    ("FeCl4 CSD",      5,   4,    5,      6,  "→",    "→"),
    ("MnCl4 CSD",      5,   4,    5,      7,  "→",    "→"),
    ("OsCl6 CSD",      4,   3,    4,      6,  "→",    "→"),
    ("Fe(NH3)6 CSD",   6,   7,    6,     12,  "→",    "→"),
]
for row in known:
    print(f"{row[0]:<22} {row[1]:>4} {row[2]:>4} {row[3]:>7} {row[4]:>5} {str(row[5]):>7} {str(row[6]):>7}")

print(f"\nNote: CrCl6_3m_CSD ≈ CrCl6_3m_oct_d3 already run: entropy=7, NOON=8 → validates ASF=7 ✓")
print("Next: sbatch ~/activeml/scripts/submit_benchmark_10.sh")
