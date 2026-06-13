"""
Generate full multi-spin active space dataset.
50 systems x 3 spin states = 150 CASSCF calculations.
Saves one JSON per system with CASSCF labels.
"""
import numpy as np
import json
import os
import sys
from pyscf import gto, scf, mcscf

# ── SYSTEM DEFINITIONS ────────────────────────────────────────
# (metal, charge, n_ligands, ligand, bond_dist_Ang, spin_states)
SYSTEMS = [
    # ── IRON (Fe) ── most important for qHPC-GREEN
    ("Fe", -2, 4, "Cl", 2.18, [4, 2, 0]),   # FeCl4^2-
    ("Fe", -3, 6, "Cl", 2.35, [5, 3, 1]),   # FeCl6^3-
    ("Fe",  2, 6, "O",  2.12, [4, 2, 0]),   # Fe(H2O)6^2+
    ("Fe",  2, 6, "N",  2.18, [4, 2, 0]),   # Fe(NH3)6^2+
    ("Fe",  3, 6, "N",  1.99, [5, 3, 1]),   # Fe(NH3)6^3+
    ("Fe",  3, 6, "O",  2.00, [5, 3, 1]),   # Fe(H2O)6^3+
    ("Fe", -2, 4, "O",  1.95, [4, 2, 0]),   # FeO4^2-
    ("Fe",  2, 4, "N",  2.05, [4, 2, 0]),   # Fe(NH3)4^2+

    # ── MANGANESE (Mn) ──
    ("Mn", -2, 4, "Cl", 2.35, [5, 3, 1]),   # MnCl4^2-
    ("Mn", -3, 6, "Cl", 2.38, [4, 2, 0]),   # MnCl6^3-
    ("Mn",  2, 6, "O",  2.18, [5, 3, 1]),   # Mn(H2O)6^2+
    ("Mn",  2, 6, "N",  2.18, [5, 3, 1]),   # Mn(NH3)6^2+
    ("Mn",  3, 6, "O",  2.00, [4, 2, 0]),   # Mn(H2O)6^3+
    ("Mn",  3, 6, "N",  2.01, [4, 2, 0]),   # Mn(NH3)6^3+
    ("Mn", -2, 4, "O",  1.93, [5, 3, 1]),   # MnO4^2-

    # ── CHROMIUM (Cr) ──
    ("Cr", -3, 6, "Cl", 2.31, [3, 1]),      # CrCl6^3-
    ("Cr",  3, 6, "O",  1.97, [3, 1]),      # Cr(H2O)6^3+
    ("Cr",  3, 6, "N",  2.07, [3, 1]),      # Cr(NH3)6^3+
    ("Cr",  2, 6, "O",  2.08, [4, 2, 0]),   # Cr(H2O)6^2+
    ("Cr", -2, 4, "Cl", 2.28, [4, 2, 0]),   # CrCl4^2-

    # ── COBALT (Co) ──
    ("Co", -2, 4, "Cl", 2.26, [3, 1]),      # CoCl4^2-
    ("Co",  3, 6, "N",  1.96, [4, 2, 0]),   # Co(NH3)6^3+
    ("Co",  2, 6, "O",  2.09, [3, 1]),      # Co(H2O)6^2+
    ("Co",  2, 6, "N",  2.18, [3, 1]),      # Co(NH3)6^2+
    ("Co",  3, 6, "O",  1.89, [4, 2, 0]),   # Co(H2O)6^3+
    ("Co", -2, 4, "O",  1.88, [3, 1]),      # CoO4^2-

    # ── NICKEL (Ni) ──
    ("Ni", -2, 4, "Cl", 2.21, [2, 0]),      # NiCl4^2-
    ("Ni",  2, 6, "O",  2.07, [2, 0]),      # Ni(H2O)6^2+
    ("Ni",  2, 6, "N",  2.11, [2, 0]),      # Ni(NH3)6^2+
    ("Ni",  2, 4, "Cl", 2.22, [2, 0]),      # NiCl4^2- (sq plan)
    ("Ni",  2, 4, "O",  2.00, [2, 0]),      # Ni(H2O)4^2+

    # ── COPPER (Cu) ──
    ("Cu", -2, 4, "Cl", 2.26, [1]),         # CuCl4^2-
    ("Cu",  2, 6, "O",  2.10, [1]),         # Cu(H2O)6^2+
    ("Cu",  2, 4, "N",  2.03, [1]),         # Cu(NH3)4^2+
    ("Cu",  2, 4, "O",  1.97, [1]),         # Cu(H2O)4^2+
    ("Cu", -3, 6, "Cl", 2.30, [1]),         # CuCl6^3-
]

# ── GEOMETRY BUILDER ─────────────────────────────────────────
def build_geometry(metal, ligand, n_lig, dist):
    positions = {
        4: [(dist,0,0),(-dist,0,0),(0,dist,0),(0,-dist,0)],
        6: [(dist,0,0),(-dist,0,0),(0,dist,0),
            (0,-dist,0),(0,0,dist),(0,0,-dist)]
    }
    atom_str = f"{metal}  0.000  0.000  0.000\n"
    for p in positions[n_lig]:
        atom_str += f"{ligand}  {p[0]:.3f}  {p[1]:.3f}  {p[2]:.3f}\n"
    return atom_str

# ── ACTIVE ELECTRON FINDER ────────────────────────────────────
def get_active_electrons(n_total):
    """Find n_active_elec such that n_core = (total-active)/2 is integer."""
    for n_act in [10, 9, 11, 8, 12, 7, 13]:
        n_core = n_total - n_act
        if n_core >= 0 and n_core % 2 == 0:
            return n_act
    return 10

# ── MAIN ─────────────────────────────────────────────────────
def run_one(metal, charge, n_lig, ligand, dist, spin):
    mult  = spin + 1
    name  = f"{metal}_{ligand}{n_lig}_chg{charge}_s{spin}"
    outdir = os.path.expanduser("~/activeml/data/generated")
    outfile = os.path.join(outdir, f"{name}.json")

    # Skip if already done
    if os.path.exists(outfile):
        with open(outfile) as f:
            r = json.load(f)
        if r.get('converged', False):
            print(f"SKIP (exists): {name}")
            return True

    os.makedirs(outdir, exist_ok=True)

    try:
        # Build molecule
        mol = gto.Mole()
        mol.atom    = build_geometry(metal, ligand, n_lig, dist)
        mol.basis   = 'def2-SVP'
        mol.charge  = charge
        mol.spin    = spin
        mol.verbose = 0
        mol.build()

        n_act = get_active_electrons(mol.nelectron)
        print(f"Running {name}: {mol.nelectron}e, "
              f"active={n_act}e/10o, mult={mult}")

        # HF
        mf = scf.UHF(mol)
        mf.max_cycle = 300
        mf.conv_tol  = 1e-10
        mf.run()
        if not mf.converged:
            print(f"  HF not converged for {name}")

        # CASSCF
        mc = mcscf.CASSCF(mf, 10, n_act)
        mc.max_cycle_macro = 300
        mc.conv_tol        = 1e-8
        mc.ah_level_shift  = 1e-3
        mc.verbose         = 0
        mc.run()

        # Natural orbital occupations
        casdm1 = mc.fcisolver.make_rdm1(
            mc.ci, mc.ncas, mc.nelecas)
        no_occ, _ = np.linalg.eigh(casdm1)
        no_occ    = np.sort(no_occ)[::-1]
        n_active  = sum(1 for n in no_occ if 0.02 < n < 1.98)

        print(f"  CASSCF: {mc.e_tot:.6f}  "
              f"Ecorr: {mc.e_tot-mf.e_tot:.4f}  "
              f"active: {n_active}  "
              f"conv: {mc.converged}")

        # Save
        result = {
            "name"        : name,
            "metal"       : metal,
            "ligand"      : ligand,
            "n_ligands"   : n_lig,
            "charge"      : charge,
            "spin"        : spin,
            "mult"        : mult,
            "dist_ang"    : dist,
            "n_electrons" : mol.nelectron,
            "n_active_e"  : n_act,
            "E_HF"        : float(mf.e_tot),
            "E_CASSCF"    : float(mc.e_tot),
            "corr_energy" : float(mc.e_tot - mf.e_tot),
            "converged"   : bool(mc.converged),
            "n_active"    : n_active,
            "no_occ"      : [float(n) for n in no_occ],
        }
        with open(outfile, "w") as f:
            json.dump(result, f, indent=2)
        return True

    except Exception as e:
        print(f"  ERROR for {name}: {e}")
        return False

# ── RUN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Get job index from command line
    job_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # Flatten all (system, spin) combinations
    all_jobs = []
    for metal, charge, n_lig, ligand, dist, spins in SYSTEMS:
        for spin in spins:
            all_jobs.append(
                (metal, charge, n_lig, ligand, dist, spin))

    total = len(all_jobs)
    print(f"Total jobs: {total}")

    if job_idx < total:
        args = all_jobs[job_idx]
        print(f"Running job {job_idx}/{total}: {args}")
        success = run_one(*args)
        sys.exit(0 if success else 1)
    else:
        print(f"Job index {job_idx} out of range (max {total-1})")
        sys.exit(1)
