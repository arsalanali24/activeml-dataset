"""
Fixed version for systems that gave positive Ecorr or did not converge.
Uses sort_mo to explicitly select active orbitals near HOMO/LUMO.
"""
import numpy as np, json, os, sys
from pyscf import gto, scf, mcscf

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

def get_active_electrons(n_total):
    for n_act in [10, 9, 11, 8, 12, 7, 13]:
        n_core = n_total - n_act
        if n_core >= 0 and n_core % 2 == 0:
            return n_act
    return 10

def run_fixed(metal, charge, n_lig, ligand, dist, spin):
    mult  = spin + 1
    name  = f"{metal}_{ligand}{n_lig}_chg{charge}_s{spin}"
    outdir = os.path.expanduser("~/activeml/data/generated")
    outfile = os.path.join(outdir, f"{name}.json")
    os.makedirs(outdir, exist_ok=True)

    mol = gto.Mole()
    mol.atom    = build_geometry(metal, ligand, n_lig, dist)
    mol.basis   = 'def2-SVP'
    mol.charge  = charge
    mol.spin    = spin
    mol.verbose = 0
    mol.build()

    n_act = get_active_electrons(mol.nelectron)
    print(f"{name}: {mol.nelectron}e, "
          f"active={n_act}e, mult={mult}")

    # HF
    mf = scf.UHF(mol)
    mf.max_cycle = 500
    mf.conv_tol  = 1e-10
    mf.damp      = 0.3
    mf.run()
    if not mf.converged:
        mf.damp = 0.5
        mf.run()
    print(f"  HF: {mf.e_tot:.6f}  conv: {mf.converged}")

    # Find HOMO and LUMO indices
    mo_occ_a = mf.mo_occ[0]
    mo_occ_b = mf.mo_occ[1]
    occ_total = mo_occ_a + mo_occ_b
    mo_e_mean = (mf.mo_energy[0] + mf.mo_energy[1]) / 2

    n_occ  = int((occ_total > 0.5).sum())
    n_virt = len(occ_total) - n_occ
    homo_idx = n_occ - 1
    lumo_idx = n_occ

    # Select active orbitals centered on HOMO/LUMO gap
    center = (homo_idx + lumo_idx) // 2
    start  = max(0, center - 5)
    end    = min(len(occ_total), start + 10)
    active_idx = list(range(start, end))

    print(f"  Active orb indices: {start}-{end-1} "
          f"(HOMO={homo_idx}, LUMO={lumo_idx})")

    # CASSCF with explicit orbital selection
    mc = mcscf.CASSCF(mf, 10, n_act)
    mc.max_cycle_macro = 500
    mc.conv_tol        = 1e-8
    mc.ah_level_shift  = 1e-3
    mc.verbose         = 0
    mo = mc.sort_mo(active_idx, base=0)
    mc.kernel(mo)

    ecorr = mc.e_tot - mf.e_tot
    print(f"  CASSCF: {mc.e_tot:.6f}  "
          f"Ecorr: {ecorr:.4f}  conv: {mc.converged}")

    if ecorr > 0:
        print(f"  Still positive Ecorr — trying wider window")
        # Try wider active space window
        start2 = max(0, center - 6)
        end2   = min(len(occ_total), start2 + 10)
        active_idx2 = list(range(start2, end2))
        mc2 = mcscf.CASSCF(mf, 10, n_act)
        mc2.max_cycle_macro = 500
        mc2.conv_tol        = 1e-8
        mc2.ah_level_shift  = 1e-2
        mc2.verbose         = 0
        mo2 = mc2.sort_mo(active_idx2, base=0)
        mc2.kernel(mo2)
        ecorr2 = mc2.e_tot - mf.e_tot
        print(f"  Retry Ecorr: {ecorr2:.4f}")
        if ecorr2 < ecorr:
            mc = mc2

    casdm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
    no_occ, _ = np.linalg.eigh(casdm1)
    no_occ    = np.sort(no_occ)[::-1]
    n_active  = sum(1 for n in no_occ if 0.02 < n < 1.98)

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
    print(f"  Saved: {outfile}")
    return result

# Systems to rerun — identified from failure analysis
RERUN = [
    ("Mn", -2, 4, "Cl", 2.35, 1),
    ("Mn", -2, 4, "Cl", 2.35, 3),
    ("Mn",  2, 6, "O",  2.18, 5),
    ("Cr",  3, 6, "O",  1.97, 3),
    ("Fe",  2, 6, "N",  2.18, 4),
    ("Co",  3, 6, "O",  1.89, 0),
    ("Fe", -2, 4, "Cl", 2.18, 0),
    ("Cu",  2, 6, "O",  2.10, 1),
]

if __name__ == "__main__":
    job_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if job_idx < len(RERUN):
        run_fixed(*RERUN[job_idx])
    else:
        print(f"Index {job_idx} out of range (max {len(RERUN)-1})")
