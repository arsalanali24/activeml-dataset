"""
Targeted expansion: bring all metals to ~30 systems.
Focus on Co, Ni, Cu which are most underrepresented.
Reuses same convergence protocol as gen_300.py.
"""
import numpy as np
import json
import os
import sys
import logging
from pyscf import gto, scf, mcscf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

def build_geometry(metal, ligand, n_lig, dist):
    positions = {
        4: [(dist,0,0),(-dist,0,0),(0,dist,0),(0,-dist,0)],
        6: [(dist,0,0),(-dist,0,0),(0,dist,0),
            (0,-dist,0),(0,0,dist),(0,0,-dist)]
    }
    atom_str = f"{metal}  0.000  0.000  0.000\n"
    for p in positions[n_lig]:
        atom_str += (f"{ligand}  {p[0]:.3f}  "
                     f"{p[1]:.3f}  {p[2]:.3f}\n")
    return atom_str

def get_active_electrons(n_total):
    for n_act in [10, 9, 11, 8, 12, 7, 13, 6, 14]:
        n_core = n_total - n_act
        if n_core >= 0 and n_core % 2 == 0:
            return n_act
    raise ValueError(f"Cannot find n_active for {n_total}")

def run_hf(mol):
    strategies = [
        dict(max_cycle=300, conv_tol=1e-10,
             damp=0.0,  level_shift=0.0),
        dict(max_cycle=500, conv_tol=1e-9,
             damp=0.3,  level_shift=0.0),
        dict(max_cycle=500, conv_tol=1e-8,
             damp=0.5,  level_shift=0.2),
        dict(max_cycle=800, conv_tol=1e-7,
             damp=0.3,  level_shift=0.5),
    ]
    for i, s in enumerate(strategies):
        mf = scf.UHF(mol)
        for k, v in s.items():
            setattr(mf, k, v)
        mf.verbose = 0
        mf.run()
        if mf.converged:
            log.info(f"  HF converged (strategy {i+1}): "
                     f"{mf.e_tot:.8f}")
            return mf
    log.warning("  HF did not converge — using best attempt")
    return mf

def run_casscf(mf, mol, n_act):
    homo_idx = int(mf.mo_occ[0].sum() +
                   mf.mo_occ[1].sum()) // 2 - 1
    lumo_idx = homo_idx + 1
    center   = (homo_idx + lumo_idx) // 2

    windows = [
        list(range(max(0, center-5), center+5)),
        list(range(max(0, center-4), center+6)),
        list(range(max(0, center-6), center+4)),
        list(range(max(0, center-3), center+7)),
        list(range(max(0, center-7), center+3)),
    ]

    best_mc    = None
    best_ecorr = 0.0

    for w_idx, window in enumerate(windows):
        if len(window) < 10:
            continue
        for shift in [1e-3, 1e-2, 5e-2, 1e-1]:
            try:
                mc = mcscf.CASSCF(mf, 10, n_act)
                mc.max_cycle_macro = 500
                mc.max_cycle_micro = 25
                mc.conv_tol        = 1e-8
                mc.conv_tol_grad   = 1e-5
                mc.ah_level_shift  = shift
                mc.verbose         = 0
                mo = mc.sort_mo(window[:10], base=0)
                mc.kernel(mo)
                ecorr = mc.e_tot - mf.e_tot
                if ecorr < 0:
                    if ecorr < best_ecorr:
                        best_mc    = mc
                        best_ecorr = ecorr
                    if mc.converged and ecorr < -0.01:
                        return mc, True
            except Exception as e:
                continue

    if best_mc is not None:
        return best_mc, best_mc.converged
    return None, False

def run_one(metal, charge, n_lig, ligand, dist, spin):
    mult  = spin + 1
    name  = (f"{metal}_{ligand}{n_lig}"
             f"_chg{charge}_spin{spin}")
    outdir  = os.path.expanduser(
        "~/activeml/data/generated300")
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"{name}.json")

    if os.path.exists(outfile):
        with open(outfile) as f:
            r = json.load(f)
        if r.get('converged') and \
                r.get('corr_energy', 0) < -0.001:
            log.info(f"SKIP (done): {name}")
            return True

    log.info(f"\n{'='*55}")
    log.info(f"Starting: {name}  mult={mult}")

    try:
        mol = gto.Mole()
        mol.atom    = build_geometry(
            metal, ligand, n_lig, dist)
        mol.basis   = 'def2-SVP'
        mol.charge  = charge
        mol.spin    = spin
        mol.verbose = 0
        mol.build()

        n_act = get_active_electrons(mol.nelectron)

        mf = run_hf(mol)
        mc, converged = run_casscf(mf, mol, n_act)

        if mc is None:
            json.dump({'name':name,'status':'failed'},
                      open(outfile,'w'))
            return False

        ecorr = mc.e_tot - mf.e_tot
        if ecorr >= 0:
            json.dump({'name':name,'status':'unphysical',
                       'corr_energy':float(ecorr)},
                      open(outfile,'w'))
            return False

        casdm1 = mc.fcisolver.make_rdm1(
            mc.ci, mc.ncas, mc.nelecas)
        no_occ, _ = np.linalg.eigh(casdm1)
        no_occ    = np.sort(no_occ)[::-1]
        n_active  = sum(1 for n in no_occ
                        if 0.02 < n < 1.98)

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
            "corr_energy" : float(ecorr),
            "hf_converged": bool(mf.converged),
            "converged"   : bool(converged),
            "n_active"    : n_active,
            "no_occ"      : [float(n) for n in no_occ],
            "status"      : "ok",
        }
        with open(outfile, 'w') as f:
            json.dump(result, f, indent=2)
        log.info(f"  Saved: active={n_active} "
                 f"Ecorr={ecorr:.4f} conv={converged}")
        return True

    except Exception as e:
        log.error(f"  Error: {e}")
        json.dump({'name':name,'status':'error',
                   'reason':str(e)},
                  open(outfile,'w'))
        return False

# ── TARGETED EXPANSION ────────────────────────────────────────
# Focus on Co(+19), Ni(+21), Cu(+21) to reach ~30 each
# Also add Fe(+12), Mn(+7), Cr(+10) for balance
SYSTEMS = [
    # ── COBALT — need 19 more (d7) ──
    ("Co", -2, 4, "Cl", 2.26, [3,1]),
    ("Co", -2, 4, "Cl", 2.36, [3,1]),
    ("Co", -2, 4, "Cl", 2.16, [3,1]),
    ("Co", -2, 4, "Cl", 2.46, [3,1]),
    ("Co", -3, 4, "Cl", 2.22, [4,2,0]),
    ("Co", -3, 4, "Cl", 2.32, [4,2,0]),
    ("Co", -3, 4, "Cl", 2.12, [4,2,0]),
    ("Co", -4, 4, "Cl", 2.28, [3,1]),
    ("Co", -1, 4, "Cl", 2.24, [2,0]),
    ("Co",  0, 4, "Cl", 2.20, [3,1]),
    ("Co", -2, 6, "Cl", 2.30, [3,1]),
    ("Co", -2, 6, "Cl", 2.40, [3,1]),
    ("Co", -3, 6, "Cl", 2.28, [4,2,0]),
    ("Co", -4, 6, "Cl", 2.35, [3,1]),
    ("Co", -4, 6, "Cl", 2.25, [3,1]),
    ("Co",  0, 6, "Cl", 2.28, [3,1]),
    ("Co", -1, 6, "Cl", 2.26, [2,0]),
    ("Co", -5, 6, "Cl", 2.38, [4,2,0]),
    ("Co", -1, 4, "Cl", 2.34, [2,0]),

    # ── NICKEL — need 21 more (d8) ──
    ("Ni", -2, 4, "Cl", 2.21, [2,0]),
    ("Ni", -2, 4, "Cl", 2.31, [2,0]),
    ("Ni", -2, 4, "Cl", 2.11, [2,0]),
    ("Ni", -2, 4, "Cl", 2.41, [2,0]),
    ("Ni", -3, 4, "Cl", 2.18, [3,1]),
    ("Ni", -3, 4, "Cl", 2.28, [3,1]),
    ("Ni", -4, 4, "Cl", 2.25, [2,0]),
    ("Ni", -1, 4, "Cl", 2.19, [1]),
    ("Ni",  0, 4, "Cl", 2.17, [2,0]),
    ("Ni",  0, 4, "Cl", 2.27, [2,0]),
    ("Ni", -2, 6, "Cl", 2.25, [2,0]),
    ("Ni", -2, 6, "Cl", 2.35, [2,0]),
    ("Ni", -3, 6, "Cl", 2.28, [3,1]),
    ("Ni", -4, 6, "Cl", 2.30, [2,0]),
    ("Ni", -4, 6, "Cl", 2.20, [2,0]),
    ("Ni", -5, 6, "Cl", 2.32, [3,1]),
    ("Ni",  0, 6, "Cl", 2.26, [2,0]),
    ("Ni", -1, 6, "Cl", 2.22, [1]),
    ("Ni", -1, 4, "Cl", 2.29, [1]),
    ("Ni", -3, 4, "Cl", 2.38, [3,1]),
    ("Ni", -4, 4, "Cl", 2.15, [2,0]),

    # ── COPPER — need 21 more (d9) ──
    ("Cu", -2, 4, "Cl", 2.26, [1]),
    ("Cu", -2, 4, "Cl", 2.36, [1]),
    ("Cu", -2, 4, "Cl", 2.16, [1]),
    ("Cu", -2, 4, "Cl", 2.46, [1]),
    ("Cu", -3, 4, "Cl", 2.22, [1]),
    ("Cu", -3, 4, "Cl", 2.32, [1]),
    ("Cu", -4, 4, "Cl", 2.28, [1]),
    ("Cu", -1, 4, "Cl", 2.24, [1]),
    ("Cu",  0, 4, "Cl", 2.20, [1]),
    ("Cu",  0, 4, "Cl", 2.30, [1]),
    ("Cu", -2, 6, "Cl", 2.28, [1]),
    ("Cu", -2, 6, "Cl", 2.38, [1]),
    ("Cu", -3, 6, "Cl", 2.30, [1]),
    ("Cu", -4, 6, "Cl", 2.32, [1]),
    ("Cu", -4, 6, "Cl", 2.22, [1]),
    ("Cu", -5, 6, "Cl", 2.35, [1]),
    ("Cu",  0, 6, "Cl", 2.26, [1]),
    ("Cu", -1, 6, "Cl", 2.24, [1]),
    ("Cu", -1, 4, "Cl", 2.34, [1]),
    ("Cu", -3, 4, "Cl", 2.42, [1]),
    ("Cu", -5, 6, "Cl", 2.25, [1]),

    # ── IRON — need 12 more (d6) ──
    ("Fe", -2, 4, "Cl", 2.28, [4,2,0]),
    ("Fe", -2, 4, "Cl", 2.08, [4,2,0]),
    ("Fe", -3, 4, "Cl", 2.23, [5,3,1]),
    ("Fe", -4, 4, "Cl", 2.20, [4,2,0]),
    ("Fe", -2, 6, "Cl", 2.25, [4,2,0]),
    ("Fe", -3, 6, "Cl", 2.40, [5,3,1]),
    ("Fe", -4, 6, "Cl", 2.30, [4,2,0]),
    ("Fe",  0, 6, "Cl", 2.22, [4,2,0]),
    ("Fe", -1, 6, "Cl", 2.28, [3,1]),
    ("Fe", -5, 6, "Cl", 2.38, [5,3,1]),
    ("Fe",  0, 4, "Cl", 2.25, [4,2,0]),
    ("Fe", -1, 4, "Cl", 2.30, [3,1]),

    # ── MANGANESE — need 7 more (d5) ──
    ("Mn", -2, 4, "Cl", 2.50, [5,3,1]),
    ("Mn", -3, 4, "Cl", 2.40, [4,2,0]),
    ("Mn", -4, 4, "Cl", 2.30, [5,3,1]),
    ("Mn", -5, 6, "Cl", 2.40, [4,2,0]),
    ("Mn",  0, 6, "Cl", 2.28, [5,3,1]),
    ("Mn", -1, 6, "Cl", 2.30, [4,2]),
    ("Mn", -1, 4, "Cl", 2.32, [4,2]),

    # ── CHROMIUM — need 10 more (d3) ──
    ("Cr", -2, 4, "Cl", 2.38, [4,2,0]),
    ("Cr", -3, 6, "Cl", 2.41, [3,1]),
    ("Cr", -4, 6, "Cl", 2.28, [4,2,0]),
    ("Cr", -5, 6, "Cl", 2.35, [3,1]),
    ("Cr",  0, 6, "Cl", 2.25, [4,2,0]),
    ("Cr", -1, 6, "Cl", 2.30, [3,1]),
    ("Cr", -1, 4, "Cl", 2.26, [3,1]),
    ("Cr", -4, 4, "Cl", 2.22, [4,2,0]),
    ("Cr",  0, 4, "Cl", 2.30, [4,2,0]),
    ("Cr", -5, 4, "Cl", 2.26, [3,1]),
]

ALL_JOBS = []
for metal, charge, n_lig, ligand, dist, spins in SYSTEMS:
    for spin in spins:
        ALL_JOBS.append(
            (metal, charge, n_lig, ligand, dist, spin))

if __name__ == "__main__":
    job_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    total   = len(ALL_JOBS)
    print(f"Total jobs: {total}")
    if job_idx >= total:
        sys.exit(1)
    metal, charge, n_lig, ligand, dist, spin = \
        ALL_JOBS[job_idx]
    success = run_one(metal, charge, n_lig, ligand,
                      dist, spin)
    sys.exit(0 if success else 1)
