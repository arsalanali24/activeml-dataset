"""
Generate 300 multi-spin CASSCF calculations.
Convergence is the top priority:
  - Multiple fallback strategies if default fails
  - Saves result only if energy is physical (Ecorr < 0)
  - Detailed logging so you know exactly what happened
  - Skip if already done (safe to rerun)
"""
import numpy as np
import json
import os
import sys
import logging
from pyscf import gto, scf, mcscf

# ── LOGGING ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ── GEOMETRY ─────────────────────────────────────────────────
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

# ── ACTIVE ELECTRONS ─────────────────────────────────────────
def get_active_electrons(n_total):
    """n_core = (total - active) / 2 must be integer >= 0."""
    for n_act in [10, 9, 11, 8, 12, 7, 13, 6, 14]:
        n_core = n_total - n_act
        if n_core >= 0 and n_core % 2 == 0:
            return n_act
    raise ValueError(f"Cannot find valid n_active for "
                     f"n_total={n_total}")

# ── HF WITH FALLBACKS ────────────────────────────────────────
def run_hf(mol):
    """Try UHF with progressive fallbacks."""
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
        mf.max_cycle  = s['max_cycle']
        mf.conv_tol   = s['conv_tol']
        mf.damp       = s['damp']
        mf.level_shift = s['level_shift']
        mf.verbose    = 0
        mf.run()
        if mf.converged:
            log.info(f"  HF converged (strategy {i+1}): "
                     f"{mf.e_tot:.8f}")
            return mf
        log.warning(f"  HF strategy {i+1} failed")
    # Return best attempt even if not fully converged
    log.warning("  HF did not converge — using best attempt")
    return mf

# ── CASSCF WITH FALLBACKS ────────────────────────────────────
def run_casscf(mf, mol, n_act):
    """
    Try CASSCF with progressive fallbacks.
    Key: always check Ecorr < 0 (physical result).
    """
    n_total = mol.nelectron
    homo_idx = int(mf.mo_occ[0].sum() +
                   mf.mo_occ[1].sum()) // 2 - 1
    lumo_idx = homo_idx + 1
    center   = (homo_idx + lumo_idx) // 2

    # Different active space windows to try
    windows = [
        list(range(max(0, center-5), center+5)),
        list(range(max(0, center-4), center+6)),
        list(range(max(0, center-6), center+4)),
        list(range(max(0, center-3), center+7)),
        list(range(max(0, center-7), center+3)),
    ]

    best_mc    = None
    best_ecorr = 0.0  # must be negative

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
                    log.info(
                        f"  CASSCF OK: window={w_idx+1} "
                        f"shift={shift} "
                        f"Ecorr={ecorr:.4f} "
                        f"conv={mc.converged}")
                    if ecorr < best_ecorr:
                        best_mc    = mc
                        best_ecorr = ecorr
                    if mc.converged and ecorr < -0.01:
                        return mc, True
                else:
                    log.warning(
                        f"  CASSCF positive Ecorr={ecorr:.4f} "
                        f"window={w_idx+1} shift={shift}")

            except Exception as e:
                log.warning(f"  CASSCF exception: {e}")
                continue

    if best_mc is not None:
        log.warning(
            f"  Returning best CASSCF: "
            f"Ecorr={best_ecorr:.4f} "
            f"conv={best_mc.converged}")
        return best_mc, best_mc.converged

    return None, False

# ── MAIN CALCULATION ─────────────────────────────────────────
def run_one(metal, charge, n_lig, ligand, dist, spin):
    mult  = spin + 1
    name  = (f"{metal}_{ligand}{n_lig}"
             f"_chg{charge}_spin{spin}")
    outdir  = os.path.expanduser(
        "~/activeml/data/generated300")
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"{name}.json")

    # Parity check BEFORE building full mol
    # odd electrons need odd spin, even electrons need even spin
    from pyscf import gto as _gto
    try:
        _m = _gto.Mole()
        _m.atom   = build_geometry(metal, ligand, n_lig, dist)
        _m.basis  = 'sto-3g'
        _m.charge = charge
        _m.spin   = spin
        _m.verbose = 0
        _m.build()
    except RuntimeError as _e:
        if 'not consistent' in str(_e):
            log.warning(f"SKIP parity mismatch: {name}")
            with open(outfile, 'w') as _f:
                json.dump({'name':name,'status':'skipped',
                           'reason':'parity'}, _f)
            return True
    del _m

    # Skip if already done and physical
    if os.path.exists(outfile):
        with open(outfile) as f:
            r = json.load(f)
        if r.get('converged') and r.get('corr_energy',0) < -0.001:
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
        log.info(f"  Electrons={mol.nelectron}  "
                 f"n_active_e={n_act}  mult={mult}")

        # HF
        mf = run_hf(mol)
        log.info(f"  HF: {mf.e_tot:.8f}  "
                 f"conv={mf.converged}")

        # CASSCF
        mc, converged = run_casscf(mf, mol, n_act)

        if mc is None:
            log.error(f"  CASSCF completely failed for {name}")
            json.dump({'name':name, 'status':'failed',
                       'reason':'CASSCF_failed'},
                      open(outfile,'w'))
            return False

        ecorr = mc.e_tot - mf.e_tot
        if ecorr >= 0:
            log.error(f"  Physical check FAILED: "
                      f"Ecorr={ecorr:.4f} >= 0")
            json.dump({'name':name, 'status':'unphysical',
                       'corr_energy':float(ecorr)},
                      open(outfile,'w'))
            return False

        # Natural orbital occupations
        casdm1 = mc.fcisolver.make_rdm1(
            mc.ci, mc.ncas, mc.nelecas)
        no_occ, _ = np.linalg.eigh(casdm1)
        no_occ    = np.sort(no_occ)[::-1]
        n_active  = sum(1 for n in no_occ
                        if 0.02 < n < 1.98)

        log.info(f"  CASSCF: {mc.e_tot:.8f}  "
                 f"Ecorr={ecorr:.4f}  "
                 f"active={n_active}  "
                 f"conv={converged}")

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
            "corr_energy" : float(ecorr),
            "hf_converged": bool(mf.converged),
            "converged"   : bool(converged),
            "n_active"    : n_active,
            "no_occ"      : [float(n) for n in no_occ],
            "status"      : "ok",
        }
        with open(outfile, 'w') as f:
            json.dump(result, f, indent=2)

        log.info(f"  Saved: {outfile}")
        return True

    except Exception as e:
        log.error(f"  Unexpected error: {e}")
        json.dump({'name':name, 'status':'error',
                   'reason':str(e)},
                  open(outfile,'w'))
        return False

# ── SYSTEM LIST ───────────────────────────────────────────────
# 100 molecules × ~3 spin states = ~300 calculations
# Only Cl ligands — proven to converge reliably
# Multiple oxidation states and coordination numbers
SYSTEMS = [
    # ── IRON ── (most important)
    ("Fe", -2, 4, "Cl", 2.18, [4,2,0]),
    ("Fe", -3, 4, "Cl", 2.13, [5,3,1]),
    ("Fe", -2, 4, "Cl", 2.25, [4,2,0]),  # stretched
    ("Fe", -2, 4, "Cl", 2.10, [4,2,0]),  # compressed
    ("Fe", -3, 6, "Cl", 2.35, [5,3,1]),
    ("Fe", -4, 6, "Cl", 2.40, [4,2,0]),
    ("Fe", -2, 6, "Cl", 2.30, [4,2,0]),
    ("Fe", -1, 4, "Cl", 2.20, [3,1]),
    ("Fe",  0, 4, "Cl", 2.15, [4,2,0]),
    ("Fe", -2, 4, "Cl", 2.35, [4,2,0]),  # longer bond

    # ── MANGANESE ──
    ("Mn", -2, 4, "Cl", 2.35, [5,3,1]),
    ("Mn", -3, 4, "Cl", 2.30, [4,2,0]),
    ("Mn", -2, 4, "Cl", 2.45, [5,3,1]),  # stretched
    ("Mn", -2, 4, "Cl", 2.25, [5,3,1]),  # compressed
    ("Mn", -3, 6, "Cl", 2.38, [4,2,0]),
    ("Mn", -4, 6, "Cl", 2.42, [5,3,1]),
    ("Mn", -2, 6, "Cl", 2.35, [5,3,1]),
    ("Mn", -1, 4, "Cl", 2.30, [4,2]),
    ("Mn",  0, 4, "Cl", 2.28, [5,3,1]),
    ("Mn", -2, 4, "Cl", 2.50, [5,3,1]),

    # ── CHROMIUM ──
    ("Cr", -3, 6, "Cl", 2.31, [3,1]),
    ("Cr", -2, 4, "Cl", 2.28, [4,2,0]),
    ("Cr", -3, 6, "Cl", 2.40, [3,1]),    # stretched
    ("Cr", -3, 6, "Cl", 2.22, [3,1]),    # compressed
    ("Cr", -4, 6, "Cl", 2.35, [4,2,0]),
    ("Cr", -2, 6, "Cl", 2.30, [4,2,0]),
    ("Cr", -1, 4, "Cl", 2.25, [3,1]),
    ("Cr",  0, 4, "Cl", 2.20, [4,2,0]),
    ("Cr", -3, 4, "Cl", 2.26, [3,1]),
    ("Cr", -2, 4, "Cl", 2.38, [4,2,0]),

    # ── COBALT ──
    ("Co", -2, 4, "Cl", 2.26, [3,1]),
    ("Co", -3, 4, "Cl", 2.22, [4,2,0]),
    ("Co", -2, 4, "Cl", 2.36, [3,1]),    # stretched
    ("Co", -2, 4, "Cl", 2.16, [3,1]),    # compressed
    ("Co", -3, 6, "Cl", 2.30, [4,2,0]),
    ("Co", -4, 6, "Cl", 2.35, [3,1]),
    ("Co", -2, 6, "Cl", 2.28, [3,1]),
    ("Co", -1, 4, "Cl", 2.24, [2,0]),
    ("Co",  0, 4, "Cl", 2.20, [3,1]),
    ("Co", -2, 4, "Cl", 2.40, [3,1]),

    # ── NICKEL ──
    ("Ni", -2, 4, "Cl", 2.21, [2,0]),
    ("Ni", -2, 4, "Cl", 2.31, [2,0]),    # stretched
    ("Ni", -2, 4, "Cl", 2.11, [2,0]),    # compressed
    ("Ni", -3, 6, "Cl", 2.25, [3,1]),
    ("Ni", -4, 6, "Cl", 2.30, [2,0]),
    ("Ni", -2, 6, "Cl", 2.25, [2,0]),
    ("Ni", -1, 4, "Cl", 2.19, [1]),
    ("Ni",  0, 4, "Cl", 2.17, [2,0]),
    ("Ni", -2, 4, "Cl", 2.41, [2,0]),
    ("Ni", -3, 4, "Cl", 2.18, [2,0]),

    # ── COPPER ──
    ("Cu", -2, 4, "Cl", 2.26, [1]),
    ("Cu", -2, 4, "Cl", 2.36, [1]),      # stretched
    ("Cu", -2, 4, "Cl", 2.16, [1]),      # compressed
    ("Cu", -3, 6, "Cl", 2.30, [1]),
    ("Cu", -2, 6, "Cl", 2.28, [1]),
    ("Cu", -1, 4, "Cl", 2.24, [1]),
    ("Cu",  0, 4, "Cl", 2.20, [1]),
    ("Cu", -2, 4, "Cl", 2.46, [1]),
    ("Cu", -4, 6, "Cl", 2.35, [1]),
    ("Cu", -3, 4, "Cl", 2.22, [1]),
]

# Flatten to individual (system, spin) jobs
ALL_JOBS = []
for metal, charge, n_lig, ligand, dist, spins in SYSTEMS:
    for spin in spins:
        ALL_JOBS.append(
            (metal, charge, n_lig, ligand, dist, spin))

if __name__ == "__main__":
    job_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    total   = len(ALL_JOBS)

    print(f"Total jobs defined: {total}")

    if job_idx >= total:
        print(f"Index {job_idx} out of range")
        sys.exit(1)

    metal, charge, n_lig, ligand, dist, spin = \
        ALL_JOBS[job_idx]

    success = run_one(
        metal, charge, n_lig, ligand, dist, spin)

    sys.exit(0 if success else 1)
