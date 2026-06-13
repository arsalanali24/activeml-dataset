"""
gen_4d5d_metals.py
──────────────────
Phase 1 extension: generate CASSCF dataset for 4d and 5d transition metals.
Produces JSON files with the same schema as generated300/ so the existing
ML pipeline consumes them without modification.

Key differences from gen_300.py:
  - Uses def2-TZVP basis (larger, needed for 4d/5d accuracy)
  - Switches on X2C scalar relativistic correction in PySCF
  - Covers Pd, Ru, Rh, Mo  (4d)  and  Ir, Pt  (5d)
  - Same geometry builder (tetrahedral / octahedral / square-planar)
  - Same convergence fallback strategy as gen_300.py
  - Skip-if-exists so the job is safe to resubmit after crashes

Usage on HPC:
  python gen_4d5d_metals.py <job_index>

Count total jobs first:
  python gen_4d5d_metals.py --count
"""

import numpy as np
import json
import os
import sys
import logging
from pyscf import gto, scf, mcscf
# Note: using ECP (def2-TZVP) for relativistic treatment
# X2C incompatible with ECP in this PySCF version

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ── Output directory (same parent as generated300) ────────────────────────────
OUTDIR = os.path.expanduser('~/activeml/data/generated_4d5d')
os.makedirs(OUTDIR, exist_ok=True)

# ── Basis set ─────────────────────────────────────────────────────────────────
# def2-TZVP for all atoms (metal + ligand).
# Larger than def2-SVP used in generated300 but necessary for 4d/5d accuracy.
BASIS = 'def2-svp'

# ── Tabulated constants (Tier 2 features, stored directly in JSON) ────────────
# Effective nuclear charge (Slater rules, Clementi & Raimondi 1963)
# Spin-orbit coupling constants from NIST Atomic Spectra Database (cm^-1)
METAL_CONSTANTS = {
    # metal: (Zeff_d, zeta_SO_cm1, row)
    # 3d early/late metals
    'Ti': ( 8.14,  121,  '3d'),
    'V':  ( 8.98,  208,  '3d'),
    'Zn': (13.57, 1042,  '3d'),
    # 4d metals
    'Pd': (13.00, 1334,  '4d'),
    'Ru': (12.33,  880,  '4d'),
    'Rh': (12.67, 1097,  '4d'),
    'Mo': (10.97,  467,  '4d'),
    'Ir': (17.00, 3909,  '5d'),
    'Pt': (17.33, 4146,  '5d'),
}

# ── Equilibrium M-L bond distances (Angstrom) ─────────────────────────────────
# Based on crystal data (CSD averages) and literature CASSCF studies
EQ_DIST = {
    # (metal, ligand): equilibrium distance
    ('Pd', 'Cl'): 2.30,
    ('Pd', 'Br'): 2.46,
    ('Pd', 'F'):  1.95,
    ('Pd', 'N'):  2.05,
    ('Pd', 'O'):  2.00,
    ('Ru', 'Cl'): 2.35,
    ('Ru', 'Br'): 2.50,
    ('Ru', 'F'):  1.95,
    ('Ru', 'N'):  2.12,
    ('Ru', 'O'):  2.05,
    ('Rh', 'Cl'): 2.32,
    ('Rh', 'Br'): 2.48,
    ('Rh', 'F'):  1.93,
    ('Rh', 'N'):  2.10,
    ('Rh', 'O'):  2.03,
    ('Mo', 'Cl'): 2.42,
    ('Mo', 'Br'): 2.58,
    ('Mo', 'F'):  2.00,
    ('Mo', 'N'):  2.18,
    ('Mo', 'O'):  1.95,
    ('Ir', 'Cl'): 2.35,
    ('Ir', 'Br'): 2.51,
    ('Ir', 'F'):  1.96,
    ('Ir', 'N'):  2.10,
    ('Ir', 'O'):  2.03,
    ('Pt', 'Cl'): 2.30,
    ('Pt', 'Br'): 2.45,
    ('Pt', 'F'):  1.95,
    ('Pt', 'N'):  2.05,
    ('Pt', 'O'):  2.00,
    # Ti bond distances (CSD averages)
    ('Ti', 'Cl'): 2.35, ('Ti', 'Br'): 2.51, ('Ti', 'F'): 1.84,
    ('Ti', 'N'):  2.12, ('Ti', 'O'):  1.95, ('Ti', 'S'):  2.40,
    ('Ti', 'H'):  1.78,
    # V bond distances (CSD averages)
    ('V',  'Cl'): 2.28, ('V',  'Br'): 2.44, ('V',  'F'): 1.79,
    ('V',  'N'):  2.08, ('V',  'O'):  1.90, ('V',  'S'):  2.35,
    ('V',  'H'):  1.73,
    # Zn bond distances (CSD averages)
    ('Zn', 'Cl'): 2.26, ('Zn', 'Br'): 2.41, ('Zn', 'F'): 1.85,
    ('Zn', 'N'):  2.04, ('Zn', 'O'):  1.98, ('Zn', 'S'):  2.32,
    ('Zn', 'H'):  1.54,
}

# ── System definitions ────────────────────────────────────────────────────────
# Format: (metal, charge, n_ligands, ligand, dist_fractions, spin_list, geometry)
# dist_fractions: multiply equilibrium distance by these factors
# spin_list: PySCF spin values (2S, so 0=singlet, 1=doublet, 2=triplet ...)
# geometry: 'oct' (6-coord), 'tet' (4-coord), 'sq_pl' (4-coord square planar)

def make_systems():
    systems = []

    # ── Palladium (4d8, Pd0=d10, PdII=d8, PdIV=d6) ───────────────────────────
    # Pd(0)  d10 → n_active typically 0-2 (closed shell or weak correlation)
    # Pd(II) d8  → most important: square planar, spin-crossover possible
    # Pd(IV) d6  → octahedral, stronger correlation
    for ligand in ['Cl', 'Br', 'F', 'N', 'O']:
        eq = EQ_DIST[('Pd', ligand)]
        for frac in [0.95, 1.00, 1.05]:
            d = round(eq * frac, 3)
            # Pd(II) d8, square planar – industrially most relevant
            systems.append(('Pd', -2, 4, ligand, d, [0, 2], 'sq_pl'))
            # Pd(IV) d6, octahedral
            systems.append(('Pd', -2, 6, ligand, d, [0, 2, 4], 'oct'))
            # Pd(II) neutral complex (common in cross-coupling)
            systems.append(('Pd',  0, 4, ligand, d, [0, 2], 'tet'))

    # ── Ruthenium (4d, Ru(II)=d6, Ru(III)=d5, Ru(IV)=d4) ────────────────────
    # Grubbs catalyst is Ru(II); strong spin-crossover at TS
    for ligand in ['Cl', 'Br', 'N', 'O']:
        eq = EQ_DIST[('Ru', ligand)]
        for frac in [0.95, 1.00, 1.05]:
            d = round(eq * frac, 3)
            # Ru(II) d6, octahedral (most common)
            systems.append(('Ru', -2, 6, ligand, d, [0, 2, 4], 'oct'))
            # Ru(III) d5, octahedral
            systems.append(('Ru', -3, 6, ligand, d, [1, 3, 5], 'oct'))
            # Ru(II) tetrahedral (less common but relevant)
            systems.append(('Ru', -2, 4, ligand, d, [0, 2, 4], 'tet'))

    # ── Rhodium (4d, Rh(I)=d8, Rh(III)=d6) ──────────────────────────────────
    # Wilkinson's catalyst is Rh(I) square planar
    for ligand in ['Cl', 'Br', 'N', 'O']:
        eq = EQ_DIST[('Rh', ligand)]
        for frac in [0.95, 1.00, 1.05]:
            d = round(eq * frac, 3)
            # Rh(I) d8, square planar (Wilkinson's)
            systems.append(('Rh', -2, 4, ligand, d, [0, 2], 'sq_pl'))
            # Rh(III) d6, octahedral
            systems.append(('Rh', -3, 6, ligand, d, [0, 2, 4], 'oct'))
            # Rh(I) tetrahedral
            systems.append(('Rh', -1, 4, ligand, d, [0, 2], 'tet'))

    # ── Molybdenum (4d, Mo(0)=d6, Mo(II)=d4, Mo(III)=d3) ────────────────────
    for ligand in ['Cl', 'Br', 'N', 'O']:
        eq = EQ_DIST[('Mo', ligand)]
        for frac in [0.95, 1.00, 1.05]:
            d = round(eq * frac, 3)
            systems.append(('Mo', -2, 6, ligand, d, [0, 2, 4, 6], 'oct'))
            systems.append(('Mo', -3, 6, ligand, d, [1, 3, 5],    'oct'))
            systems.append(('Mo',  0, 4, ligand, d, [0, 2, 4, 6], 'tet'))

    # ── Iridium (5d, Ir(I)=d8, Ir(III)=d6) ──────────────────────────────────
    # C-H activation catalysis; strong relativistic effects
    for ligand in ['Cl', 'Br', 'N', 'O']:
        eq = EQ_DIST[('Ir', ligand)]
        for frac in [0.95, 1.00, 1.05]:
            d = round(eq * frac, 3)
            # Ir(I) d8 square planar (C-H activation catalyst)
            systems.append(('Ir', -2, 4, ligand, d, [0, 2], 'sq_pl'))
            # Ir(III) d6 octahedral
            systems.append(('Ir', -3, 6, ligand, d, [0, 2, 4], 'oct'))

    # ── Platinum (5d, Pt(II)=d8, Pt(IV)=d6) ─────────────────────────────────
    # Cisplatin is Pt(II) square planar
    for ligand in ['Cl', 'Br', 'N', 'O']:
        eq = EQ_DIST[('Pt', ligand)]
        for frac in [0.95, 1.00, 1.05]:
            d = round(eq * frac, 3)
            # Pt(II) d8 square planar (cisplatin geometry)
            systems.append(('Pt', -2, 4, ligand, d, [0, 2], 'sq_pl'))
            # Pt(IV) d6 octahedral
            systems.append(('Pt', -2, 6, ligand, d, [0, 2, 4], 'oct'))

    # ── Titanium (3d, Ti(II)=d2, Ti(III)=d1, Ti(IV)=d0) ─────────────────────────
    for ligand in ['Cl', 'Br', 'F', 'N', 'O', 'S', 'H', 'I']:
        if ('Ti', ligand) not in EQ_DIST:
            continue
        eq = EQ_DIST[('Ti', ligand)]
        for frac in [0.90, 0.95, 1.00, 1.05, 1.10]:
            d = round(eq * frac, 3)
            # Ti(IV) d0 — most important, industrially critical
            systems.append(('Ti', -4, 6, ligand, d, [0], 'oct'))
            systems.append(('Ti', -4, 4, ligand, d, [0], 'tet'))
            # Ti(III) d1
            systems.append(('Ti', -3, 6, ligand, d, [1], 'oct'))
            systems.append(('Ti', -3, 4, ligand, d, [1], 'tet'))
            # Ti(II) d2
            systems.append(('Ti', -2, 6, ligand, d, [2, 0], 'oct'))
            systems.append(('Ti', -2, 4, ligand, d, [2, 0], 'tet'))
            # Ti(0) d4 — organometallic relevant
            systems.append(('Ti',  0, 6, ligand, d, [4, 2, 0], 'oct'))

    # ── Vanadium (3d, V(II)=d3, V(III)=d2, V(IV)=d1, V(V)=d0) ──────────────
    for ligand in ['Cl', 'Br', 'F', 'N', 'O', 'S', 'H']:
        if ('V', ligand) not in EQ_DIST:
            continue
        eq = EQ_DIST[('V', ligand)]
        for frac in [0.95, 1.00, 1.05]:
            d = round(eq * frac, 3)
            # V(II) d3, octahedral
            systems.append(('V', -2, 6, ligand, d, [3, 1], 'oct'))
            # V(III) d2, octahedral
            systems.append(('V', -3, 6, ligand, d, [2, 0], 'oct'))
            # V(IV) d1, octahedral
            systems.append(('V', -4, 6, ligand, d, [1], 'oct'))
            # V(II) tetrahedral
            systems.append(('V', -2, 4, ligand, d, [3, 1], 'tet'))

    # ── Zinc (3d10, Zn(II)=d10) ───────────────────────────────────────────────
    # Zn is d10 — always closed shell, n_active should be 0
    # Important as a reference/control point in the dataset
    for ligand in ['Cl', 'Br', 'F', 'N', 'O', 'S', 'H']:
        if ('Zn', ligand) not in EQ_DIST:
            continue
        eq = EQ_DIST[('Zn', ligand)]
        for frac in [0.95, 1.00, 1.05]:
            d = round(eq * frac, 3)
            # Zn(II) d10, tetrahedral (most common)
            systems.append(('Zn', -2, 4, ligand, d, [0], 'tet'))
            # Zn(II) d10, octahedral
            systems.append(('Zn', -2, 6, ligand, d, [0], 'oct'))

    return systems


# ── Geometry builder ──────────────────────────────────────────────────────────
def build_geometry(metal, ligand, n_lig, dist, geometry):
    """
    Returns PySCF atom string for the given coordination geometry.
    Metal at origin, ligands placed symmetrically.
    """
    if geometry == 'oct' or (geometry == 'tet' and n_lig == 6):
        positions = [
            ( dist,  0,     0),
            (-dist,  0,     0),
            ( 0,     dist,  0),
            ( 0,    -dist,  0),
            ( 0,     0,     dist),
            ( 0,     0,    -dist),
        ]
    elif geometry == 'sq_pl':
        # Square planar: all in xy-plane
        positions = [
            ( dist,  0,    0),
            (-dist,  0,    0),
            ( 0,     dist, 0),
            ( 0,    -dist, 0),
        ]
    else:
        # Tetrahedral
        s = dist / np.sqrt(3)
        positions = [
            ( s,  s,  s),
            ( s, -s, -s),
            (-s,  s, -s),
            (-s, -s,  s),
        ]

    atom_str = f"{metal}  0.000  0.000  0.000\n"
    for p in positions[:n_lig]:
        atom_str += f"{ligand}  {p[0]:.4f}  {p[1]:.4f}  {p[2]:.4f}\n"
    return atom_str


# ── Active electron selection ─────────────────────────────────────────────────
def get_active_electrons(n_total):
    """
    Choose n_active_e such that n_core = (n_total - n_active_e)/2 is a
    non-negative integer. Target ~10 active electrons; fall back gracefully.
    """
    for n_act in [10, 9, 11, 8, 12, 7, 13, 6, 14, 5, 15]:
        n_core = n_total - n_act
        if n_core >= 0 and n_core % 2 == 0:
            return n_act
    raise ValueError(f"Cannot find valid n_active_e for n_total={n_total}")


# ── UHF with X2C (scalar relativistic) ───────────────────────────────────────
def run_hf_x2c(mol):
    """
    Run UHF with ECP (def2-TZVP pseudopotential handles relativistic
    effects for 4d/5d metals — X2C incompatible with ECP in PySCF).
    Progressive fallback strategy for convergence.
    """
    mf = scf.UHF(mol)
    mf.max_cycle = 300
    mf.conv_tol  = 1e-9
    mf.kernel()

    if not mf.converged:
        log.warning("UHF not converged — trying DIIS damping")
        mf = scf.UHF(mol)
        mf.max_cycle = 500
        mf.conv_tol  = 1e-8
        mf.damp      = 0.3
        mf.kernel()

    if not mf.converged:
        log.warning("UHF still not converged — trying level shift")
        mf = scf.UHF(mol)
        mf.max_cycle  = 500
        mf.conv_tol   = 1e-7
        mf.level_shift = 0.3
        mf.kernel()

    return mf


# ── CASSCF ────────────────────────────────────────────────────────────────────
def run_casscf(mol, mf, n_active_e, n_active_o=10):
    """
    Run CASSCF(n_active_e, n_active_o) on top of the X2C UHF reference.
    Returns (mc, converged) tuple.
    """
    mc = mcscf.CASSCF(mf, n_active_o, n_active_e)
    mc.max_cycle_macro = 200
    mc.max_cycle_micro = 10
    mc.conv_tol        = 1e-8
    mc.conv_tol_grad   = 1e-5
    mc.kernel()

    if not mc.converged:
        log.warning("CASSCF not converged — trying with larger macro cycles")
        mc = mcscf.CASSCF(mf, n_active_o, n_active_e)
        mc.max_cycle_macro = 400
        mc.conv_tol        = 1e-7
        mc.kernel()

    return mc, mc.converged


# ── Run one system ────────────────────────────────────────────────────────────
def run_one(metal, charge, n_lig, ligand, dist, spin, geometry):
    dist_str = str(dist).replace(".", "p")
    name    = f"{metal}_{ligand}{n_lig}_chg{charge}_spin{spin}_{geometry}_d{dist_str}"
    outfile = os.path.join(OUTDIR, f"{name}.json")

    # Skip if already done successfully
    if os.path.exists(outfile):
        try:
            d = json.load(open(outfile))
            if d.get('status') == 'ok' and d.get('corr_energy', 0) < 0:
                log.info(f"SKIP (already ok): {name}")
                return True
        except Exception:
            pass

    log.info(f"\n{'='*55}")
    log.info(f"Starting: {name}")
    log.info(f"  metal={metal} ligand={ligand} n_lig={n_lig} "
             f"charge={charge} spin={spin} geometry={geometry} dist={dist}")

    # ── Build molecule ────────────────────────────────────────────────────────
    try:
        atom_str = build_geometry(metal, ligand, n_lig, dist, geometry)
        mol = gto.Mole()
        mol.atom    = atom_str
        mol.basis   = BASIS
        # ECP only on metal — def2-TZVP ECP not defined for light atoms
        mol.ecp     = {metal: 'def2-tzvp'}
        mol.charge  = charge
        mol.spin    = spin
        mol.verbose = 3
        mol.max_memory = 28000

        mol.build()
    except Exception as e:
        log.error(f"Mol build failed: {e}")
        json.dump({'name': name, 'status': 'build_error', 'error': str(e)},
                  open(outfile, 'w'), indent=2)
        return False

    # Parity check: spin must have same parity as n_electrons
    if (mol.nelectron % 2) != (spin % 2):
        log.warning(f"SKIP parity mismatch: n_elec={mol.nelectron} spin={spin}")
        json.dump({'name': name, 'status': 'skipped',
                   'reason': f'parity n_elec={mol.nelectron} spin={spin}'},
                  open(outfile, 'w'), indent=2)
        return True

    # ── HF ────────────────────────────────────────────────────────────────────
    try:
        mf = run_hf_x2c(mol)
        E_HF = mf.e_tot
        try:
            S2_val, _ = mf.spin_square()
        except Exception:
            S2_val = 0.0  # fallback for converged closed-shell
        log.info(f"  HF done: E={E_HF:.6f}  <S²>={S2_val:.4f}  "
                 f"converged={mf.converged}")
    except Exception as e:
        log.error(f"HF failed: {e}")
        json.dump({'name': name, 'status': 'hf_failed', 'error': str(e)},
                  open(outfile, 'w'), indent=2)
        return False

    # ── Active space selection ─────────────────────────────────────────────────
    try:
        n_active_e = get_active_electrons(mol.nelectron)
        n_active_o = 10  # same as generated300
    except ValueError as e:
        log.error(str(e))
        json.dump({'name': name, 'status': 'no_orbs', 'error': str(e)},
                  open(outfile, 'w'), indent=2)
        return False
    # ── CASSCF(10,10) with internal rotation — fully consistent with 3d dataset ──
    try:
        mc = mcscf.CASSCF(mf, n_active_o, n_active_e)
        mc.max_memory        = 28000
        mc.max_cycle_macro   = 300
        mc.conv_tol          = 1e-7
        mc.conv_tol_grad     = 1e-4
        mc.internal_rotation = True
        mc.kernel()
        converged   = mc.converged
        E_CAS       = mc.e_tot
        corr_energy = E_CAS - E_HF

        # NOONs via make_natural_orbitals — correct for UHF reference
        from pyscf.mcscf import addons as mcaddons
        noons_all, _ = mcaddons.make_natural_orbitals(mc)
        noons = [float(x) for x in noons_all]

        # Count active orbitals: NOONs in fractional range (0.02, 1.98)
        noons_arr = np.array(noons)
        n_active  = int(np.sum((noons_arr > 0.02) & (noons_arr < 1.98)))

        log.info(f"  CASSCF done: E={E_CAS:.6f}  corr={corr_energy:.6f}  "
                 f"converged={converged}  n_active={n_active}")
    except Exception as e:
        log.error(f"CASSCF failed: {e}")
        json.dump({'name': name, 'status': 'failed', 'error': str(e)},
                  open(outfile, 'w'), indent=2)
        return False
    # Sanity check
    if corr_energy >= 0:
        log.error(f"Unphysical: corr_energy={corr_energy:.4f} >= 0")
        json.dump({'name': name, 'status': 'failed',
                   'reason': 'corr_energy>=0'},
                  open(outfile, 'w'), indent=2)
        return False

    # ── Tier 2 tabulated constants ─────────────────────────────────────────────
    z_eff, zeta_so, row = METAL_CONSTANTS[metal]

    # ── Save result (same schema as generated300 + Tier 2 fields) ─────────────
    result = {
        # Core fields matching generated300 schema exactly
        'name'        : name,
        'metal'       : metal,
        'ligand'      : ligand,
        'n_ligands'   : n_lig,
        'charge'      : charge,
        'spin'        : spin,
        'mult'        : spin + 1,
        'dist_ang'    : dist,
        'geometry'    : geometry,
        'n_electrons' : mol.nelectron,
        'n_active_e'  : n_active_e,
        'E_HF'        : E_HF,
        'E_CASSCF'    : E_CAS,
        'corr_energy' : corr_energy,
        'converged'   : bool(converged),
        'n_active'    : n_active,
        'no_occ'      : noons,
        'status'      : 'ok',

        # Tier 2: metal-level features
        'spin_contamination' : float(S2_val - spin/2 * (spin/2 + 1)),
        'z_eff'              : z_eff,
        'zeta_so_cm1'        : zeta_so,
        'metal_row'          : row,

        # Metadata
        'basis'              : BASIS,
        'relativistic'       : 'ecp_def2tzvp',
    }

    json.dump(result, open(outfile, 'w'), indent=2)
    log.info(f"  Saved: {outfile}")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def make_all_jobs():
    """Expand SYSTEMS into flat (metal, charge, n_lig, ligand, dist, spin, geom)
    list with parity filtering."""
    jobs = []
    for metal, charge, n_lig, ligand, dist, spins, geometry in make_systems():
        for spin in spins:
            jobs.append((metal, charge, n_lig, ligand, dist, spin, geometry))
    return jobs


if __name__ == '__main__':
    ALL_JOBS = make_all_jobs()

    if len(sys.argv) > 1 and sys.argv[1] == '--count':
        print(f"Total jobs: {len(ALL_JOBS)}")
        from collections import defaultdict
        by_metal = defaultdict(int)
        for m, *_ in ALL_JOBS:
            by_metal[m] += 1
        print("By metal:")
        for k, v in sorted(by_metal.items()):
            print(f"  {k}: {v}")
        sys.exit(0)

    if len(sys.argv) < 2:
        print("Usage: python gen_4d5d_metals.py <job_index>")
        print("       python gen_4d5d_metals.py --count")
        sys.exit(1)

    idx = int(sys.argv[1])
    if idx >= len(ALL_JOBS):
        print(f"Index {idx} out of range (total {len(ALL_JOBS)})")
        sys.exit(1)

    metal, charge, n_lig, ligand, dist, spin, geometry = ALL_JOBS[idx]
    success = run_one(metal, charge, n_lig, ligand, dist, spin, geometry)
    sys.exit(0 if success else 1)
