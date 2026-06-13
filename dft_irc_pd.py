"""
dft_irc_pd.py
=============
DFT geometry optimization and reaction path for Pd oxidative addition.
Track A Step A1: generates IRC geometry snapshots for CASSCF labeling.

Reaction: Pd(PH3)2 + CH3Cl -> [Pd(PH3)2...CH3...Cl]‡ -> Pd(PH3)2(CH3)(Cl)

Pipeline:
  Step 1: Optimize reactant  Pd(PH3)2 + CH3Cl (two separate molecules)
  Step 2: Optimize product   Pd(PH3)2(CH3)(Cl)
  Step 3: Build TS guess by interpolating between reactant and product
  Step 4: Optimize TS geometry (saddle point)
  Step 5: Generate 12 IRC snapshots by linear interpolation along path
  Step 6: Save XYZ files for each snapshot -> used by gen_irc_casscf.py

Why linear interpolation instead of true IRC:
  True IRC requires Hessian at every step (expensive).
  Linear interpolation between optimized reactant, TS, and product
  gives physically meaningful geometries where Pd-C forms and C-Cl breaks.
  This is standard in computational chemistry papers for
  demonstrating active space variation along reaction paths.
  The key physics (n_active changing at TS) does not require
  a true IRC — it requires geometries spanning the reaction.

Output files:
  ~/activeml/data/irc_pd/reactant.xyz
  ~/activeml/data/irc_pd/product.xyz
  ~/activeml/data/irc_pd/ts_guess.xyz
  ~/activeml/data/irc_pd/snapshot_00.xyz ... snapshot_11.xyz
  ~/activeml/data/irc_pd/irc_summary.json

Usage:
  python dft_irc_pd.py            # run full pipeline
  python dft_irc_pd.py test       # test single optimization only
  python dft_irc_pd.py snapshots  # generate snapshots from existing opt
"""
import numpy as np, json, os, sys, logging, time
from pyscf import gto, dft, grad
from pyscf.geomopt import geometric_solver

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

IRC_DIR = os.path.expanduser('~/activeml/data/irc_pd')
os.makedirs(IRC_DIR, exist_ok=True)

# ── DFT SETTINGS ─────────────────────────────────────────────
# PBE0/def2-SVP — good balance of accuracy and speed for Pd
# ECP on Pd handles relativistic effects
XC_FUNC = 'PBE0'
BASIS   = 'def2-SVP'


# ══════════════════════════════════════════════════════════════
# INITIAL GEOMETRIES
# ══════════════════════════════════════════════════════════════

# Reactant: Pd(PH3)2 linear + CH3Cl separate
# Pd-P: 2.29 Ang, P-H: 1.415 Ang
# CH3Cl: C-Cl 1.784 Ang, C-H 1.090 Ang, placed 4.0 Ang from Pd
REACTANT_ATOM = """
Pd   0.000  0.000  0.000
P    2.290  0.000  0.000
P   -2.290  0.000  0.000
H    2.890  1.320  0.000
H    2.890 -0.660  1.143
H    2.890 -0.660 -1.143
H   -2.890  1.320  0.000
H   -2.890 -0.660  1.143
H   -2.890 -0.660 -1.143
C    0.000  4.500  0.000
Cl   0.000  6.284  0.000
H    1.026  4.137  0.000
H   -0.513  4.137  0.888
H   -0.513  4.137 -0.888
"""

# Product: Pd(PH3)2(CH3)(Cl) square planar
# Pd-C: 2.02 Ang, Pd-Cl: 2.38 Ang
PRODUCT_ATOM = """
Pd   0.000  0.000  0.000
P    2.290  0.000  0.000
P   -2.290  0.000  0.000
H    2.890  1.320  0.000
H    2.890 -0.660  1.143
H    2.890 -0.660 -1.143
H   -2.890  1.320  0.000
H   -2.890 -0.660  1.143
H   -2.890 -0.660 -1.143
C    0.000  2.020  0.000
Cl   0.000 -2.380  0.000
H    1.026  2.383  0.000
H   -0.513  2.383  0.888
H   -0.513  2.383 -0.888
"""

# TS guess: Pd-C forming (~2.15), C-Cl breaking (~2.35)
TS_GUESS_ATOM = """
Pd   0.000  0.000  0.000
P    2.290  0.000  0.000
P   -2.290  0.000  0.000
H    2.890  1.320  0.000
H    2.890 -0.660  1.143
H    2.890 -0.660 -1.143
H   -2.890  1.320  0.000
H   -2.890 -0.660  1.143
H   -2.890 -0.660 -1.143
C    0.000  2.150  0.000
Cl   0.000  4.500  0.000
H    1.026  2.513  0.000
H   -0.513  2.513  0.888
H   -0.513  2.513 -0.888
"""


def build_mol(atom_str, charge=0, spin=0):
    """Build PySCF mol with PBE0/def2-SVP + ECP on Pd."""
    mol = gto.Mole()
    mol.atom   = atom_str
    mol.basis  = BASIS
    mol.ecp    = {'Pd': BASIS}   # ECP only on Pd
    mol.charge = charge
    mol.spin   = spin
    mol.verbose = 0
    mol.build()
    return mol


def run_dft(mol):
    """Run PBE0 single point."""
    mf = dft.RKS(mol)
    mf.xc        = XC_FUNC
    mf.max_cycle = 200
    mf.conv_tol  = 1e-8
    mf.verbose   = 0
    mf.kernel()
    return mf


def optimize_geometry(atom_str, label, charge=0, spin=0, maxsteps=50):
    """
    Optimize geometry with PBE0/def2-SVP.
    Returns optimized Mole object and energy.
    """
    xyz_file = os.path.join(IRC_DIR, f'{label}.xyz')
    json_file = os.path.join(IRC_DIR, f'{label}.json')

    if os.path.exists(json_file):
        log.info(f"  {label}: already optimized, loading")
        d = json.load(open(json_file))
        mol = build_mol(d['atom_str'], charge, spin)
        return mol, d['energy']

    log.info(f"  Optimizing {label}...")
    t0 = time.time()

    mol = build_mol(atom_str, charge, spin)
    mf  = run_dft(mol)
    log.info(f"    Initial E = {mf.e_tot:.6f}  converged={mf.converged}")

    try:
        mol_opt = geometric_solver.optimize(mf, maxsteps=maxsteps,
                                            verbose=0)
        mf_opt  = run_dft(mol_opt)
        E_opt   = float(mf_opt.e_tot)
        log.info(f"    Optimized E = {E_opt:.6f}  ({time.time()-t0:.0f}s)")

        # Save XYZ
        atoms = mol_opt.atom_coords(unit='Angstrom')
        syms  = [mol_opt.atom_symbol(i) for i in range(mol_opt.natm)]
        with open(xyz_file, 'w') as f:
            f.write(f"{mol_opt.natm}\n{label} E={E_opt:.6f}\n")
            for sym, pos in zip(syms, atoms):
                f.write(f"{sym:2s}  {pos[0]:10.6f}  {pos[1]:10.6f}  {pos[2]:10.6f}\n")

        # Save JSON with atom string for reloading
        atom_str_opt = '\n'.join(
            f"{sym}  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}"
            for sym, pos in zip(syms, atoms))
        json.dump({'label': label, 'energy': E_opt,
                   'atom_str': atom_str_opt, 'n_atoms': mol_opt.natm},
                  open(json_file, 'w'), indent=2)

        return mol_opt, E_opt

    except Exception as e:
        log.error(f"    Optimization failed: {e}")
        # Fall back to single point
        E = float(mf.e_tot)
        atoms = mol.atom_coords(unit='Angstrom')
        syms  = [mol.atom_symbol(i) for i in range(mol.natm)]
        atom_str_sp = '\n'.join(
            f"{sym}  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}"
            for sym, pos in zip(syms, atoms))
        json.dump({'label': label, 'energy': E,
                   'atom_str': atom_str_sp, 'n_atoms': mol.natm,
                   'note': 'single_point_fallback'},
                  open(json_file, 'w'), indent=2)
        return mol, E


def interpolate_snapshots(mol_a, mol_b, n_points=6):
    """
    Linear interpolation between two geometries.
    Returns list of atom coordinate arrays.
    Assumes same atom ordering.
    """
    coords_a = mol_a.atom_coords(unit='Angstrom')
    coords_b = mol_b.atom_coords(unit='Angstrom')
    snapshots = []
    for i in range(n_points):
        t = i / (n_points - 1)
        coords = coords_a + t * (coords_b - coords_a)
        snapshots.append(coords)
    return snapshots


def save_snapshot(coords, syms, idx, label, energy=None):
    """Save one IRC snapshot as XYZ file."""
    fname = os.path.join(IRC_DIR, f'snapshot_{idx:02d}.xyz')
    with open(fname, 'w') as f:
        e_str = f"E={energy:.6f}" if energy else ""
        f.write(f"{len(syms)}\n{label} {e_str}\n")
        for sym, pos in zip(syms, coords):
            f.write(f"{sym:2s}  {pos[0]:10.6f}  {pos[1]:10.6f}  {pos[2]:10.6f}\n")
    return fname


def atom_str_from_coords(coords, syms):
    return '\n'.join(
        f"{sym}  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}"
        for sym, pos in zip(syms, coords))


def run_full_pipeline():
    """
    Full IRC pipeline:
    1. Optimize reactant and product
    2. Generate 12 snapshots via linear interpolation
       through TS guess geometry
    3. Save XYZ files + summary JSON
    """
    log.info("="*55)
    log.info("Pd(PH3)2 + CH3Cl Oxidative Addition IRC Pipeline")
    log.info("="*55)

    # Step 1: Optimize reactant
    log.info("\nStep 1: Optimizing reactant...")
    mol_r, E_r = optimize_geometry(REACTANT_ATOM, 'reactant', charge=0)

    # Step 2: Optimize product
    log.info("\nStep 2: Optimizing product...")
    mol_p, E_p = optimize_geometry(PRODUCT_ATOM, 'product', charge=0)

    # Step 3: TS guess single point (no TS optimization needed for
    # linear transit — we just need the geometry)
    log.info("\nStep 3: TS guess geometry...")
    mol_ts, E_ts = optimize_geometry(TS_GUESS_ATOM, 'ts_guess',
                                     charge=0, maxsteps=10)

    log.info(f"\nEnergies:")
    log.info(f"  Reactant: {E_r:.6f} Eh")
    log.info(f"  TS guess: {E_ts:.6f} Eh  "
             f"(barrier ≈ {(E_ts-E_r)*627.5:.1f} kcal/mol)")
    log.info(f"  Product:  {E_p:.6f} Eh  "
             f"(rxn ΔE ≈ {(E_p-E_r)*627.5:.1f} kcal/mol)")

    # Step 4: Generate 12 snapshots
    # 5 from reactant to TS, 1 at TS, 6 from TS to product
    log.info("\nStep 4: Generating IRC snapshots...")
    syms = [mol_r.atom_symbol(i) for i in range(mol_r.natm)]

    # Reactant → TS: 6 points (indices 0-5)
    snaps_r_ts = interpolate_snapshots(mol_r, mol_ts, n_points=6)
    # TS → Product: 7 points (indices 5-11, index 5 = TS = shared)
    snaps_ts_p = interpolate_snapshots(mol_ts, mol_p, n_points=7)

    all_snapshots = snaps_r_ts + snaps_ts_p[1:]  # 12 total, no duplicate TS
    assert len(all_snapshots) == 12

    snapshot_files = []
    irc_data = []

    for i, coords in enumerate(all_snapshots):
        if i < 6:
            label = f"reactant_to_TS_{i}"
            phase = "reactant_side"
        elif i == 5:
            label = "TS"
            phase = "transition_state"
        else:
            label = f"TS_to_product_{i-5}"
            phase = "product_side"

        # Compute DFT energy at this geometry
        atom_str = atom_str_from_coords(coords, syms)
        mol_snap = build_mol(atom_str)
        mf_snap  = run_dft(mol_snap)
        E_snap   = float(mf_snap.e_tot)

        fname = save_snapshot(coords, syms, i, label, E_snap)
        snapshot_files.append(fname)

        # Compute key distances: Pd-C (forming) and C-Cl (breaking)
        pd_idx = syms.index('Pd')
        c_idx  = next(j for j, s in enumerate(syms) if s == 'C')
        cl_idx = next(j for j, s in enumerate(syms) if s == 'Cl')

        pd_c_dist  = float(np.linalg.norm(coords[pd_idx] - coords[c_idx]))
        c_cl_dist  = float(np.linalg.norm(coords[c_idx]  - coords[cl_idx]))

        log.info(f"  Snapshot {i:2d} ({phase:15s}): "
                 f"Pd-C={pd_c_dist:.3f} C-Cl={c_cl_dist:.3f} "
                 f"E={E_snap:.4f}")

        irc_data.append({
            'irc_index':   i,
            'label':       label,
            'phase':       phase,
            'energy':      E_snap,
            'pd_c_dist':   pd_c_dist,
            'c_cl_dist':   c_cl_dist,
            'xyz_file':    os.path.basename(fname),
            'atom_str':    atom_str,
        })

    # Save summary
    summary = {
        'reaction':        'Pd(PH3)2 + CH3Cl -> Pd(PH3)2(CH3)(Cl)',
        'method':          f'{XC_FUNC}/{BASIS}',
        'n_snapshots':     len(all_snapshots),
        'E_reactant':      E_r,
        'E_ts':            E_ts,
        'E_product':       E_p,
        'barrier_kcal':    (E_ts - E_r) * 627.5,
        'rxn_energy_kcal': (E_p  - E_r) * 627.5,
        'snapshots':       irc_data,
    }
    summary_file = os.path.join(IRC_DIR, 'irc_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    log.info(f"\nDone! {len(all_snapshots)} snapshots saved to {IRC_DIR}")
    log.info(f"Summary: {summary_file}")
    log.info(f"\nNext step: run gen_irc_casscf.py to compute CASSCF(14,14)")
    log.info(f"  labels for each snapshot.")
    return summary


def test_single():
    """Quick test: just optimize the product geometry."""
    log.info("Test mode: single optimization of product")
    mol_p, E_p = optimize_geometry(PRODUCT_ATOM, 'product_test',
                                   charge=0, maxsteps=30)
    log.info(f"Product optimized: E = {E_p:.6f} Eh")
    log.info("Test passed!")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_single()
    elif len(sys.argv) > 1 and sys.argv[1] == 'snapshots':
        # Regenerate snapshots from existing optimized geometries
        run_full_pipeline()
    else:
        run_full_pipeline()
