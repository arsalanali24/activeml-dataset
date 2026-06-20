#!/usr/bin/env python3
"""
gen_dmrg_labels_v2.py
=====================
DMRG label generation with two key fixes over v1:

Fix 1: Uses 'atom' field from JSON when available (correct geometry for
        benchmark systems including NH3 with H atoms).

Fix 2: Uses MP2 natural orbitals for orbital window selection (--use_mp2_nos)
        instead of UHF energy window. This matches what ASF does and gives
        entropy values comparable to ASF benchmark results.

Usage:
    python gen_dmrg_labels_v2.py --system system.json --scratch /path
                                  --use_mp2_nos --n_orb_window 20
                                  --bond_dim_max 500
"""
import argparse, json, logging, os, sys, time, traceback
import numpy as np

os.environ.setdefault("MKL_DEBUG_CPU_TYPE", "5")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("dmrg_v2")

ENTROPY_THRESHOLD = 0.138
N_ORB_WINDOW      = 20


# ── Load system ───────────────────────────────────────────────────────────
def load_system(path):
    with open(path) as f:
        d = json.load(f)
    if "label" not in d:
        d["label"] = os.path.splitext(os.path.basename(path))[0]
    if "spin" not in d and "multiplicity" in d:
        d["spin"] = int(d["multiplicity"]) - 1
    for k in ["charge","spin"]:
        if k not in d:
            raise ValueError(f"Missing field: {k}  in {path}")
    return d


# ── UHF ──────────────────────────────────────────────────────────────────
def run_uhf(system):
    """Run UHF. Uses JSON 'atom' field if present, else rebuilds geometry."""
    from pyscf import scf
    import sys, os
    sys.path.insert(0, os.path.expanduser('~/activeml/scripts'))
    from geometry_utils import rebuild_mol_from_params
    mol = rebuild_mol_from_params(system)  # checks 'atom' field first
    mf = scf.UHF(mol)
    mf.max_cycle = 300; mf.conv_tol = 1e-9; mf.level_shift = 0.3
    mf.kernel()
    if not mf.converged:
        log.warning("UHF NOT CONVERGED for %s", system["label"])
    na = int(mf.mo_occ[0].sum()); nb = int(mf.mo_occ[1].sum())
    gap = (mf.mo_energy[0][na] - mf.mo_energy[0][na-1]) * 27.211
    log.info("UHF: Nα=%d Nβ=%d HOMO-LUMO gap=%.3f eV  E=%.6f", na, nb, gap, mf.e_tot)
    return mol, mf


# ── Orbital window ────────────────────────────────────────────────────────
def get_window_energy(mf, n_window):
    """UHF energy window — fallback."""
    mo_occ_a = mf.mo_occ[0]; n_occ = int(mo_occ_a.sum()); nmo = len(mo_occ_a)
    nh = n_window // 2
    occ_idx = list(range(max(0, n_occ-nh), n_occ))
    vir_idx = list(range(n_occ, min(nmo, n_occ+nh)))
    win = sorted(set(occ_idx+vir_idx))[:n_window]
    log.info("Energy window: %d orbitals, indices %d–%d", len(win), win[0], win[-1])
    return win


def get_window_mp2no(mol, mf, n_window):
    """MP2 NO window — preferred (matches ASF)."""
    from geometry_utils import get_mp2_natural_orbitals, select_window_mp2no
    no_coeff, no_occ = get_mp2_natural_orbitals(mol, mf)
    win_idx = select_window_mp2no(no_occ, n_window)
    return no_coeff, no_occ, win_idx


# ── Integrals ─────────────────────────────────────────────────────────────
def get_integrals(mol, mf, win_idx, no_coeff=None, no_occ=None):
    """Get h1e, g2e. Uses MP2 NO basis if no_coeff provided."""
    from geometry_utils import get_integrals_mp2no, get_integrals_uhf
    if no_coeff is not None:
        return get_integrals_mp2no(mol, mf, no_coeff, win_idx, no_occ)
    return get_integrals_uhf(mol, mf, win_idx)


# ── DMRG sweep ────────────────────────────────────────────────────────────
def run_dmrg_sweep(h1e, g2e, ecore, n_orb, n_alpha, n_beta,
                   scratch_dir, tag, bond_dims, n_sweeps, n_threads):
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    os.makedirs(scratch_dir, exist_ok=True)
    n = len(bond_dims)
    noises = [1e-4]*max(1,n-2) + [1e-5, 0.0]; noises = noises[:n]
    thrds  = [1e-8]*n
    drv = DMRGDriver(scratch=scratch_dir, symm_type=SymmetryTypes.SZ,
                     n_threads=n_threads)
    spin = max(0, n_alpha - n_beta)
    drv.initialize_system(n_sites=n_orb, n_elec=n_alpha+n_beta, spin=spin)
    mpo = drv.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)
    ket = drv.get_random_mps(tag=tag, bond_dim=bond_dims[0], nroots=1)
    energy = drv.dmrg(mpo, ket, n_sweeps=n_sweeps,
                      bond_dims=bond_dims, noises=noises, thrds=thrds, iprint=1)
    log.info("%s: E=%.8f Ha", tag, energy)

    # Extract s1
    s1 = None
    if hasattr(drv, "get_orbital_entropies"):
        try: s1 = np.array(drv.get_orbital_entropies(ket)); log.info("s1 via API")
        except: pass
    noon = None
    if hasattr(drv, "get_1pdm"):
        try:
            dm1 = drv.get_1pdm(ket); dm1 = np.array(dm1)
            if dm1.ndim == 3 and dm1.shape[0] == 2:
                noon = np.diag(dm1[0]) + np.diag(dm1[1])
            elif dm1.ndim == 2:
                noon = np.diag(dm1)
            if noon is not None and s1 is None:
                p = noon/2.0
                s1 = np.where((p>1e-12)&(p<1-1e-12),
                              -p*np.log(np.maximum(p,1e-14))
                              -(1-p)*np.log(np.maximum(1-p,1e-14)), 0.0)
        except Exception as e:
            log.warning("1-PDM extraction: %s", e)
    if s1 is None:  s1   = np.zeros(n_orb)
    if noon is None: noon = np.zeros(n_orb)

    s2 = None
    if hasattr(drv,"get_2pdm"):
        try: dm2=drv.get_2pdm(ket); dm2=np.array(dm2)
              # s2[i,j] = joint entropy, I_ij = s1[i]+s1[j]-s2[i,j]
        except: pass

    return {"energy":float(energy),"s1":s1,"s2":s2,"noon":noon}


# ── Main pipeline ─────────────────────────────────────────────────────────
def process_system(json_path, scratch_base, n_orb_window, bond_dim_max,
                   use_mp2_nos, n_threads):
    t0   = time.time()
    sys_ = load_system(json_path)
    label = sys_["label"]
    out_dir  = os.path.join(scratch_base, label)
    out_path = os.path.join(out_dir, "dmrg_features.json")
    if os.path.exists(out_path):
        log.info("Already done: %s", label); return json.load(open(out_path))

    log.info("═"*55); log.info("Processing: %s", label); log.info("═"*55)

    mol, mf = run_uhf(sys_)

    # Orbital window
    no_coeff = None
    if use_mp2_nos:
        try:
            no_coeff, no_occ, win_idx = get_window_mp2no(mol, mf, n_orb_window)
            log.info("Using MP2 NO window")
        except Exception as e:
            log.warning("MP2 failed (%s) — falling back to energy window", e)
            win_idx = get_window_energy(mf, n_orb_window)
    else:
        win_idx = get_window_energy(mf, n_orb_window)

    n_orb = len(win_idx)

    # Integrals
    h1e, g2e, ecore, n_alpha, n_beta = get_integrals(mol, mf, win_idx, no_coeff,
                                                     no_occ if use_mp2_nos else None)

    # Fast DMRG M=200
    fast = run_dmrg_sweep(h1e, g2e, ecore, n_orb, n_alpha, n_beta,
                           os.path.join(out_dir,"fast"), "GS_FAST",
                           [200,200], 5, n_threads)
    s1_fast = fast["s1"]
    n_fast_e = int((s1_fast > ENTROPY_THRESHOLD).sum())
    n_fast_n = int(sum(1 for n in fast["noon"] if 0.02 < float(n) < 1.98))

    # Production DMRG M=500
    prod_dims = [250,500,500,bond_dim_max,bond_dim_max]
    prod = run_dmrg_sweep(h1e, g2e, ecore, n_orb, n_alpha, n_beta,
                           os.path.join(out_dir,"prod"), "GS_PROD",
                           prod_dims, 12, n_threads)

    s1   = prod["s1"];  noon = prod["noon"]
    n_e  = int((s1 > ENTROPY_THRESHOLD).sum())
    n_n  = int(sum(1 for n in noon if 0.02 < float(n) < 1.98))
    cliff_sel = s1[s1 > ENTROPY_THRESHOLD]
    cliff_exc = s1[s1 <= ENTROPY_THRESHOLD]
    cliff = (float(cliff_sel.min()/max(cliff_exc.max(),1e-12))
             if len(cliff_sel) and len(cliff_exc) else 999.0)
    sens_flag = int(n_fast_e != n_e)
    elapsed = time.time() - t0

    log.info("DONE: n_active_entropy=%d n_active_noon=%d cliff=%.3f flag=%d t=%.0fs",
             n_e, n_n, cliff, sens_flag, elapsed)
    top5 = np.argsort(s1)[::-1][:5]
    log.info("Top-5 entropy: %s → %s", list(top5), [f"{s1[i]:.3f}" for i in top5])

    result = {
        "label": label, "system_json": json_path,
        "n_active_entropy":  n_e,
        "n_active_noon":     n_n,
        "n_active":          n_e,   # primary label
        "s1":    s1.tolist(), "noon": noon.tolist(),
        "Iij":   None,
        "e_dmrg":            prod["energy"],
        "entropy_cliff_ratio": cliff,
        "dmrg_sensitivity_flag": sens_flag,
        "n_fast_entropy":    n_fast_e,
        "n_fast_noon":       n_fast_n,
        "win_idx":  win_idx,
        "n_alpha_cas": n_alpha, "n_beta_cas": n_beta,
        "use_mp2_nos": use_mp2_nos,
        "elapsed_s": round(elapsed, 1),
        "orbital_basis": "MP2_NO" if (use_mp2_nos and no_coeff is not None) else "UHF_energy",
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path,"w") as f:
        json.dump(result, f, indent=2)
    log.info("Saved: %s", out_path)
    return result


# ── Entry point ───────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system",       required=True)
    p.add_argument("--scratch",      required=True)
    p.add_argument("--n_orb_window", type=int, default=N_ORB_WINDOW)
    p.add_argument("--bond_dim_max", type=int, default=500)
    p.add_argument("--use_mp2_nos",  action="store_true",
                   help="Use MP2 NOs for orbital window (recommended, matches ASF)")
    p.add_argument("--n_threads",    type=int,
                   default=int(os.environ.get("OMP_NUM_THREADS",4)))
    args = p.parse_args()

    log.info("gen_dmrg_labels_v2.py | use_mp2_nos=%s | system=%s",
             args.use_mp2_nos, args.system)
    try:
        r = process_system(args.system, args.scratch, args.n_orb_window,
                           args.bond_dim_max, args.use_mp2_nos, args.n_threads)
        print(f"\n{'═'*55}")
        print(f"  DONE: {r['label']}")
        print(f"  orbital_basis    = {r['orbital_basis']}")
        print(f"  n_active_entropy = {r['n_active_entropy']}")
        print(f"  n_active_noon    = {r['n_active_noon']}")
        print(f"  cliff            = {r['entropy_cliff_ratio']:.3f}")
        print(f"  elapsed          = {r['elapsed_s']:.0f}s")
        print(f"{'═'*55}\n")
        sys.exit(0)
    except Exception:
        log.error(traceback.format_exc()); sys.exit(1)

if __name__ == "__main__":
    main()
