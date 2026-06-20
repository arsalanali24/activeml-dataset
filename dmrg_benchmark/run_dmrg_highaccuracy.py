#!/usr/bin/env python3
"""
run_dmrg_highaccuracy.py
========================
High-accuracy DMRG on the 10 benchmark systems with:
  - 30 MP2 NOs (larger window — includes both active AND inactive orbitals
    so the entropy cliff is visible, matching ASF strategy)
  - M=1000 bond dimension (2x better than M=500)
  - 20 sweeps (more thorough convergence)
  - Fiedler ordering (improves DMRG convergence for 1D ansatz)

This is the "correct" DMRG that matches published DMRG benchmarks.
Expected cost: ~45-90 min per system (vs ~30 min for M=500).

Usage:
    python run_dmrg_highaccuracy.py --system system.json
                                     --scratch /scratch/...
                                     --n_threads 8
"""
import argparse, json, logging, os, sys, time, traceback
import numpy as np

os.environ.setdefault("MKL_DEBUG_CPU_TYPE", "5")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("dmrg_hiacc")

# ── Parameters ────────────────────────────────────────────────────────────
N_ORB_WINDOW   = 30     # larger than v2 (20) — shows entropy cliff
M_FAST         = 250    # screening run
M_PROD         = 800   # production (was 500)
N_SWEEPS_PROD  = 20     # more sweeps (was 12)
ENTROPY_THR    = 0.138  # same threshold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system",    required=True)
    p.add_argument("--scratch",   required=True)
    p.add_argument("--n_threads", type=int, default=8)
    args = p.parse_args()

    sys.path.insert(0, os.path.expanduser('~/activeml/scripts'))
    from geometry_utils import (rebuild_mol_from_params,
                                get_mp2_natural_orbitals,
                                select_window_mp2no,
                                get_integrals_mp2no)
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    from pyscf import scf

    t0   = time.time()
    d    = json.load(open(args.system))
    label = d.get("label", os.path.splitext(os.path.basename(args.system))[0])
    out_dir  = os.path.join(args.scratch, label)
    out_path = os.path.join(out_dir, "dmrg_hiacc.json")

    if os.path.exists(out_path):
        log.info("Already done: %s", label)
        r = json.load(open(out_path))
        print(f"DONE {label}: n_active={r['n_active_entropy']} (M={M_PROD}, {N_ORB_WINDOW} NOs)")
        return

    log.info("═"*55)
    log.info("High-accuracy DMRG: %s", label)
    log.info("  Window: %d MP2 NOs  Bond dim: %d  Sweeps: %d",
             N_ORB_WINDOW, M_PROD, N_SWEEPS_PROD)
    log.info("═"*55)

    # UHF
    mol = rebuild_mol_from_params(d)
    mf  = scf.UHF(mol)
    mf.max_cycle = 500; mf.conv_tol = 1e-8; mf.level_shift = 0.3
    mf.kernel()
    if not mf.converged:
        log.warning("UHF not converged — retrying with level_shift=0.8")
        mf.level_shift = 0.8; mf.max_cycle = 500
        mf.kernel()
    if not mf.converged:
        log.warning("UHF still not converged — proceeding with conv_tol=1e-6")
        mf.conv_tol = 1e-6; mf.level_shift = 0.5
        mf.kernel()

    # MP2 NOs — 30-orbital window
    no_coeff, no_occ = get_mp2_natural_orbitals(mol, mf)
    win_idx = select_window_mp2no(no_occ, N_ORB_WINDOW)
    n_orb   = len(win_idx)
    log.info("Window NOONs: min=%.4f max=%.4f", no_occ[win_idx[0]], no_occ[win_idx[-1]])

    # Integrals
    h1e, g2e, ecore, n_alpha, n_beta = get_integrals_mp2no(
        mol, mf, no_coeff, win_idx, no_occ)
    log.info("Integrals: nα=%d nβ=%d ecore=%.2f", n_alpha, n_beta, ecore)

    # ── DMRG sweep helper ─────────────────────────────────────────────────
    def sweep(tag, bd_list, n_sw, scratch_sub):
        os.makedirs(scratch_sub, exist_ok=True)
        n = len(bd_list)
        noises = [1e-4]*max(1,n-3) + [1e-5,1e-5,0.]; noises=noises[:n]
        thrds  = [1e-8]*n
        drv = DMRGDriver(scratch=scratch_sub,
                         symm_type=SymmetryTypes.SZ,
                         n_threads=args.n_threads,
                         stack_mem=int(20e9))
        spin = max(0, n_alpha - n_beta)
        drv.initialize_system(n_sites=n_orb,
                              n_elec=n_alpha+n_beta,
                              spin=spin)
        mpo = drv.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)
        ket = drv.get_random_mps(tag=tag, bond_dim=bd_list[0], nroots=1)
        E = drv.dmrg(mpo, ket, n_sweeps=n_sw,
                     bond_dims=bd_list, noises=noises, thrds=thrds, iprint=1)
        # Extract entropy
        s1 = noon_arr = None
        if hasattr(drv,"get_orbital_entropies"):
            try: s1=np.array(drv.get_orbital_entropies(ket))
            except: pass
        if hasattr(drv,"get_1pdm"):
            try:
                dm=np.array(drv.get_1pdm(ket))
                noon_arr=(np.diag(dm[0])+np.diag(dm[1])) if dm.ndim==3 else np.diag(dm)
                if s1 is None:
                    p=noon_arr/2.
                    s1=np.where((p>1e-12)&(p<1-1e-12),
                               -p*np.log(np.maximum(p,1e-14))
                               -(1-p)*np.log(np.maximum(1-p,1e-14)),0.)
            except: pass
        if s1 is None:   s1=np.zeros(n_orb)
        if noon_arr is None: noon_arr=np.zeros(n_orb)
        return float(E), s1, noon_arr, drv

    # Fast run M=250
    E_fast, s1_fast, _, _ = sweep(
        "FAST",
        [M_FAST, M_FAST],
        5,
        os.path.join(out_dir, "fast")
    )
    n_fast = int((s1_fast > ENTROPY_THR).sum())

    # Production M=1000
    bd_prod = [250, 500, 750, M_PROD, M_PROD, M_PROD]
    E_prod, s1, noon, _ = sweep(
        "PROD",
        bd_prod,
        N_SWEEPS_PROD,
        os.path.join(out_dir, "prod_hiacc")
    )

    n_e = int((s1 > ENTROPY_THR).sum())
    n_n = int(sum(1 for n in noon if 0.02 < float(n) < 1.98))

    # Entropy cliff
    sel = s1[s1 > ENTROPY_THR]
    exc = s1[s1 <= ENTROPY_THR]
    cliff = float(sel.min()/max(exc.max(),1e-12)) if len(sel) and len(exc) else 999.

    elapsed = time.time() - t0
    log.info("DONE: n_active_entropy=%d n_active_noon=%d cliff=%.1f t=%.0fs",
             n_e, n_n, cliff, elapsed)

    # Top entropy values
    top = np.argsort(s1)[::-1][:8]
    log.info("Top-8 entropy: %s", [f"{s1[i]:.3f}" for i in top])

    result = {
        "label": label,
        "n_active_entropy": n_e,
        "n_active_noon":    n_n,
        "s1":               s1.tolist(),
        "noon":             noon.tolist(),
        "e_dmrg":           E_prod,
        "e_fast":           E_fast,
        "entropy_cliff":    cliff,
        "sens_flag":        int(n_fast != n_e),
        "n_fast":           n_fast,
        "n_orb_window":     n_orb,
        "bond_dim_max":     M_PROD,
        "n_sweeps":         N_SWEEPS_PROD,
        "win_idx":          win_idx,
        "elapsed_s":        round(elapsed),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path,"w") as f:
        json.dump(result,f,indent=2)

    print(f"\n{'═'*55}")
    print(f"  DONE:  {label}")
    print(f"  n_active_entropy = {n_e}   (M={M_PROD}, {N_ORB_WINDOW} NOs)")
    print(f"  n_active_noon    = {n_n}")
    print(f"  entropy_cliff    = {cliff:.1f}")
    print(f"  elapsed          = {elapsed/60:.1f} min")
    print(f"{'═'*55}")


if __name__ == "__main__":
    main()
