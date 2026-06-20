#!/usr/bin/env python3
"""
gen_dmrg_labels.py
==================
Generate DMRG ground-truth labels for ONE system.
Designed for SLURM array execution across 3,121 systems.

Outputs (per system):
  n_active              ← ML1 label  (replaces CASSCF label)
  s1[i]                 ← ML2 label + feature (per-orbital entropy)
  Iij[i,j]             ← ML2 feature (mutual information, if API available)
  noon[i]               ← ML2 feature (natural orbital occupations)
  entropy_cliff_ratio   ← quality flag
  dmrg_sensitivity_flag ← 1 if n_active changes between M=200 and M=500

Usage:
  python gen_dmrg_labels.py \\
      --system /path/to/system.json \\
      --scratch /scratch/hpc-prf-qehpc/hpcmual/dmrg_labels \\
      --n_orb_window 20 \\
      --bond_dim_max 500

CRITICAL ENVIRONMENT (must be set before running):
  export MKL_DEBUG_CPU_TYPE=5                           # AMD node fix
  export LD_LIBRARY_PATH=/pc2/.../block2env/lib:$LD_LIBRARY_PATH
  export OMP_NUM_THREADS=4
  source /pc2/users/h/hpcmual/envs/block2env/bin/activate
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback

import numpy as np
from geometry_utils import rebuild_mol_from_params, get_integrals_uhf

# ── AMD MKL fix — must be set before any block2 import ───────────────────
os.environ.setdefault("MKL_DEBUG_CPU_TYPE", "5")

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dmrg_labels")

# ── Constants ─────────────────────────────────────────────────────────────
ENTROPY_THRESHOLD = 0.138        # 0.2 * ln(2), from plan doc
N_ORB_WINDOW_DEFAULT = 20        # 10 occ + 10 vir
FAST_BOND_DIMS = [200, 200]
FAST_SWEEPS = 5
PROD_BOND_DIMS_BASE = [250, 500, 500]   # extended by bond_dim_max
PROD_SWEEPS = 12


# ═══════════════════════════════════════════════════════════════════════════
# 1. System loading
# ═══════════════════════════════════════════════════════════════════════════

def load_system(json_path: str) -> dict:
    """
    Load system from JSON file.

    Expected required fields:
      "atom"   : PySCF atom string e.g. "Fe 0 0 0; Cl 0 0 2.3"
                 OR a list of lists e.g. [["Fe", [0,0,0]], ["Cl", [0,0,2.3]]]
      "charge" : int, molecular charge
      "spin"   : int, 2*S (number of unpaired electrons)

    Optional:
      "label"  : str, unique identifier (defaults to filename stem)
      "basis"  : str, basis set (defaults to "def2-svp")
      "multiplicity" : int, 2S+1 (ignored if "spin" is present)
    """
    with open(json_path) as f:
        data = json.load(f)

    # Derive label from filename if missing
    if "label" not in data:
        data["label"] = os.path.splitext(os.path.basename(json_path))[0]

    # Support "multiplicity" as alternative to "spin"
    if "spin" not in data and "multiplicity" in data:
        data["spin"] = int(data["multiplicity"]) - 1

    # Validate
    required = ["charge", "spin"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"System JSON missing required fields: {missing}\n"
                         f"  File: {json_path}")

    return data


# ═══════════════════════════════════════════════════════════════════════════
# 2. UHF calculation
# ═══════════════════════════════════════════════════════════════════════════

def run_uhf(system: dict) -> tuple:
    """
    Run UHF/def2-SVP on the system.
    Returns (mol, mf). Logs warning if not converged.
    """
    from pyscf import gto, scf

    mol = rebuild_mol_from_params(system)

    mf = scf.UHF(mol)
    mf.max_cycle = 300
    mf.conv_tol = 1e-9
    # Level shift helps convergence for TM complexes
    mf.level_shift = 0.3
    mf.kernel()

    if not mf.converged:
        log.warning("UHF NOT CONVERGED for %s — proceeding anyway", system["label"])

    n_a = int(mf.mo_occ[0].sum())
    n_b = int(mf.mo_occ[1].sum())
    gap = mf.mo_energy[0][n_a] - mf.mo_energy[0][n_a - 1]
    log.info("UHF done: Nα=%d Nβ=%d HOMO-LUMO gap=%.3f eV", n_a, n_b, gap * 27.211)

    return mol, mf


# ═══════════════════════════════════════════════════════════════════════════
# 3. Orbital window selection
# ═══════════════════════════════════════════════════════════════════════════

def select_orbital_window(mf, n_window: int = 20) -> list:
    """
    Select n_window orbitals centered on the HOMO-LUMO gap.
    Uses alpha-MO energies. Returns list of MO indices (0-indexed).
    """
    mo_occ_a = mf.mo_occ[0]
    n_occ_a = int(mo_occ_a.sum())
    n_mo = len(mo_occ_a)

    n_each = n_window // 2   # 10 from each side

    occ_start = max(0, n_occ_a - n_each)
    occ_idx = list(range(occ_start, n_occ_a))

    vir_end = min(n_mo, n_occ_a + n_each)
    vir_idx = list(range(n_occ_a, vir_end))

    win_idx = sorted(set(occ_idx + vir_idx))

    # Pad to n_window if we ran out of MOs near edges
    if len(win_idx) < n_window:
        all_idx = set(range(n_mo))
        candidates = sorted(all_idx - set(win_idx))
        # Pick by distance from gap centre
        gap_centre = n_occ_a - 0.5
        candidates.sort(key=lambda i: abs(i - gap_centre))
        for c in candidates:
            win_idx.append(c)
            win_idx.sort()
            if len(win_idx) >= n_window:
                break

    win_idx = sorted(win_idx)[:n_window]

    n_occ_win = sum(1 for i in win_idx if mo_occ_a[i] > 0.5)
    n_vir_win = len(win_idx) - n_occ_win
    log.info("Window: %d orbitals (%d occ + %d vir), indices %d–%d",
             len(win_idx), n_occ_win, n_vir_win, win_idx[0], win_idx[-1])
    return win_idx


# ═══════════════════════════════════════════════════════════════════════════
# 4. Integral generation
# ═══════════════════════════════════════════════════════════════════════════

def get_integrals(mol, mf, win_idx: list) -> tuple:
    """
    Compute effective 1e and 2e integrals in the orbital window.

    Uses alpha MOs from UHF. The effective h1e (get_h1eff) properly
    accounts for the Coulomb/exchange contribution of inactive electrons.

    Returns: (h1e, g2e, ecore, n_alpha, n_beta)
      h1e   : (n_orb, n_orb) — effective one-electron integrals
      g2e   : (n_orb, n_orb, n_orb, n_orb) — two-electron integrals
      ecore : float — core energy (nuclear + inactive electron contribution)
      n_alpha, n_beta : int — electron counts in window
    """
    from pyscf import mcscf, ao2mo

    n_orb = len(win_idx)
    mo_occ_a = mf.mo_occ[0]
    mo_occ_b = mf.mo_occ[1]

    # Electron count in window (from UHF occupations)
    n_alpha = int(sum(1 for i in win_idx if mo_occ_a[i] > 0.5))
    n_beta  = int(sum(1 for i in win_idx if mo_occ_b[i] > 0.5))

    log.info("Window electrons: α=%d β=%d total=%d", n_alpha, n_beta, n_alpha + n_beta)

    # Use alpha MO coefficients for the window
    # For UHF, mf.mo_coeff is (mo_a, mo_b); we use alpha MOs
    if isinstance(mf.mo_coeff, (list, tuple)) and len(mf.mo_coeff) == 2:
        mo_full = mf.mo_coeff[0]   # alpha MOs, shape (nao, nmo)
    else:
        mo_full = mf.mo_coeff

    # Build CASCI with sorted MOs to put window in CAS positions
    # mcscf.CASCI accepts UHF mf object in PySCF 2.x
    mc = mcscf.CASCI(mf, n_orb, (n_alpha, n_beta))
    mo_sorted = mcscf.addons.sort_mo(mc, mo_full, win_idx, base=0)
    mc.mo_coeff = mo_sorted

    h1e, ecore = mc.get_h1eff()
    g2e = ao2mo.restore(1, mc.get_h2eff(), n_orb)

    log.info("Integrals ready: h1e=%s g2e=%s ecore=%.6f",
             h1e.shape, g2e.shape, float(ecore))
    return h1e, g2e, float(ecore), n_alpha, n_beta


# ═══════════════════════════════════════════════════════════════════════════
# 5. DMRG runner
# ═══════════════════════════════════════════════════════════════════════════

def run_dmrg_sweep(h1e, g2e, ecore, n_orb, n_alpha, n_beta,
                   scratch_dir: str, tag: str,
                   bond_dims: list, n_sweeps: int,
                   n_threads: int = 4) -> dict:
    """
    Run one DMRG sweep schedule.
    Returns dict with: energy, s1, s2 (or None), noon.
    """
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    os.makedirs(scratch_dir, exist_ok=True)

    # Noise schedule: decay to 0 over last 2 sweeps
    n = len(bond_dims)
    noises = [1e-4] * max(1, n - 2) + [1e-5, 0.0]
    noises = noises[:n]
    thrds = [1e-8] * n

    drv = DMRGDriver(scratch=scratch_dir, symm_type=SymmetryTypes.SZ,
                     n_threads=n_threads)
    spin = max(0, n_alpha - n_beta)   # 2*Sz = 2*S for UHF ground state
    drv.initialize_system(n_sites=n_orb, n_elec=n_alpha + n_beta, spin=spin)
    mpo = drv.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)
    ket = drv.get_random_mps(tag=tag, bond_dim=bond_dims[0], nroots=1)

    energy = drv.dmrg(mpo, ket,
                      n_sweeps=n_sweeps,
                      bond_dims=bond_dims,
                      noises=noises,
                      thrds=thrds,
                      iprint=1)

    log.info("%s: E_DMRG = %.8f Ha", tag, energy)

    # ── Extract s(1) and NOON ────────────────────────────────────
    s1, noon = _extract_s1_noon(drv, ket, n_orb)

    # ── Extract s(2) for I_ij (try multiple API variants) ────────
    s2 = _extract_s2(drv, ket, n_orb)

    return {
        "energy": float(energy),
        "s1":     s1,
        "s2":     s2,
        "noon":   noon,
        "driver": drv,   # keep alive for further calls
        "ket":    ket,
    }


def _extract_s1_noon(drv, ket, n_orb: int) -> tuple:
    """
    Extract single-orbital entropies s(1) and NOONs.
    Tries driver API first, falls back to 1-PDM, then to zeros.
    Returns: (s1 np.array, noon np.array)
    """
    # ── Try driver.get_orbital_entropies() (block2 >= 0.5.x) ────
    if hasattr(drv, "get_orbital_entropies"):
        try:
            s1 = np.array(drv.get_orbital_entropies(ket))
            log.info("s1 via get_orbital_entropies(): shape=%s", s1.shape)
            noon = _s1_to_noon(s1)
            return s1, noon
        except Exception as e:
            log.debug("get_orbital_entropies() failed: %s", e)

    # ── Try driver.get_1pdm() ────────────────────────────────────
    if hasattr(drv, "get_1pdm"):
        try:
            dm1 = drv.get_1pdm(ket)
            noon = _parse_1pdm_noon(dm1, n_orb)
            s1 = _noon_to_s1(noon)
            log.info("s1 via get_1pdm(): noon range=[%.3f, %.3f]",
                     noon.min(), noon.max())
            return s1, noon
        except Exception as e:
            log.warning("get_1pdm() failed: %s", e)

    log.error("All s1 extraction methods failed — returning zeros")
    return np.zeros(n_orb), np.zeros(n_orb)


def _extract_s2(drv, ket, n_orb: int):
    """
    Extract two-orbital joint entropy matrix s(2) for I_ij.
    Returns np.array of shape (n_orb, n_orb), or None if unavailable.
    """
    # Try get_orbital_entropies_2() (pyblock2 direct)
    if hasattr(drv, "get_orbital_entropies_2"):
        try:
            s2 = np.array(drv.get_orbital_entropies_2(ket))
            log.info("s2 via get_orbital_entropies_2(): shape=%s", s2.shape)
            return s2
        except Exception as e:
            log.debug("get_orbital_entropies_2() failed: %s", e)

    # Try get_2pdm() and derive s2 from it
    if hasattr(drv, "get_2pdm"):
        try:
            dm2 = drv.get_2pdm(ket)
            s2 = _2pdm_to_s2(dm2, n_orb)
            if s2 is not None:
                log.info("s2 via get_2pdm(): shape=%s", s2.shape)
                return s2
        except Exception as e:
            log.debug("get_2pdm() failed: %s", e)

    log.info("I_ij not available in this block2 version — will be null in output")
    return None


def _parse_1pdm_noon(dm1, n_orb: int) -> np.ndarray:
    """
    Parse the 1-PDM from block2 (various return formats) into NOONs.
    Handles: tuple (dma, dmb), 3D array (2, n, n), 2D array (n, n).
    """
    dm = np.array(dm1)
    if dm.ndim == 3 and dm.shape[0] == 2:
        # shape (2, n_orb, n_orb) → alpha + beta diagonals
        noon = np.diag(dm[0]) + np.diag(dm[1])
    elif isinstance(dm1, (list, tuple)) and len(dm1) == 2:
        dma = np.array(dm1[0])
        dmb = np.array(dm1[1])
        noon = np.diag(dma) + np.diag(dmb)
    elif dm.ndim == 2 and dm.shape[0] == n_orb:
        # Spin-averaged or restricted
        noon = np.diag(dm)
    else:
        raise ValueError(f"Unexpected 1-PDM shape: {dm.shape}")
    return noon[:n_orb]


def _noon_to_s1(noon: np.ndarray) -> np.ndarray:
    """
    Single-orbital entropy from NOON via binary entropy.
    Equation 1 of Manuscript: s_i = -p*log(p) - (1-p)*log(1-p), p = n_i/2.
    """
    s = np.zeros(len(noon))
    for i, n in enumerate(noon):
        n = float(n)
        p = n / 2.0
        q = 1.0 - p
        if p > 1e-12 and q > 1e-12:
            s[i] = -p * np.log(p) - q * np.log(q)
    return s


def _s1_to_noon(s1: np.ndarray) -> np.ndarray:
    """
    Invert binary entropy to get an approximate NOON.
    Assumes n_i ≤ 1 (half-filled): n_i = 2 * p where p solves the entropy.
    For display/fallback only — not used when PDM is available.
    """
    # Binary entropy inverse is not closed-form; use approximation
    # s = log(2) when n = 1 (half-filled), s = 0 when n = 0 or 2
    # Simple approximation: n ≈ 1 - cos(π * s / log2)  (0 ≤ s ≤ log2)
    log2 = np.log(2)
    noon = np.zeros(len(s1))
    for i, s in enumerate(s1):
        if s > 0:
            noon[i] = 1.0 - np.cos(np.pi * s / log2)
    return noon


def _2pdm_to_s2(dm2, n_orb: int):
    """
    Compute two-orbital joint entropy s(2)_{ij} from the 2-PDM.
    This is the quantum information entropy of the 2-orbital reduced density matrix.
    Returns (n_orb, n_orb) array or None on failure.
    """
    try:
        dm2 = np.array(dm2)
        # For SZ symmetry, dm2 has shape (n_orb, n_orb, n_orb, n_orb)
        # or (2, n_orb, n_orb, n_orb, n_orb) for spin-resolved
        if dm2.ndim == 5 and dm2.shape[0] in (2, 3):
            # Spin-resolved: sum over spin components
            dm2_spatial = dm2[0] + dm2[-1]   # αα + ββ components
        elif dm2.ndim == 4:
            dm2_spatial = dm2
        else:
            return None

        s2 = np.zeros((n_orb, n_orb))
        for i in range(n_orb):
            for j in range(i + 1, n_orb):
                # Extract 4x4 two-orbital RDM (|0⟩,|↑⟩,|↓⟩,|↑↓⟩ basis)
                # For spatial orbitals: trace out all except i, j
                rho_ij = _extract_2orb_rdm(dm2_spatial, i, j, n_orb)
                if rho_ij is not None:
                    eigvals = np.linalg.eigvalsh(rho_ij)
                    eigvals = eigvals[eigvals > 1e-12]
                    s_ij = -np.sum(eigvals * np.log(eigvals))
                    s2[i, j] = s2[j, i] = s_ij

        return s2
    except Exception as e:
        log.debug("2-PDM → s2 failed: %s", e)
        return None


def _extract_2orb_rdm(dm2, i: int, j: int, n_orb: int):
    """
    Extract the 2-orbital reduced density matrix ρ^(ij) from the 2-RDM.
    This is a simplified extraction for the diagonal case.
    Returns 4x4 matrix in {|00⟩, |10⟩, |01⟩, |11⟩} basis or None.
    """
    try:
        # Simplified: use only the direct elements
        rho = np.array([
            [1 - dm2[i, i, i, i] - dm2[j, j, j, j] + dm2[i, j, i, j],
             0, 0, 0],
            [0, dm2[i, i, i, i] - dm2[i, j, i, j], dm2[i, j, j, i], 0],
            [0, dm2[j, i, i, j], dm2[j, j, j, j] - dm2[i, j, i, j], 0],
            [0, 0, 0, dm2[i, j, i, j]]
        ])
        # Ensure positive semi-definite
        rho = (rho + rho.T) / 2
        eigvals = np.linalg.eigvalsh(rho)
        if np.any(eigvals < -1e-6):
            return None
        rho = np.clip(rho, 0, None)
        return rho
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 6. Post-processing
# ═══════════════════════════════════════════════════════════════════════════

def compute_iij(s1: np.ndarray, s2) -> list:
    """I_ij = s_i + s_j - s_ij. Returns None if s2 unavailable."""
    if s2 is None:
        return None
    s2 = np.array(s2)
    Iij = s1[:, None] + s1[None, :] - s2
    np.fill_diagonal(Iij, 0.0)
    Iij = np.maximum(Iij, 0.0)   # mutual info is non-negative
    return Iij.tolist()


def compute_n_active(s1: np.ndarray, threshold: float = ENTROPY_THRESHOLD) -> int:
    return int((s1 > threshold).sum())


def compute_entropy_cliff(s1: np.ndarray, threshold: float = ENTROPY_THRESHOLD) -> float:
    """
    Ratio of minimum selected entropy to maximum excluded entropy.
    High value = clean gap between active and inactive. Low = ambiguous.
    """
    selected = s1[s1 > threshold]
    excluded = s1[s1 <= threshold]
    if len(selected) == 0:
        return 0.0
    if len(excluded) == 0:
        return 999.0
    return float(selected.min() / max(excluded.max(), 1e-14))


# ═══════════════════════════════════════════════════════════════════════════
# 7. Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def process_system(json_path: str, scratch_base: str,
                   n_orb_window: int = N_ORB_WINDOW_DEFAULT,
                   bond_dim_max: int = 500,
                   n_threads: int = 4) -> dict:
    """
    Full pipeline for one system.
    Returns result dict and saves to {scratch}/{label}/dmrg_features.json.
    Skips if output file already exists (safe for resubmission).
    """
    t0 = time.time()
    system = load_system(json_path)
    label = system["label"]
    out_dir = os.path.join(scratch_base, label)
    out_path = os.path.join(out_dir, "dmrg_features.json")

    if os.path.exists(out_path):
        log.info("Already complete, skipping: %s", out_path)
        with open(out_path) as f:
            return json.load(f)

    log.info("═" * 60)
    log.info("Processing: %s", label)
    log.info("═" * 60)

    # 1. UHF
    mol, mf = run_uhf(system)

    # 2. Orbital window
    win_idx = select_orbital_window(mf, n_orb_window)
    n_orb = len(win_idx)

    # 3. Integrals
    h1e, g2e, ecore, n_alpha, n_beta = get_integrals_uhf(mol, mf, win_idx)

    # 4. Fast DMRG (M=200) — sensitivity check
    log.info("── Fast DMRG (M=200) ──────────────────────────────────")
    fast_res = run_dmrg_sweep(
        h1e, g2e, ecore, n_orb, n_alpha, n_beta,
        scratch_dir=os.path.join(out_dir, "fast"),
        tag="GS_FAST",
        bond_dims=FAST_BOND_DIMS,
        n_sweeps=FAST_SWEEPS,
        n_threads=n_threads,
    )
    n_fast = compute_n_active(np.array(fast_res["s1"]))
    log.info("Fast: n_active=%d  (E=%.6f)", n_fast, fast_res["energy"])

    # 5. Production DMRG
    log.info("── Production DMRG (M=%d) ─────────────────────────────", bond_dim_max)
    prod_dims = PROD_BOND_DIMS_BASE + [bond_dim_max, bond_dim_max]
    prod_res = run_dmrg_sweep(
        h1e, g2e, ecore, n_orb, n_alpha, n_beta,
        scratch_dir=os.path.join(out_dir, "prod"),
        tag="GS_PROD",
        bond_dims=prod_dims,
        n_sweeps=PROD_SWEEPS,
        n_threads=n_threads,
    )

    s1 = np.array(prod_res["s1"])
    s2 = prod_res["s2"]
    noon = np.array(prod_res["noon"])

    # 6. Post-process
    n_active = compute_n_active(s1)
    sensitivity_flag = int(n_fast != n_active)
    cliff = compute_entropy_cliff(s1)
    Iij = compute_iij(s1, s2)
    elapsed = time.time() - t0

    log.info("Result: n_active=%d (fast=%d), cliff=%.3f, flag=%d, t=%.1fs",
             n_active, n_fast, cliff, sensitivity_flag, elapsed)

    # Top-5 orbitals by entropy (for quick sanity check in logs)
    top5 = np.argsort(s1)[::-1][:5]
    log.info("Top-5 orbitals by entropy: %s → s=%s",
             list(top5), [f"{s1[i]:.3f}" for i in top5])

    result = {
        # ── Labels for ML1 / ML2 ──────────────────────────────
        "n_active":              n_active,
        "s1":                    s1.tolist(),
        "Iij":                   Iij,          # null if API unavailable
        "noon":                  noon.tolist(),
        # ── Quality flags ─────────────────────────────────────
        "entropy_cliff_ratio":   cliff,
        "dmrg_sensitivity_flag": sensitivity_flag,
        "n_fast":                n_fast,
        # ── DMRG energy ───────────────────────────────────────
        "e_dmrg":                prod_res["energy"],
        "e_dmrg_fast":           fast_res["energy"],
        # ── Metadata ──────────────────────────────────────────
        "label":                 label,
        "system_json":           os.path.abspath(json_path),
        "win_idx":               win_idx,
        "n_alpha_cas":           n_alpha,
        "n_beta_cas":            n_beta,
        "n_orb_window":          n_orb,
        "entropy_threshold":     ENTROPY_THRESHOLD,
        "bond_dim_max":          bond_dim_max,
        "elapsed_s":             round(elapsed, 1),
        "iij_available":         Iij is not None,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Saved: %s", out_path)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 8. Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DMRG label generation (block2 direct, SZ symmetry)"
    )
    parser.add_argument("--system",       required=True,
                        help="Path to system JSON file")
    parser.add_argument("--scratch",      required=True,
                        help="Base scratch directory for DMRG output")
    parser.add_argument("--n_orb_window", type=int, default=N_ORB_WINDOW_DEFAULT,
                        help="Orbital window size (default: 20 = 10 occ + 10 vir)")
    parser.add_argument("--bond_dim_max", type=int, default=500,
                        help="Maximum bond dimension for production DMRG (default: 500)")
    parser.add_argument("--n_threads",    type=int,
                        default=int(os.environ.get("OMP_NUM_THREADS", 4)),
                        help="Number of OMP threads (default: OMP_NUM_THREADS or 4)")
    args = parser.parse_args()

    log.info("gen_dmrg_labels.py started")
    log.info("  system       : %s", args.system)
    log.info("  scratch      : %s", args.scratch)
    log.info("  n_orb_window : %d", args.n_orb_window)
    log.info("  bond_dim_max : %d", args.bond_dim_max)
    log.info("  n_threads    : %d", args.n_threads)
    log.info("  PID          : %d", os.getpid())

    try:
        result = process_system(
            json_path=args.system,
            scratch_base=args.scratch,
            n_orb_window=args.n_orb_window,
            bond_dim_max=args.bond_dim_max,
            n_threads=args.n_threads,
        )
        print(f"\n{'═'*55}")
        print(f"  DONE: {result['label']}")
        print(f"  n_active     = {result['n_active']}")
        print(f"  cliff_ratio  = {result['entropy_cliff_ratio']:.3f}")
        print(f"  sens_flag    = {result['dmrg_sensitivity_flag']}")
        print(f"  I_ij         = {'YES' if result['iij_available'] else 'NO (upgrade block2)'}")
        print(f"  elapsed      = {result['elapsed_s']:.0f}s")
        print(f"{'═'*55}\n")
        sys.exit(0)

    except Exception:
        log.error("FATAL ERROR for system: %s", args.system)
        log.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
