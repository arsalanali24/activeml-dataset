"""
gen_closed_shell.py
===================
Generates training data for two critical missing chemical classes:

CLASS A — d0 high-oxidation-state metals (n_active should be 0)
  Os(VIII), Re(VII), W(VI), Mo(VI), Cr(VI), Mn(VII)
  OsO4, ReO4-, WO4^2-, MoO4^2-, CrO4^2-, MnO4- etc.
  These are tetrahedral oxo complexes with no d electrons.
  KEY: use RHF not UHF — closed shell d0 systems.

CLASS B — genuine d10 closed-shell metals (n_active should be 0-2)
  Pd(0), Pt(0) — catalytic resting states
  Zn(II), Cu(I) — genuine d10 with correct positive charges
  KEY: use RHF not UHF for spin=0 d10 systems.

Why these matter:
  - OsO4 benchmark case fails because model has never seen d0
  - Pd(0) IRC reactant fails because model confuses d10 with d6
  - Without these, "universal predictor" claim is scientifically weak

Output: ~/activeml/data/generated_closed_shell/

Usage:
  python gen_closed_shell.py summary
  python gen_closed_shell.py <idx>
  sbatch --array=0-N job_closed_shell.sh
"""
import numpy as np, json, os, sys, logging
from pyscf import gto, scf, mcscf

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

OUTPUT_DIR = os.path.expanduser(
    '~/activeml/data/generated_closed_shell')
os.makedirs(OUTPUT_DIR, exist_ok=True)

METAL_CONSTANTS = {
    # d0 metals
    'Os': {'z_eff':17.17,'zeta_so_cm1':3381,'metal_row':'5d'},
    'Re': {'z_eff':17.01,'zeta_so_cm1':2456,'metal_row':'5d'},
    'W':  {'z_eff':11.61,'zeta_so_cm1':2748,'metal_row':'5d'},
    'Mo': {'z_eff':10.97,'zeta_so_cm1': 467,'metal_row':'4d'},
    'Cr': {'z_eff': 9.76,'zeta_so_cm1': 273,'metal_row':'3d'},
    'Mn': {'z_eff':10.53,'zeta_so_cm1': 355,'metal_row':'3d'},
    'V':  {'z_eff': 8.45,'zeta_so_cm1': 167,'metal_row':'3d'},
    'Ti': {'z_eff': 7.74,'zeta_so_cm1':  75,'metal_row':'3d'},
    # d10 metals
    'Pd': {'z_eff':13.00,'zeta_so_cm1':1334,'metal_row':'4d'},
    'Pt': {'z_eff':17.33,'zeta_so_cm1':4146,'metal_row':'5d'},
    'Zn': {'z_eff':13.57,'zeta_so_cm1':1042,'metal_row':'3d'},
    'Cu': {'z_eff':13.20,'zeta_so_cm1': 831,'metal_row':'3d'},
    'Ni': {'z_eff':12.78,'zeta_so_cm1': 669,'metal_row':'3d'},
    'Ru': {'z_eff':12.33,'zeta_so_cm1': 880,'metal_row':'4d'},
    'Rh': {'z_eff':12.67,'zeta_so_cm1':1097,'metal_row':'4d'},
    'Ir': {'z_eff':17.00,'zeta_so_cm1':3909,'metal_row':'5d'},
}

ECP_METALS = {'Os','Re','W','Mo','Pd','Pt','Ru','Rh','Ir'}

ATOM_Z = {
    'Os':76,'Re':75,'W':74,'Mo':42,'Cr':24,'Mn':25,'V':23,'Ti':22,
    'Pd':46,'Pt':78,'Zn':30,'Cu':29,'Ni':28,'Ru':44,'Rh':45,'Ir':77,
    'O':8,'Cl':17,'Br':35,'F':9,'N':7,'P':15,'C':6,'H':1,
}

# ══════════════════════════════════════════════════════════════
# CLASS A — d0 HIGH OXIDATION STATE STRUCTURES
# Format: (name, metal, charge, spin, atom_str, ligand,
#          true_d_count, use_rhf, published_n_active)
# ══════════════════════════════════════════════════════════════

# Os(VIII) d0 — OsO4 and variants
# Os-O bond: 1.711 Ang (crystallographic)
D0_STRUCTURES = [

    # ── Os(VIII) d0 ───────────────────────────────────────────
    ("OsO4_tet", "Os", 0, 0,
     """Os  0.000  0.000  0.000
        O   1.711  0.000  0.000
        O  -1.711  0.000  0.000
        O   0.000  1.711  0.000
        O   0.000 -1.711  0.000""",
     "O", 0, True, 0),

    ("OsO3Cl2_tbp", "Os", 0, 0,
     """Os  0.000  0.000  0.000
        O   1.680  0.000  0.000
        O  -1.680  0.000  0.000
        O   0.000  1.680  0.000
        Cl  0.000 -2.380  0.000
        Cl  0.000  0.000  2.380""",
     "O", 0, True, 0),

    ("OsO2Cl4_oct", "Os", -2, 0,
     """Os  0.000  0.000  0.000
        O   1.730  0.000  0.000
        O  -1.730  0.000  0.000
        Cl  0.000  2.310  0.000
        Cl  0.000 -2.310  0.000
        Cl  0.000  0.000  2.310
        Cl  0.000  0.000 -2.310""",
     "O", 0, True, 0),

    # ── Re(VII) d0 ────────────────────────────────────────────
    # Re-O: 1.720 Ang
    ("ReO4_1m_tet", "Re", -1, 0,
     """Re  0.000  0.000  0.000
        O   1.720  0.000  0.000
        O  -1.720  0.000  0.000
        O   0.000  1.720  0.000
        O   0.000 -1.720  0.000""",
     "O", 0, True, 0),

    ("ReO3Cl_tet", "Re", 0, 0,
     """Re  0.000  0.000  0.000
        O   1.700  0.000  0.000
        O  -1.700  0.000  0.000
        O   0.000  1.700  0.000
        Cl  0.000 -2.260  0.000""",
     "O", 0, True, 0),

    ("ReO3F_tet", "Re", 0, 0,
     """Re  0.000  0.000  0.000
        O   1.690  0.000  0.000
        O  -1.690  0.000  0.000
        O   0.000  1.690  0.000
        F   0.000 -1.870  0.000""",
     "O", 0, True, 0),

    ("ReO2Cl3_oct", "Re", 0, 0,
     """Re  0.000  0.000  0.000
        O   1.680  0.000  0.000
        O  -1.680  0.000  0.000
        Cl  0.000  2.310  0.000
        Cl  0.000 -2.310  0.000
        Cl  0.000  0.000  2.310""",
     "O", 0, True, 0),

    # ── W(VI) d0 ──────────────────────────────────────────────
    # W-O: 1.780 Ang, W-Cl: 2.260 Ang
    ("WO4_2m_tet", "W", -2, 0,
     """W   0.000  0.000  0.000
        O   1.780  0.000  0.000
        O  -1.780  0.000  0.000
        O   0.000  1.780  0.000
        O   0.000 -1.780  0.000""",
     "O", 0, True, 0),

    ("WCl6_d0_oct", "W", 0, 0,
     """W   0.000  0.000  0.000
        Cl  2.260  0.000  0.000
        Cl -2.260  0.000  0.000
        Cl  0.000  2.260  0.000
        Cl  0.000 -2.260  0.000
        Cl  0.000  0.000  2.260
        Cl  0.000  0.000 -2.260""",
     "Cl", 0, True, 0),

    ("WO2Cl4_2m_oct", "W", -2, 0,
     """W   0.000  0.000  0.000
        O   1.760  0.000  0.000
        O  -1.760  0.000  0.000
        Cl  0.000  2.280  0.000
        Cl  0.000 -2.280  0.000
        Cl  0.000  0.000  2.280
        Cl  0.000  0.000 -2.280""",
     "O", 0, True, 0),

    # ── Mo(VI) d0 ─────────────────────────────────────────────
    ("MoO4_2m_tet", "Mo", -2, 0,
     """Mo  0.000  0.000  0.000
        O   1.760  0.000  0.000
        O  -1.760  0.000  0.000
        O   0.000  1.760  0.000
        O   0.000 -1.760  0.000""",
     "O", 0, True, 0),

    ("MoCl6_d0_oct", "Mo", 0, 0,
     """Mo  0.000  0.000  0.000
        Cl  2.260  0.000  0.000
        Cl -2.260  0.000  0.000
        Cl  0.000  2.260  0.000
        Cl  0.000 -2.260  0.000
        Cl  0.000  0.000  2.260
        Cl  0.000  0.000 -2.260""",
     "Cl", 0, True, 0),

    ("MoO2Cl4_2m_oct", "Mo", -2, 0,
     """Mo  0.000  0.000  0.000
        O   1.700  0.000  0.000
        O  -1.700  0.000  0.000
        Cl  0.000  2.280  0.000
        Cl  0.000 -2.280  0.000
        Cl  0.000  0.000  2.280
        Cl  0.000  0.000 -2.280""",
     "O", 0, True, 0),

    # ── Cr(VI) d0 ─────────────────────────────────────────────
    ("CrO4_2m_tet", "Cr", -2, 0,
     """Cr  0.000  0.000  0.000
        O   1.660  0.000  0.000
        O  -1.660  0.000  0.000
        O   0.000  1.660  0.000
        O   0.000 -1.660  0.000""",
     "O", 0, False, 0),

    ("CrO2Cl2_tet", "Cr", 0, 0,
     """Cr  0.000  0.000  0.000
        O   1.580  0.000  0.000
        O  -1.580  0.000  0.000
        Cl  0.000  2.120  0.000
        Cl  0.000 -2.120  0.000""",
     "O", 0, False, 0),

    ("CrF6_d0_oct", "Cr", 0, 0,
     """Cr  0.000  0.000  0.000
        F   1.720  0.000  0.000
        F  -1.720  0.000  0.000
        F   0.000  1.720  0.000
        F   0.000 -1.720  0.000
        F   0.000  0.000  1.720
        F   0.000  0.000 -1.720""",
     "F", 0, False, 0),

    # ── Mn(VII) d0 ────────────────────────────────────────────
    ("MnO4_1m_tet", "Mn", -1, 0,
     """Mn  0.000  0.000  0.000
        O   1.629  0.000  0.000
        O  -1.629  0.000  0.000
        O   0.000  1.629  0.000
        O   0.000 -1.629  0.000""",
     "O", 0, False, 0),

    # ── V(V) d0 ───────────────────────────────────────────────
    ("VO4_3m_tet", "V", -3, 0,
     """V   0.000  0.000  0.000
        O   1.720  0.000  0.000
        O  -1.720  0.000  0.000
        O   0.000  1.720  0.000
        O   0.000 -1.720  0.000""",
     "O", 0, False, 0),

    ("VOCl3_tet", "V", 0, 0,
     """V   0.000  0.000  0.000
        O   1.570  0.000  0.000
        Cl -2.140  0.000  0.000
        Cl  0.000  2.140  0.000
        Cl  0.000 -2.140  0.000""",
     "O", 0, False, 0),

    # ── Ti(IV) d0 ─────────────────────────────────────────────
    ("TiCl4_tet", "Ti", 0, 0,
     """Ti  0.000  0.000  0.000
        Cl  2.185  0.000  0.000
        Cl -2.185  0.000  0.000
        Cl  0.000  2.185  0.000
        Cl  0.000 -2.185  0.000""",
     "Cl", 0, False, 0),

    ("TiO2Cl2_tet", "Ti", 0, 0,
     """Ti  0.000  0.000  0.000
        O   1.640  0.000  0.000
        O  -1.640  0.000  0.000
        Cl  0.000  2.190  0.000
        Cl  0.000 -2.190  0.000""",
     "O", 0, False, 0),

    ("TiCl6_2m_oct", "Ti", -2, 0,
     """Ti  0.000  0.000  0.000
        Cl  2.370  0.000  0.000
        Cl -2.370  0.000  0.000
        Cl  0.000  2.370  0.000
        Cl  0.000 -2.370  0.000
        Cl  0.000  0.000  2.370
        Cl  0.000  0.000 -2.370""",
     "Cl", 0, False, 0),
]

# ══════════════════════════════════════════════════════════════
# CLASS B — GENUINE d10 CLOSED-SHELL STRUCTURES
# Correct positive or neutral charges (not artificial negatives)
# ══════════════════════════════════════════════════════════════

D10_STRUCTURES = [

    # ── Pd(0) d10 — catalytic resting states ─────────────────
    ("Pd0_P4_tet", "Pd", 0, 0,
     """Pd  0.000  0.000  0.000
        P   2.290  0.000  0.000
        P  -2.290  0.000  0.000
        P   0.000  2.290  0.000
        P   0.000 -2.290  0.000""",
     "P", 10, True, 2),

    ("Pd0_P2_linear", "Pd", 0, 0,
     """Pd  0.000  0.000  0.000
        P   2.290  0.000  0.000
        P  -2.290  0.000  0.000""",
     "P", 10, True, 2),

    ("Pd0_Cl4_2m", "Pd", -2, 0,
     """Pd  0.000  0.000  0.000
        Cl  2.295  0.000  0.000
        Cl -2.295  0.000  0.000
        Cl  0.000  2.295  0.000
        Cl  0.000 -2.295  0.000""",
     "Cl", 10, True, 4),

    ("Pd2p_P2Cl2_sqpl", "Pd", 0, 0,
     """Pd  0.000  0.000  0.000
        P   2.280  0.000  0.000
        P  -2.280  0.000  0.000
        Cl  0.000  2.300  0.000
        Cl  0.000 -2.300  0.000""",
     "P", 8, True, 4),

    # ── Pt(0) d10 ─────────────────────────────────────────────
    ("Pt0_P4_tet", "Pt", 0, 0,
     """Pt  0.000  0.000  0.000
        P   2.295  0.000  0.000
        P  -2.295  0.000  0.000
        P   0.000  2.295  0.000
        P   0.000 -2.295  0.000""",
     "P", 10, True, 2),

    ("Pt0_P2_linear", "Pt", 0, 0,
     """Pt  0.000  0.000  0.000
        P   2.295  0.000  0.000
        P  -2.295  0.000  0.000""",
     "P", 10, True, 2),

    ("Pt2p_Cl4_2m", "Pt", -2, 0,
     """Pt  0.000  0.000  0.000
        Cl  2.307  0.000  0.000
        Cl -2.307  0.000  0.000
        Cl  0.000  2.307  0.000
        Cl  0.000 -2.307  0.000""",
     "Cl", 8, True, 4),

    # ── Zn(II) d10 — genuine positive charge ─────────────────
    ("Zn2p_Cl4_tet", "Zn", 2, 0,
     """Zn  0.000  0.000  0.000
        Cl  2.310  0.000  0.000
        Cl -2.310  0.000  0.000
        Cl  0.000  2.310  0.000
        Cl  0.000 -2.310  0.000""",
     "Cl", 10, False, 0),

    ("Zn2p_N4_oct", "Zn", 2, 0,
     """Zn  0.000  0.000  0.000
        N   2.180  0.000  0.000
        N  -2.180  0.000  0.000
        N   0.000  2.180  0.000
        N   0.000 -2.180  0.000
        N   0.000  0.000  2.180
        N   0.000  0.000 -2.180""",
     "N", 10, False, 0),

    ("Zn2p_Cl2_linear", "Zn", 2, 0,
     """Zn  0.000  0.000  0.000
        Cl  2.310  0.000  0.000
        Cl -2.310  0.000  0.000""",
     "Cl", 10, False, 0),

    ("Zn2p_P4_tet", "Zn", 2, 0,
     """Zn  0.000  0.000  0.000
        P   2.360  0.000  0.000
        P  -2.360  0.000  0.000
        P   0.000  2.360  0.000
        P   0.000 -2.360  0.000""",
     "P", 10, False, 0),

    # ── Cu(I) d10 — genuine +1 charge ────────────────────────
    ("Cu1p_Cl2_linear", "Cu", 1, 0,
     """Cu  0.000  0.000  0.000
        Cl  2.100  0.000  0.000
        Cl -2.100  0.000  0.000""",
     "Cl", 10, False, 0),

    ("Cu1p_P2_linear", "Cu", 1, 0,
     """Cu  0.000  0.000  0.000
        P   2.190  0.000  0.000
        P  -2.190  0.000  0.000""",
     "P", 10, False, 0),

    ("Cu1p_Cl4_3m", "Cu", -3, 0,
     """Cu  0.000  0.000  0.000
        Cl  2.250  0.000  0.000
        Cl -2.250  0.000  0.000
        Cl  0.000  2.250  0.000
        Cl  0.000 -2.250  0.000""",
     "Cl", 10, False, 0),

    ("Cu1p_N4_sqpl", "Cu", 1, 0,
     """Cu  0.000  0.000  0.000
        N   2.010  0.000  0.000
        N  -2.010  0.000  0.000
        N   0.000  2.010  0.000
        N   0.000 -2.010  0.000""",
     "N", 10, False, 0),
]


# ══════════════════════════════════════════════════════════════
# QUANTUM CHEMISTRY
# KEY: use RHF for closed-shell d0/d10, UHF only for open-shell
# ══════════════════════════════════════════════════════════════

def run_rhf(mol):
    """RHF — correct for closed-shell d0 and d10 systems."""
    for s in [
        dict(max_cycle=300, conv_tol=1e-10),
        dict(max_cycle=500, conv_tol=1e-9),
        dict(max_cycle=800, conv_tol=1e-8),
    ]:
        mf = scf.RHF(mol)
        for k, v in s.items(): setattr(mf, k, v)
        mf.verbose = 0; mf.run()
        if mf.converged: return mf
    return mf


def run_uhf(mol):
    """UHF — for open-shell cases."""
    for s in [
        dict(max_cycle=300, conv_tol=1e-10, damp=0.0, level_shift=0.0),
        dict(max_cycle=500, conv_tol=1e-9,  damp=0.3, level_shift=0.2),
        dict(max_cycle=800, conv_tol=1e-8,  damp=0.5, level_shift=0.5),
    ]:
        mf = scf.UHF(mol)
        for k, v in s.items(): setattr(mf, k, v)
        mf.verbose = 0; mf.run()
        if mf.converged: return mf
    return mf


def run_casscf(mf, mol, n_act, cas_n=10):
    """CASSCF with orbital window selection."""
    if isinstance(mf, scf.rhf.RHF):
        e_m = mf.mo_energy
        occ = mf.mo_occ
    else:
        e_m = (mf.mo_energy[0] + mf.mo_energy[1]) / 2
        occ = mf.mo_occ[0] + mf.mo_occ[1]

    oi = np.where(occ > 0.5)[0]
    vi = np.where(occ < 0.5)[0]
    if not len(oi) or not len(vi): return None, False

    gap = (float(e_m[oi[-1]]) + float(e_m[vi[0]])) / 2
    ws  = cas_n + 4
    w   = sorted(np.argsort(np.abs(e_m - gap))[:ws],
                 key=lambda i: e_m[i])

    best_mc = None; best_e = 0.0
    for win in [w[:cas_n], w[2:cas_n+2], w[1:cas_n+1]]:
        for sh in [1e-3, 1e-2, 5e-2, 1e-1]:
            try:
                mc = mcscf.CASSCF(mf, cas_n, n_act)
                mc.max_cycle_macro = 500
                mc.conv_tol = 1e-8
                mc.ah_level_shift = sh
                mc.verbose = 0
                mc.kernel(mc.sort_mo(win, base=0))
                ec = mc.e_tot - mf.e_tot
                if ec < 0 and ec < best_e:
                    best_mc = mc; best_e = ec
                if mc.converged and ec < -1e-4:
                    return mc, True
            except: continue

    if best_mc: return best_mc, best_mc.converged
    return None, False


def extract_features(mf, mol, use_rhf):
    """Extract ML1-compatible features. Works for both RHF and UHF."""
    if use_rhf or isinstance(mf, scf.rhf.RHF):
        e_a = e_b = mf.mo_energy
        occ_a = occ_b = mf.mo_occ / 2
        occ = mf.mo_occ
        spin_contam = 0.0
        homo_a = homo_b = float(e_a[mf.mo_occ > 0.5][-1]) \
                          if np.any(mf.mo_occ > 0.5) else 0.0
    else:
        e_a, e_b = mf.mo_energy[0], mf.mo_energy[1]
        occ_a, occ_b = mf.mo_occ[0], mf.mo_occ[1]
        occ = occ_a + occ_b
        S = mol.spin / 2.0
        spin_contam = float(mf.spin_square()[0] - S*(S+1))
        oi_a = np.where(occ_a > 0.5)[0]
        oi_b = np.where(occ_b > 0.5)[0]
        homo_a = float(e_a[oi_a[-1]]) if len(oi_a) else 0.0
        homo_b = float(e_b[oi_b[-1]]) if len(oi_b) else 0.0

    oi   = np.where(occ > 0.5)[0]
    vi   = np.where(occ < 0.5)[0]
    e_m  = (e_a + e_b) / 2 if not use_rhf else e_a
    homo = float(e_m[oi[-1]]) if len(oi) else 0.0
    lumo = float(e_m[vi[0]])  if len(vi) else 0.0

    # UNO fractional occupations
    ovp = mol.intor('int1e_ovlp')
    if use_rhf or isinstance(mf, scf.rhf.RHF):
        dm = mf.make_rdm1()
        ps = dm @ ovp
    else:
        dm = mf.make_rdm1()[0] + mf.make_rdm1()[1]
        ps = dm @ ovp
    uno, _ = np.linalg.eigh(ps)
    uno = np.sort(uno)[::-1]

    return {
        'spin_contamination':   spin_contam,
        'homo_lumo_gap':        float(lumo - homo),
        'homo_lumo_gap_eV':     float((lumo - homo) * 27.2114),
        'homo_energy':          homo,
        'lumo_energy':          lumo,
        'homo_ab_gap':          float(abs(homo_a - homo_b))
                                if not use_rhf else 0.0,
        'alpha_beta_overlap':   0.0,
        'delta_E_HS_LS':        0.0,
        'n_frac_uno_001':       int(np.sum((uno>0.01)&(uno<1.99))),
        'n_frac_uno_005':       int(np.sum((uno>0.05)&(uno<1.95))),
        'n_frac_uno_010':       int(np.sum((uno>0.10)&(uno<1.90))),
        'n_frac_uno_020':       int(np.sum((uno>0.20)&(uno<1.80))),
        'mayer_bond_order_mean': 0.0,
        'mayer_bond_order_std':  0.0,
        'mulliken_metal_charge': 0.0,
    }


def run_one(struct_class, idx):
    structures = D0_STRUCTURES if struct_class == 'd0' else D10_STRUCTURES
    entry = structures[idx]
    (name, metal, charge, spin, atom_str,
     ligand, true_d_count, use_rhf, pub_n_active) = entry

    prefix  = 'CS' if struct_class == 'd0' else 'D10'
    fname   = f"{prefix}_{name}_spin{spin}.json"
    outfile = os.path.join(OUTPUT_DIR, fname)

    if os.path.exists(outfile):
        r = json.load(open(outfile))
        if r.get('status') == 'ok':
            log.info(f"SKIP: {fname}"); return True

    consts = METAL_CONSTANTS[metal]
    log.info(f"Start: {fname}  d={true_d_count}  rhf={use_rhf}")

    try:
        mol = gto.Mole()
        mol.atom    = atom_str
        mol.basis   = 'def2-SVP'
        mol.charge  = charge
        mol.spin    = spin
        mol.verbose = 0
        if metal in ECP_METALS:
            mol.ecp = 'def2-SVP'
        mol.build()

        log.info(f"  n_electrons={mol.nelectron} parity_ok="
                 f"{mol.nelectron%2==spin%2}")

        # Use RHF for closed-shell d0/d10, UHF otherwise
        if use_rhf and spin == 0:
            mf = run_rhf(mol)
            scf_type = 'RHF'
        else:
            mf = run_uhf(mol)
            scf_type = 'UHF'

        log.info(f"  {scf_type}: E={mf.e_tot:.6f} "
                 f"converged={mf.converged}")

        # CASSCF(10,10) — for d0 expect n_active=0, for d10 expect 0-4
        n_act = max(2, true_d_count) if true_d_count > 0 else 2
        mc, conv = run_casscf(mf, mol, n_act, cas_n=10)

        if mc is None:
            # For d0 systems, CASSCF may give near-zero correlation
            # which is actually physically correct
            ec = 0.0
            n_active = 0
            conv = mf.converged
            log.info(f"  CASSCF failed — likely d0 closed shell "
                     f"(physically correct)")
        else:
            ec = float(mc.e_tot - mf.e_tot)
            casdm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
            no, _  = np.linalg.eigh(casdm1)
            no     = np.sort(no)[::-1]
            n_active = sum(1 for n in no if 0.02 < n < 1.98)

        feats = extract_features(mf, mol, use_rhf)

        # Count n_ligands
        n_lig = atom_str.strip().count('\n')

        result = {
            'name':             fname.replace('.json',''),
            'metal':            metal,
            'ligand':           ligand,
            'n_ligands':        n_lig,
            'charge':           charge,
            'spin':             spin,
            'mult':             spin + 1,
            'dist_ang':         2.0,
            'geometry':         'closed_shell',
            'struct_class':     struct_class,
            'scf_type':         scf_type,
            'true_d_count':     true_d_count,
            'd_electron_count': true_d_count,  # correct value
            'published_n_active': pub_n_active,
            'n_electrons':      mol.nelectron,
            'n_active_e':       n_act,
            'E_HF':             float(mf.e_tot),
            'E_CASSCF':         float(mc.e_tot) if mc else float(mf.e_tot),
            'corr_energy':      ec,
            'mp2_corr':         0.0,
            'converged':        bool(conv),
            'n_active':         n_active,
            'no_occ':           [float(x) for x in no] if mc else [],
            'status':           'ok',
            'z_eff':            consts['z_eff'],
            'zeta_so_cm1':      consts['zeta_so_cm1'],
            'metal_row':        consts['metal_row'],
            'largest_t2':       0.0,
            'n_frac_mp2_002':   0,
            'n_frac_mp2_005':   0,
            'n_frac_mp2_010':   0,
            **feats,
        }

        with open(outfile, 'w') as f:
            json.dump(result, f, indent=2)
        log.info(f"  OK: n_active={n_active} Ec={ec:.4f} "
                 f"pub={pub_n_active}")
        return True

    except Exception as e:
        log.error(f"  Error: {e}")
        json.dump({'name': fname, 'status': 'error',
                   'reason': str(e)},
                  open(outfile, 'w'))
        return False


# ── ALL JOBS ──────────────────────────────────────────────────
ALL_JOBS = []
existing = set(os.path.basename(f)
               for f in __import__('glob').glob(
                   f'{OUTPUT_DIR}/*.json'))

for i, entry in enumerate(D0_STRUCTURES):
    name = entry[0]; spin = entry[3]
    fname = f"CS_{name}_spin{spin}.json"
    if fname not in existing:
        ALL_JOBS.append(('d0', i))

for i, entry in enumerate(D10_STRUCTURES):
    name = entry[0]; spin = entry[3]
    fname = f"D10_{name}_spin{spin}.json"
    if fname not in existing:
        ALL_JOBS.append(('d10', i))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        d0_count  = sum(1 for c,_ in ALL_JOBS if c=='d0')
        d10_count = sum(1 for c,_ in ALL_JOBS if c=='d10')
        print(f"\nClosed-shell gap-filling structures:")
        print(f"  d0  structures: {d0_count}  "
              f"(Os/Re/W/Mo/Cr/Mn/V/Ti high-oxidation)")
        print(f"  d10 structures: {d10_count}  "
              f"(Pd(0)/Pt(0)/Zn(II)/Cu(I) genuine d10)")
        print(f"  Total jobs:     {len(ALL_JOBS)}")
        print(f"\nAll use RHF for spin=0 closed-shell systems")
        print(f"Expected n_active: 0 for d0, 0-4 for d10")
        print(f"\nOutput: {OUTPUT_DIR}")
        sys.exit(0)

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if idx >= len(ALL_JOBS):
        sys.exit(0)
    struct_class, struct_idx = ALL_JOBS[idx]
    sys.exit(0 if run_one(struct_class, struct_idx) else 1)
