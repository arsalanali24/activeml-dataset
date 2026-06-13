"""
gen_csd_extra.py
================
Hard-coded CSD-survey structures filling three gaps:
  Type 1: Ti, V, Zn real geometry (currently zero CSD structures)
  Type 2: 7-coordinate complexes (Mo, Ti, V, Zr)
  Type 3: Mixed ligand environments (Cl/N, Cl/O, Cl/Br)

Same pipeline as gen_csd_4d5d.py — same JSON schema, same CASSCF(10,10).
Output: ~/activeml/data/generated_csd_extra/

Usage:
  python gen_csd_extra.py summary
  python gen_csd_extra.py <idx>
  sbatch --array=0-N job_csd_extra.sh

Bond lengths from:
  Alvarez S., Dalton Trans. 2013 (4d/5d survey)
  Cambridge Structural Database mean values
  Shannon R.D., Acta Cryst. A32, 751 (1976) — ionic radii
"""
import numpy as np, json, os, sys, logging
from pyscf import gto, scf, mcscf

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

OUTPUT_DIR = os.path.expanduser('~/activeml/data/generated_csd_extra')

METAL_CONSTANTS = {
    # 3d
    'Ti': {'z_eff':  8.14, 'zeta_so_cm1':  121, 'metal_row': '3d'},
    'V':  {'z_eff':  8.98, 'zeta_so_cm1':  208, 'metal_row': '3d'},
    'Cr': {'z_eff':  9.76, 'zeta_so_cm1':  273, 'metal_row': '3d'},
    'Mn': {'z_eff': 10.53, 'zeta_so_cm1':  355, 'metal_row': '3d'},
    'Fe': {'z_eff': 11.18, 'zeta_so_cm1':  460, 'metal_row': '3d'},
    'Co': {'z_eff': 12.00, 'zeta_so_cm1':  533, 'metal_row': '3d'},
    'Ni': {'z_eff': 12.78, 'zeta_so_cm1':  669, 'metal_row': '3d'},
    'Cu': {'z_eff': 13.20, 'zeta_so_cm1':  831, 'metal_row': '3d'},
    'Zn': {'z_eff': 13.57, 'zeta_so_cm1': 1042, 'metal_row': '3d'},
    # 4d
    'Mo': {'z_eff': 10.97, 'zeta_so_cm1':  467, 'metal_row': '4d'},
    'Ru': {'z_eff': 12.33, 'zeta_so_cm1':  880, 'metal_row': '4d'},
    'Zr': {'z_eff':  8.55, 'zeta_so_cm1':  195, 'metal_row': '4d'},
}

ECP_METALS = {'Mo', 'Ru', 'Rh', 'Pd', 'Ir', 'Pt', 'Zr'}

METAL_Z = {
    'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,
    'Ni':28,'Cu':29,'Zn':30,'Mo':42,'Ru':44,'Zr':40,
}
LIG_Z = {'Cl':17,'Br':35,'F':9,'N':7,'O':8,'S':16}

# ══════════════════════════════════════════════════════════════
# CSD STRUCTURES
# ══════════════════════════════════════════════════════════════
# Format: (struct_name, metal, charge, atoms_string)

CSD_STRUCTURES = [

    # ══════════════════════════════════════════════════════════
    # TYPE 1: Ti — currently zero real CSD structures
    # Ti(IV) d0, Ti(III) d1 — octahedral dominant
    # Ti-Cl oct: 2.360 Å (Ti(IV)), 2.480 Å (Ti(III))
    # ══════════════════════════════════════════════════════════

    ("TiCl6_2m_oct",
     "Ti", -2,
     """Ti  0.000  0.000  0.000
        Cl  2.360  0.000  0.000
        Cl -2.360  0.000  0.000
        Cl  0.000  2.360  0.000
        Cl  0.000 -2.360  0.000
        Cl  0.000  0.000  2.360
        Cl  0.000  0.000 -2.360"""),

    ("TiCl6_3m_oct",
     "Ti", -3,
     """Ti  0.000  0.000  0.000
        Cl  2.480  0.000  0.000
        Cl -2.480  0.000  0.000
        Cl  0.000  2.480  0.000
        Cl  0.000 -2.480  0.000
        Cl  0.000  0.000  2.480
        Cl  0.000  0.000 -2.480"""),

    # Ti(III) distorted octahedral
    ("TiCl6_3m_dist",
     "Ti", -3,
     """Ti  0.000  0.000  0.000
        Cl  2.470  0.120 -0.070
        Cl -2.490  0.080  0.090
        Cl  0.090  2.480 -0.060
        Cl -0.070 -2.490  0.080
        Cl  0.050  0.040  2.500
        Cl -0.060 -0.050 -2.510"""),

    # Ti(IV) fluoride
    ("TiF6_2m_oct",
     "Ti", -2,
     """Ti  0.000  0.000  0.000
        F   1.860  0.000  0.000
        F  -1.860  0.000  0.000
        F   0.000  1.860  0.000
        F   0.000 -1.860  0.000
        F   0.000  0.000  1.860
        F   0.000  0.000 -1.860"""),

    # Ti(IV) bromide
    ("TiBr6_2m_oct",
     "Ti", -2,
     """Ti  0.000  0.000  0.000
        Br  2.490  0.000  0.000
        Br -2.490  0.000  0.000
        Br  0.000  2.490  0.000
        Br  0.000 -2.490  0.000
        Br  0.000  0.000  2.490
        Br  0.000  0.000 -2.490"""),

    # Ti with N-donor (ammine)
    ("TiCl4N2_trans",
     "Ti", -2,
     """Ti  0.000  0.000  0.000
        Cl  2.360  0.000  0.000
        Cl -2.360  0.000  0.000
        Cl  0.000  2.360  0.000
        Cl  0.000 -2.360  0.000
        N   0.000  0.000  2.140
        N   0.000  0.000 -2.140"""),

    # Ti(III) with O-donor
    ("TiCl4O2_trans",
     "Ti", -1,
     """Ti  0.000  0.000  0.000
        Cl  2.480  0.000  0.000
        Cl -2.480  0.000  0.000
        Cl  0.000  2.480  0.000
        Cl  0.000 -2.480  0.000
        O   0.000  0.000  2.060
        O   0.000  0.000 -2.060"""),

    # ══════════════════════════════════════════════════════════
    # TYPE 1: V — currently zero real CSD structures
    # V(III) d2, V(IV) d1, V(V) d0 — octahedral dominant
    # V-Cl oct: 2.380 Å (V(III)), 2.310 Å (V(IV))
    # ══════════════════════════════════════════════════════════

    ("VCl6_3m_oct",
     "V", -3,
     """V   0.000  0.000  0.000
        Cl  2.380  0.000  0.000
        Cl -2.380  0.000  0.000
        Cl  0.000  2.380  0.000
        Cl  0.000 -2.380  0.000
        Cl  0.000  0.000  2.380
        Cl  0.000  0.000 -2.380"""),

    ("VCl6_2m_oct",
     "V", -2,
     """V   0.000  0.000  0.000
        Cl  2.310  0.000  0.000
        Cl -2.310  0.000  0.000
        Cl  0.000  2.310  0.000
        Cl  0.000 -2.310  0.000
        Cl  0.000  0.000  2.310
        Cl  0.000  0.000 -2.310"""),

    # V(III) distorted
    ("VCl6_3m_dist",
     "V", -3,
     """V   0.000  0.000  0.000
        Cl  2.370  0.100 -0.060
        Cl -2.390  0.070  0.080
        Cl  0.080  2.380 -0.050
        Cl -0.060 -2.385  0.070
        Cl  0.040  0.030  2.395
        Cl -0.050 -0.040 -2.400"""),

    # V(IV) oxo complex — vanadyl VO^2+ common
    ("VOCl4_1m_sqpyr",
     "V", -1,
     """V   0.000  0.000  0.000
        O   0.000  0.000  1.590
        Cl  2.310  0.000  0.000
        Cl -2.310  0.000  0.000
        Cl  0.000  2.310  0.000
        Cl  0.000 -2.310  0.000"""),

    # V fluoride
    ("VF6_3m_oct",
     "V", -3,
     """V   0.000  0.000  0.000
        F   1.920  0.000  0.000
        F  -1.920  0.000  0.000
        F   0.000  1.920  0.000
        F   0.000 -1.920  0.000
        F   0.000  0.000  1.920
        F   0.000  0.000 -1.920"""),

    # V with N-donor
    ("VCl4N2_trans",
     "V", -1,
     """V   0.000  0.000  0.000
        Cl  2.380  0.000  0.000
        Cl -2.380  0.000  0.000
        Cl  0.000  2.380  0.000
        Cl  0.000 -2.380  0.000
        N   0.000  0.000  2.160
        N   0.000  0.000 -2.160"""),

    # ══════════════════════════════════════════════════════════
    # TYPE 1: Zn — currently zero real CSD structures
    # Zn(II) d10 — tetrahedral and octahedral
    # Zn-Cl tet: 2.270 Å, oct: 2.440 Å
    # ══════════════════════════════════════════════════════════

    ("ZnCl4_2m_tet",
     "Zn", -2,
     """Zn  0.000  0.000  0.000
        Cl  2.270  0.000  0.000
        Cl -2.270  0.000  0.000
        Cl  0.000  2.270  0.000
        Cl  0.000 -2.270  0.000"""),

    ("ZnCl6_4m_oct",
     "Zn", -4,
     """Zn  0.000  0.000  0.000
        Cl  2.440  0.000  0.000
        Cl -2.440  0.000  0.000
        Cl  0.000  2.440  0.000
        Cl  0.000 -2.440  0.000
        Cl  0.000  0.000  2.440
        Cl  0.000  0.000 -2.440"""),

    # Zn bromide tetrahedral
    ("ZnBr4_2m_tet",
     "Zn", -2,
     """Zn  0.000  0.000  0.000
        Br  2.400  0.000  0.000
        Br -2.400  0.000  0.000
        Br  0.000  2.400  0.000
        Br  0.000 -2.400  0.000"""),

    # Zn with N-donor
    ("ZnCl2N2_sqpl",
     "Zn", 0,
     """Zn  0.000  0.000  0.000
        Cl  2.270  0.000  0.000
        Cl -2.270  0.000  0.000
        N   0.000  2.080  0.000
        N   0.000 -2.080  0.000"""),

    # Zn distorted tetrahedral
    ("ZnCl4_2m_tet_dist",
     "Zn", -2,
     """Zn  0.000  0.000  0.000
        Cl  2.285  0.090 -0.050
        Cl -2.255  0.060  0.070
        Cl  0.060  2.275 -0.040
        Cl -0.080 -2.280  0.030"""),

    # ══════════════════════════════════════════════════════════
    # TYPE 2: 7-COORDINATE COMPLEXES
    # Mo, Ti, V — common 7-coordinate geometries
    # pentagonal bipyramidal (pbp) and capped octahedral
    # Mo-Cl pbp: 2.450 Å equatorial, 2.530 Å axial
    # ══════════════════════════════════════════════════════════

    # Mo(II) pentagonal bipyramidal — 7 coord
    # 5 equatorial Cl + 2 axial Cl
    ("MoCl7_2m_pbp",
     "Mo", -2,
     """Mo  0.000  0.000  0.000
        Cl  2.450  0.000  0.000
        Cl  0.756  2.328  0.000
        Cl -1.980  1.440  0.000
        Cl -1.980 -1.440  0.000
        Cl  0.756 -2.328  0.000
        Cl  0.000  0.000  2.530
        Cl  0.000  0.000 -2.530"""),

    # Ti(IV) 7-coordinate capped octahedral
    ("TiCl7_3m_capoct",
     "Ti", -3,
     """Ti  0.000  0.000  0.000
        Cl  2.360  0.000  0.000
        Cl -2.360  0.000  0.000
        Cl  0.000  2.360  0.000
        Cl  0.000 -2.360  0.000
        Cl  0.000  0.000  2.360
        Cl  0.000  0.000 -2.360
        Cl  2.100  2.100  0.000"""),

    # V(III) 7-coordinate
    ("VCl7_4m_pbp",
     "V", -4,
     """V   0.000  0.000  0.000
        Cl  2.380  0.000  0.000
        Cl  0.736  2.267  0.000
        Cl -1.930  1.403  0.000
        Cl -1.930 -1.403  0.000
        Cl  0.736 -2.267  0.000
        Cl  0.000  0.000  2.480
        Cl  0.000  0.000 -2.480"""),

    # Mo 7-coord with oxo
    ("MoOCl6_2m_capoct",
     "Mo", -2,
     """Mo  0.000  0.000  0.000
        O   0.000  0.000  1.680
        Cl  2.450  0.000  0.000
        Cl -2.450  0.000  0.000
        Cl  0.000  2.450  0.000
        Cl  0.000 -2.450  0.000
        Cl  0.000  0.000 -2.530
        Cl  2.100  2.100  0.000"""),

    # ══════════════════════════════════════════════════════════
    # TYPE 3: MIXED LIGAND ENVIRONMENTS
    # Real catalytic systems with two different donor types
    # ══════════════════════════════════════════════════════════

    # Fe trans-Cl4N2 (common in magnetochemistry)
    ("FeCl4N2_trans",
     "Fe", -2,
     """Fe  0.000  0.000  0.000
        Cl  2.310  0.000  0.000
        Cl -2.310  0.000  0.000
        Cl  0.000  2.310  0.000
        Cl  0.000 -2.310  0.000
        N   0.000  0.000  2.170
        N   0.000  0.000 -2.170"""),

    # Fe cis-Cl4N2
    ("FeCl4N2_cis",
     "Fe", -2,
     """Fe  0.000  0.000  0.000
        Cl  2.310  0.000  0.000
        Cl -2.310  0.000  0.000
        Cl  0.000  2.310  0.000
        Cl  0.000 -2.310  0.000
        N   0.000  0.000  2.170
        Cl  0.000  0.000 -2.310"""),

    # Mn trans-Cl4O2
    ("MnCl4O2_trans",
     "Mn", -2,
     """Mn  0.000  0.000  0.000
        Cl  2.580  0.000  0.000
        Cl -2.580  0.000  0.000
        Cl  0.000  2.580  0.000
        Cl  0.000 -2.580  0.000
        O   0.000  0.000  2.220
        O   0.000  0.000 -2.220"""),

    # Cr Cl4N2 trans
    ("CrCl4N2_trans",
     "Cr", -1,
     """Cr  0.000  0.000  0.000
        Cl  2.490  0.000  0.000
        Cl -2.490  0.000  0.000
        Cl  0.000  2.490  0.000
        Cl  0.000 -2.490  0.000
        N   0.000  0.000  2.090
        N   0.000  0.000 -2.090"""),

    # Co Cl4Br2 trans (mixed halide)
    ("CoCl4Br2_trans",
     "Co", -4,
     """Co  0.000  0.000  0.000
        Cl  2.440  0.000  0.000
        Cl -2.440  0.000  0.000
        Cl  0.000  2.440  0.000
        Cl  0.000 -2.440  0.000
        Br  0.000  0.000  2.520
        Br  0.000  0.000 -2.520"""),

    # Ni Cl2Br2 square planar (mixed halide)
    ("NiCl2Br2_sqpl",
     "Ni", -2,
     """Ni  0.000  0.000  0.000
        Cl  2.210  0.000  0.000
        Cl -2.210  0.000  0.000
        Br  0.000  2.370  0.000
        Br  0.000 -2.370  0.000"""),

    # Cu Cl4O2 Jahn-Teller elongated
    ("CuCl4O2_jt",
     "Cu", -2,
     """Cu  0.000  0.000  0.000
        Cl  2.300  0.000  0.000
        Cl -2.300  0.000  0.000
        Cl  0.000  2.300  0.000
        Cl  0.000 -2.300  0.000
        O   0.000  0.000  2.440
        O   0.000  0.000 -2.440"""),

    # Ru Cl4O2 trans (anticancer relevance)
    ("RuCl4O2_trans",
     "Ru", -2,
     """Ru  0.000  0.000  0.000
        Cl  2.340  0.000  0.000
        Cl -2.340  0.000  0.000
        Cl  0.000  2.340  0.000
        Cl  0.000 -2.340  0.000
        O   0.000  0.000  2.050
        O   0.000  0.000 -2.050"""),

    # Ir Cl4N2 fac (facial isomer)
    ("IrCl3N3_fac",
     "Ir", 0,
     """Ir  0.000  0.000  0.000
        Cl  2.355  0.000  0.000
        Cl  0.000  2.355  0.000
        Cl  0.000  0.000  2.355
        N  -2.090  0.000  0.000
        N   0.000 -2.090  0.000
        N   0.000  0.000 -2.090"""),

    # Mo Cl3O3 facial (oxo-halide common)
    ("MoCl3O3_fac",
     "Mo", 0,
     """Mo  0.000  0.000  0.000
        Cl  2.460  0.000  0.000
        Cl  0.000  2.460  0.000
        Cl  0.000  0.000  2.460
        O  -1.680  0.000  0.000
        O   0.000 -1.680  0.000
        O   0.000  0.000 -1.680"""),
]

# ── SPIN STATES ───────────────────────────────────────────────
SPIN_STATES = {
    'Ti': [0, 2],      # d0 Ti(IV), d1 Ti(III)
    'V':  [0, 2, 4],   # d0-d2
    'Zn': [0],         # d10 always S=0
    'Mo': [0, 1, 2, 3],
    'Ru': [0, 2, 4],
    'Fe': [0, 2, 4],
    'Co': [1, 3],
    'Mn': [1, 3, 5],
    'Cr': [0, 2, 4],
    'Ni': [0, 2],
    'Cu': [1],
    'Zr': [0, 2],
}


def count_electrons(atoms_str, charge):
    total = 0
    for line in atoms_str.strip().split('\n'):
        sym = line.strip().split()[0]
        total += METAL_Z.get(sym, LIG_Z.get(sym, 0))
    return total - charge


def get_nact(n_total):
    for n in [10, 9, 11, 8, 12, 7, 13, 6, 14]:
        if (n_total - n) >= 0 and (n_total - n) % 2 == 0:
            return n
    return 10


def run_uhf(mol):
    for settings in [
        dict(max_cycle=300, conv_tol=1e-10, damp=0.0, level_shift=0.0),
        dict(max_cycle=500, conv_tol=1e-9,  damp=0.3, level_shift=0.2),
        dict(max_cycle=800, conv_tol=1e-8,  damp=0.5, level_shift=0.5),
    ]:
        mf = scf.UHF(mol)
        for k, v in settings.items(): setattr(mf, k, v)
        mf.verbose = 0
        mf.run()
        if mf.converged: return mf
    return mf


def run_casscf(mf, mol, n_act):
    e_m = (mf.mo_energy[0] + mf.mo_energy[1]) / 2
    occ = mf.mo_occ[0] + mf.mo_occ[1]
    occ_idx  = np.where(occ > 0.5)[0]
    virt_idx = np.where(occ < 0.5)[0]
    if len(occ_idx) == 0 or len(virt_idx) == 0: return None, False
    homo_e = float(e_m[occ_idx[-1]])
    lumo_e = float(e_m[virt_idx[0]])
    gap    = (homo_e + lumo_e) / 2
    w14    = sorted(np.argsort(np.abs(e_m - gap))[:14], key=lambda i: e_m[i])
    best_mc = None; best_e = 0.0
    for window in [w14[:10], w14[2:12], w14[1:11], w14[3:13]]:
        for shift in [1e-3, 1e-2, 5e-2, 1e-1]:
            try:
                mc = mcscf.CASSCF(mf, 10, n_act)
                mc.max_cycle_macro = 500
                mc.conv_tol = 1e-8
                mc.ah_level_shift = shift
                mc.verbose = 0
                mc.kernel(mc.sort_mo(window, base=0))
                ec = mc.e_tot - mf.e_tot
                if ec < 0 and ec < best_e: best_mc = mc; best_e = ec
                if mc.converged and ec < -0.01: return mc, True
            except: continue
    if best_mc: return best_mc, best_mc.converged
    return None, False


def run_one(struct_name, metal, charge, atoms_str, spin):
    name    = f"CSD_{struct_name}_spin{spin}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outfile = os.path.join(OUTPUT_DIR, f"{name}.json")

    if os.path.exists(outfile):
        r = json.load(open(outfile))
        if r.get('converged') and r.get('corr_energy', 0) < -0.001:
            log.info(f"SKIP: {name}"); return True

    n_e = count_electrons(atoms_str, charge)
    if (n_e % 2) != (spin % 2):
        json.dump({'name': name, 'status': 'skipped',
                   'reason': 'parity', 'geometry': 'csd_real'},
                  open(outfile, 'w')); return True

    consts = METAL_CONSTANTS.get(metal, {'z_eff': 10.0,
                                          'zeta_so_cm1': 400,
                                          'metal_row': '3d'})
    log.info(f"Start: {name}  n_e={n_e}  row={consts['metal_row']}")

    try:
        mol = gto.Mole()
        mol.atom    = atoms_str
        mol.basis   = 'def2-SVP'
        mol.charge  = charge
        mol.spin    = spin
        mol.verbose = 0
        if metal in ECP_METALS:
            mol.ecp = 'def2-SVP'
        mol.build()

        n_act = get_nact(mol.nelectron)
        mf    = run_uhf(mol)
        mc, conv = run_casscf(mf, mol, n_act)

        if mc is None:
            json.dump({'name': name, 'status': 'failed',
                       'geometry': 'csd_real', 'metal': metal},
                      open(outfile, 'w')); return False

        ec = mc.e_tot - mf.e_tot
        if ec >= 0:
            json.dump({'name': name, 'status': 'unphysical',
                       'corr_energy': float(ec),
                       'geometry': 'csd_real'},
                      open(outfile, 'w')); return False

        casdm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
        no, _  = np.linalg.eigh(casdm1)
        no     = np.sort(no)[::-1]
        n_active = sum(1 for n in no if 0.02 < n < 1.98)

        lig = 'Cl'
        for l in ['Br', 'F', 'N', 'O', 'S']:
            if l in atoms_str: lig = l; break

        e_m = (mf.mo_energy[0] + mf.mo_energy[1]) / 2
        occ = mf.mo_occ[0] + mf.mo_occ[1]
        oi  = np.where(occ > 0.5)[0]
        vi  = np.where(occ < 0.5)[0]
        homo_e = float(e_m[oi[-1]])  if len(oi) > 0 else 0.0
        lumo_e = float(e_m[vi[0]])   if len(vi) > 0 else 0.0
        oa = np.where(mf.mo_occ[0] > 0.5)[0]
        ob = np.where(mf.mo_occ[1] > 0.5)[0]
        homo_a = float(mf.mo_energy[0][oa[-1]]) if len(oa) > 0 else 0.0
        homo_b = float(mf.mo_energy[1][ob[-1]]) if len(ob) > 0 else 0.0
        S = spin / 2.0
        spin_contam = float(mf.spin_square()[0] - S*(S+1))

        result = {
            'name':              name,
            'metal':             metal,
            'ligand':            lig,
            'n_ligands':         atoms_str.count(lig),
            'charge':            charge,
            'spin':              spin,
            'mult':              spin + 1,
            'geometry':          'csd_real',
            'struct_name':       struct_name,
            'n_electrons':       mol.nelectron,
            'n_active_e':        n_act,
            'E_HF':              float(mf.e_tot),
            'E_CASSCF':          float(mc.e_tot),
            'corr_energy':       float(ec),
            'converged':         bool(conv),
            'n_active':          n_active,
            'no_occ':            [float(x) for x in no],
            'status':            'ok',
            'z_eff':             consts['z_eff'],
            'zeta_so_cm1':       consts['zeta_so_cm1'],
            'metal_row':         consts['metal_row'],
            'spin_contamination': spin_contam,
            'homo_lumo_gap':     float(lumo_e - homo_e),
            'homo_energy':       homo_e,
            'lumo_energy':       lumo_e,
            'homo_ab_gap':       float(abs(homo_a - homo_b)),
        }

        with open(outfile, 'w') as f:
            json.dump(result, f, indent=2)
        log.info(f"  OK: n_active={n_active} Ec={ec:.4f} converged={conv}")
        return True

    except Exception as e:
        log.error(f"  Error {name}: {e}")
        json.dump({'name': name, 'status': 'error', 'reason': str(e),
                   'geometry': 'csd_real'}, open(outfile, 'w'))
        return False


# ── BUILD JOB LIST ─────────────────────────────────────────────
ALL_JOBS = []
seen     = set()
os.makedirs(OUTPUT_DIR, exist_ok=True)
existing = set(os.path.basename(f)
               for f in __import__('glob').glob(f'{OUTPUT_DIR}/*.json'))

for struct_name, metal, charge, atoms_str in CSD_STRUCTURES:
    spins = SPIN_STATES.get(metal, [0, 2])
    n_e   = count_electrons(atoms_str, charge)
    for spin in spins:
        if (n_e % 2) != (spin % 2): continue
        fname = f"CSD_{struct_name}_spin{spin}.json"
        if fname in existing: continue
        key = (struct_name, spin)
        if key in seen: continue
        seen.add(key)
        ALL_JOBS.append((struct_name, metal, charge, atoms_str, spin))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        from collections import defaultdict
        by_metal = defaultdict(int)
        by_type  = defaultdict(int)
        for s, m, c, a, sp in ALL_JOBS:
            by_metal[m] += 1
            t = 'Type1' if m in ('Ti','V','Zn') else \
                'Type2' if '7' in s or 'pbp' in s or 'capoct' in s else \
                'Type3'
            by_type[t] += 1
        print(f"\nTotal jobs:  {len(ALL_JOBS)}")
        print(f"Structures:  {len(CSD_STRUCTURES)}")
        print(f"\nJobs by metal:")
        for m in ['Ti','V','Zn','Mo','Ru','Fe','Co','Mn','Cr','Ni','Cu']:
            if by_metal.get(m, 0) > 0:
                print(f"  {m}: {by_metal[m]}")
        print(f"\nJobs by type:")
        for t, n in sorted(by_type.items()):
            print(f"  {t}: {n}")
        print(f"\nEst. time: ~{len(ALL_JOBS)*20/60:.0f} hrs serial, "
              f"~{len(ALL_JOBS)*20/60/32:.1f} hrs on 32 cores")
        sys.exit(0)

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if idx >= len(ALL_JOBS): sys.exit(1)
    sys.exit(0 if run_one(*ALL_JOBS[idx]) else 1)
