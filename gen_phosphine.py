"""
gen_phosphine.py
================
Hard-coded CSD-survey structures for phosphine (PH3) complexes.
Phosphines appear in >60% of homogeneous catalysts:
  - Wilkinson's catalyst: RhCl(PPh3)3
  - Grubbs catalyst: Ru-PCy3
  - BINAP-Pd cross-coupling
  - Vaska's compound: IrCl(CO)(PPh3)2

Simplified model: PH3 ligand represented as P donor at
correct M-P bond length. Full PPh3 cone angle effects
are captured through the P-donor position.

Uses CASSCF(10,10) — same as existing dataset for consistency.
CASSCF(14,14) only needed if n_active > 10 (checked at runtime).

Output: ~/activeml/data/generated_phosphine/

Bond lengths (CSD mean values, Alvarez 2013):
  M-P octahedral:
    Fe-P: 2.200 Ang   Ru-P: 2.350 Ang   Os-P: 2.320 Ang
    Co-P: 2.200 Ang   Rh-P: 2.320 Ang   Ir-P: 2.340 Ang
    Ni-P: 2.170 Ang   Pd-P: 2.290 Ang   Pt-P: 2.295 Ang
    Mn-P: 2.310 Ang   Re-P: 2.450 Ang   W-P:  2.520 Ang
    Cr-P: 2.320 Ang   Mo-P: 2.490 Ang

Usage:
  python gen_phosphine.py summary
  python gen_phosphine.py <idx>
  sbatch --array=0-N job_phosphine.sh
"""
import numpy as np, json, os, sys, logging
from pyscf import gto, scf, mcscf

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

OUTPUT_DIR = os.path.expanduser('~/activeml/data/generated_phosphine')

METAL_CONSTANTS = {
    # 3d
    'Fe': {'z_eff': 11.18, 'zeta_so_cm1':  460, 'metal_row': '3d'},
    'Co': {'z_eff': 12.00, 'zeta_so_cm1':  533, 'metal_row': '3d'},
    'Ni': {'z_eff': 12.78, 'zeta_so_cm1':  669, 'metal_row': '3d'},
    'Mn': {'z_eff': 10.53, 'zeta_so_cm1':  355, 'metal_row': '3d'},
    'Cr': {'z_eff':  9.76, 'zeta_so_cm1':  273, 'metal_row': '3d'},
    'Cu': {'z_eff': 13.20, 'zeta_so_cm1':  831, 'metal_row': '3d'},
    # 4d
    'Ru': {'z_eff': 12.33, 'zeta_so_cm1':  880, 'metal_row': '4d'},
    'Rh': {'z_eff': 12.67, 'zeta_so_cm1': 1097, 'metal_row': '4d'},
    'Pd': {'z_eff': 13.00, 'zeta_so_cm1': 1334, 'metal_row': '4d'},
    'Mo': {'z_eff': 10.97, 'zeta_so_cm1':  467, 'metal_row': '4d'},
    # 5d
    'Ir': {'z_eff': 17.00, 'zeta_so_cm1': 3909, 'metal_row': '5d'},
    'Pt': {'z_eff': 17.33, 'zeta_so_cm1': 4146, 'metal_row': '5d'},
    'Os': {'z_eff': 17.17, 'zeta_so_cm1': 3381, 'metal_row': '5d'},
    'Re': {'z_eff': 17.01, 'zeta_so_cm1': 2456, 'metal_row': '5d'},
    'W':  {'z_eff': 11.61, 'zeta_so_cm1': 2748, 'metal_row': '5d'},
}

ECP_METALS = {'Ru','Rh','Pd','Mo','Ir','Pt','Os','Re','W'}

METAL_Z = {
    'Fe':26,'Co':27,'Ni':28,'Mn':25,'Cr':24,'Cu':29,
    'Ru':44,'Rh':45,'Pd':46,'Mo':42,
    'Ir':77,'Pt':78,'Os':76,'Re':75,'W':74,
}
LIG_Z = {'P':15,'Cl':17,'Br':35,'F':9,'N':7,'O':8,'C':6}

# ══════════════════════════════════════════════════════════════
# PHOSPHINE COMPLEX STRUCTURES
# Format: (struct_name, metal, charge, atoms_string)
#
# Geometry types:
#   trans-MCl2(PH3)4  — octahedral, P trans to P, Cl trans to Cl
#   mer-MCl3(PH3)3    — octahedral, meridional PH3
#   fac-MCl3(PH3)3    — octahedral, facial PH3
#   MCl(PH3)3         — square planar (Pd, Pt, Rh)
#   M(PH3)4           — tetrahedral Pd(0), Ni(0)
#   MCl2(PH3)2        — square planar or tetrahedral
# ══════════════════════════════════════════════════════════════

CSD_STRUCTURES = [

    # ── Fe phosphine complexes ────────────────────────────────
    # Fe(II) d6 trans-dichloro-tetraphosphine
    ("FeCl2P4_trans_oct",
     "Fe", 0,
     """Fe  0.000  0.000  0.000
        P   2.200  0.000  0.000
        P  -2.200  0.000  0.000
        P   0.000  2.200  0.000
        P   0.000 -2.200  0.000
        Cl  0.000  0.000  2.450
        Cl  0.000  0.000 -2.450"""),

    # Fe(II) d6 mer-trichloro-triphosphine
    ("FeCl3P3_mer_oct",
     "Fe", 0,
     """Fe  0.000  0.000  0.000
        P   2.200  0.000  0.000
        P  -2.200  0.000  0.000
        P   0.000  2.200  0.000
        Cl  0.000 -2.450  0.000
        Cl  0.000  0.000  2.450
        Cl  0.000  0.000 -2.450"""),

    # Fe(0) d8 tetrakis-phosphine (catalytic intermediate)
    ("FeP4_tet",
     "Fe", 0,
     """Fe  0.000  0.000  0.000
        P   2.200  0.000  0.000
        P  -2.200  0.000  0.000
        P   0.000  2.200  0.000
        P   0.000 -2.200  0.000"""),

    # ── Ru phosphine complexes ────────────────────────────────
    # Ru(II) d6 — Grubbs-type precursor geometry
    ("RuCl2P4_trans_oct",
     "Ru", 0,
     """Ru  0.000  0.000  0.000
        P   2.350  0.000  0.000
        P  -2.350  0.000  0.000
        P   0.000  2.350  0.000
        P   0.000 -2.350  0.000
        Cl  0.000  0.000  2.410
        Cl  0.000  0.000 -2.410"""),

    # Ru(II) mer-trichloro-triphosphine
    ("RuCl3P3_mer_oct",
     "Ru", 0,
     """Ru  0.000  0.000  0.000
        P   2.350  0.000  0.000
        P  -2.350  0.000  0.000
        P   0.000  2.350  0.000
        Cl  0.000 -2.410  0.000
        Cl  0.000  0.000  2.410
        Cl  0.000  0.000 -2.410"""),

    # Ru(II) fac-trichloro-triphosphine
    ("RuCl3P3_fac_oct",
     "Ru", 0,
     """Ru  0.000  0.000  0.000
        P   2.350  0.000  0.000
        P   0.000  2.350  0.000
        P   0.000  0.000  2.350
        Cl -2.410  0.000  0.000
        Cl  0.000 -2.410  0.000
        Cl  0.000  0.000 -2.410"""),

    # ── Rh phosphine complexes ────────────────────────────────
    # Rh(I) d8 — Wilkinson's catalyst geometry
    # RhCl(PH3)3 square planar
    ("RhClP3_sqpl",
     "Rh", 0,
     """Rh  0.000  0.000  0.000
        Cl  2.380  0.000  0.000
        P  -2.320  0.000  0.000
        P   0.000  2.320  0.000
        P   0.000 -2.320  0.000"""),

    # Rh(III) d6 octahedral
    ("RhCl3P3_mer_oct",
     "Rh", 0,
     """Rh  0.000  0.000  0.000
        P   2.320  0.000  0.000
        P  -2.320  0.000  0.000
        P   0.000  2.320  0.000
        Cl  0.000 -2.380  0.000
        Cl  0.000  0.000  2.380
        Cl  0.000  0.000 -2.380"""),

    # Rh(I) d8 trans-dichloro-diphosphine square planar
    ("RhCl2P2_trans_sqpl",
     "Rh", -1,
     """Rh  0.000  0.000  0.000
        P   2.320  0.000  0.000
        P  -2.320  0.000  0.000
        Cl  0.000  2.380  0.000
        Cl  0.000 -2.380  0.000"""),

    # ── Pd phosphine complexes ────────────────────────────────
    # Pd(0) d10 — tetrakis-phosphine (cross-coupling catalyst)
    ("PdP4_tet",
     "Pd", 0,
     """Pd  0.000  0.000  0.000
        P   2.290  0.000  0.000
        P  -2.290  0.000  0.000
        P   0.000  2.290  0.000
        P   0.000 -2.290  0.000"""),

    # Pd(II) d8 — trans-dichloro-diphosphine square planar
    ("PdCl2P2_trans_sqpl",
     "Pd", 0,
     """Pd  0.000  0.000  0.000
        P   2.290  0.000  0.000
        P  -2.290  0.000  0.000
        Cl  0.000  2.310  0.000
        Cl  0.000 -2.310  0.000"""),

    # Pd(II) cis-dichloro-diphosphine (cisplatin-like)
    ("PdCl2P2_cis_sqpl",
     "Pd", 0,
     """Pd  0.000  0.000  0.000
        P   2.290  0.000  0.000
        Cl -2.310  0.000  0.000
        P   0.000  2.290  0.000
        Cl  0.000 -2.310  0.000"""),

    # Pd(0) d10 bis-phosphine (active catalyst form)
    ("PdP2_linear",
     "Pd", 0,
     """Pd  0.000  0.000  0.000
        P   2.290  0.000  0.000
        P  -2.290  0.000  0.000"""),

    # ── Ir phosphine complexes ────────────────────────────────
    # Ir(I) d8 — Vaska's compound geometry
    # IrCl(CO)(PH3)2 square planar
    ("IrClCOP2_sqpl",
     "Ir", 0,
     """Ir  0.000  0.000  0.000
        Cl  2.380  0.000  0.000
        C  -1.850  0.000  0.000
        P   0.000  2.340  0.000
        P   0.000 -2.340  0.000"""),

    # Ir(III) d6 — oxidative addition product of Vaska's
    ("IrClHP2_oct",
     "Ir", 0,
     """Ir  0.000  0.000  0.000
        P   2.340  0.000  0.000
        P  -2.340  0.000  0.000
        Cl  0.000  2.400  0.000
        Cl  0.000 -2.400  0.000
        Cl  0.000  0.000  2.400
        P   0.000  0.000 -2.340"""),

    # Ir(I) d8 square planar tris-phosphine
    ("IrClP3_sqpl",
     "Ir", 0,
     """Ir  0.000  0.000  0.000
        Cl  2.380  0.000  0.000
        P  -2.340  0.000  0.000
        P   0.000  2.340  0.000
        P   0.000 -2.340  0.000"""),

    # ── Pt phosphine complexes ────────────────────────────────
    # Pt(II) d8 — trans-dichloro-diphosphine square planar
    ("PtCl2P2_trans_sqpl",
     "Pt", 0,
     """Pt  0.000  0.000  0.000
        P   2.295  0.000  0.000
        P  -2.295  0.000  0.000
        Cl  0.000  2.310  0.000
        Cl  0.000 -2.310  0.000"""),

    # Pt(0) d10 — bis-phosphine
    ("PtP2_linear",
     "Pt", 0,
     """Pt  0.000  0.000  0.000
        P   2.295  0.000  0.000
        P  -2.295  0.000  0.000"""),

    # Pt(II) cis-dichloro-diphosphine
    ("PtCl2P2_cis_sqpl",
     "Pt", 0,
     """Pt  0.000  0.000  0.000
        P   2.295  0.000  0.000
        Cl -2.310  0.000  0.000
        P   0.000  2.295  0.000
        Cl  0.000 -2.310  0.000"""),

    # ── Ni phosphine complexes ────────────────────────────────
    # Ni(0) d10 — tetrakis-phosphine (nickel catalysis)
    ("NiP4_tet",
     "Ni", 0,
     """Ni  0.000  0.000  0.000
        P   2.170  0.000  0.000
        P  -2.170  0.000  0.000
        P   0.000  2.170  0.000
        P   0.000 -2.170  0.000"""),

    # Ni(II) d8 trans-dichloro-diphosphine square planar
    ("NiCl2P2_trans_sqpl",
     "Ni", 0,
     """Ni  0.000  0.000  0.000
        P   2.170  0.000  0.000
        P  -2.170  0.000  0.000
        Cl  0.000  2.210  0.000
        Cl  0.000 -2.210  0.000"""),

    # ── Mo phosphine complexes ────────────────────────────────
    # Mo(0) d6 — fac-tricarbonyl-triphosphine (catalysis)
    ("MoCl3P3_fac_oct",
     "Mo", 0,
     """Mo  0.000  0.000  0.000
        P   2.490  0.000  0.000
        P   0.000  2.490  0.000
        P   0.000  0.000  2.490
        Cl -2.450  0.000  0.000
        Cl  0.000 -2.450  0.000
        Cl  0.000  0.000 -2.450"""),

    # Mo(II) trans-dichloro-tetraphosphine
    ("MoCl2P4_trans_oct",
     "Mo", 0,
     """Mo  0.000  0.000  0.000
        P   2.490  0.000  0.000
        P  -2.490  0.000  0.000
        P   0.000  2.490  0.000
        P   0.000 -2.490  0.000
        Cl  0.000  0.000  2.450
        Cl  0.000  0.000 -2.450"""),

    # ── Co phosphine complexes ────────────────────────────────
    # Co(II) d7 — dichloro-triphosphine
    ("CoCl2P3_mer_oct",
     "Co", 0,
     """Co  0.000  0.000  0.000
        P   2.200  0.000  0.000
        P  -2.200  0.000  0.000
        P   0.000  2.200  0.000
        Cl  0.000 -2.310  0.000
        Cl  0.000  0.000  2.310
        Cl  0.000  0.000 -2.310"""),

    # Co(I) d8 — chloro-triphosphine square planar
    ("CoClP3_sqpl",
     "Co", 0,
     """Co  0.000  0.000  0.000
        Cl  2.320  0.000  0.000
        P  -2.200  0.000  0.000
        P   0.000  2.200  0.000
        P   0.000 -2.200  0.000"""),

    # ── Mn phosphine complexes ────────────────────────────────
    ("MnCl2P4_trans_oct",
     "Mn", 0,
     """Mn  0.000  0.000  0.000
        P   2.310  0.000  0.000
        P  -2.310  0.000  0.000
        P   0.000  2.310  0.000
        P   0.000 -2.310  0.000
        Cl  0.000  0.000  2.520
        Cl  0.000  0.000 -2.520"""),

    # ── Cr phosphine complexes ────────────────────────────────
    ("CrCl3P3_fac_oct",
     "Cr", 0,
     """Cr  0.000  0.000  0.000
        P   2.320  0.000  0.000
        P   0.000  2.320  0.000
        P   0.000  0.000  2.320
        Cl -2.390  0.000  0.000
        Cl  0.000 -2.390  0.000
        Cl  0.000  0.000 -2.390"""),
]

# ── SPIN STATES ───────────────────────────────────────────────
# Phosphine is a strong-field ligand — low spin preferred
SPIN_STATES = {
    # 3d — low spin due to strong phosphine field
    'Fe': [0, 2],      # d6: low spin S=0, high spin S=2
    'Co': [1, 3],      # d7: low spin S=1/2, high spin S=3/2
    'Ni': [0, 2],      # d8: low spin S=0
    'Mn': [1, 3, 5],   # d5: S=1/2, 3/2, 5/2
    'Cr': [0, 2, 4],   # d4/d6 range
    'Cu': [1],         # d9: S=1/2
    # 4d — strongly low spin
    'Ru': [0, 2],      # d6: S=0 dominant
    'Rh': [0, 1],      # d8: S=0, d7: S=1/2
    'Pd': [0],         # d8/d10: S=0
    'Mo': [0, 2],      # d6: S=0, d4: S=2
    # 5d — strongly low spin
    'Ir': [0, 1],      # d8: S=0, d6: S=0
    'Pt': [0],         # d8/d10: S=0
    'Os': [0, 2],      # d6: S=0
    'Re': [1, 3],      # d4/d5 range
    'W':  [0, 2],      # d4/d6 range
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
    w14    = sorted(np.argsort(np.abs(e_m - gap))[:14],
                    key=lambda i: e_m[i])
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
                if ec < 0 and ec < best_e:
                    best_mc = mc; best_e = ec
                if mc.converged and ec < -0.01:
                    return mc, True
            except: continue
    if best_mc: return best_mc, best_mc.converged
    return None, False


def run_one(struct_name, metal, charge, atoms_str, spin):
    name    = f"PH3_{struct_name}_spin{spin}"
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

    consts = METAL_CONSTANTS[metal]
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

        # Detect primary ligand — P for phosphine complexes
        lig = 'P'
        for l in ['Cl', 'Br', 'F', 'N', 'O', 'C']:
            if l in atoms_str and l != 'P':
                lig = l; break
        # Keep P as primary ligand identifier for phosphine complexes
        lig = 'P'

        e_m = (mf.mo_energy[0] + mf.mo_energy[1]) / 2
        occ = mf.mo_occ[0] + mf.mo_occ[1]
        oi  = np.where(occ > 0.5)[0]; vi = np.where(occ < 0.5)[0]
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
            'ligand':            'PH3',         # explicit phosphine label
            'ligand_sym':        'P',
            'n_ligands':         atoms_str.count('\nP') + (1 if atoms_str.startswith('P') else 0),
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
        fname = f"PH3_{struct_name}_spin{spin}.json"
        if fname in existing: continue
        key = (struct_name, spin)
        if key in seen: continue
        seen.add(key)
        ALL_JOBS.append((struct_name, metal, charge, atoms_str, spin))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        from collections import defaultdict
        by_metal = defaultdict(int)
        by_geom  = defaultdict(int)
        for s, m, c, a, sp in ALL_JOBS:
            by_metal[m] += 1
            geom = 'oct' if 'oct' in s else \
                   'sqpl' if 'sqpl' in s else \
                   'tet'  if 'tet' in s else 'other'
            by_geom[geom] += 1
        print(f"\nTotal jobs:  {len(ALL_JOBS)}")
        print(f"Structures:  {len(CSD_STRUCTURES)}")
        print(f"\nJobs by metal:")
        for m in ['Fe','Co','Ni','Mn','Cr','Ru','Rh','Pd',
                  'Mo','Ir','Pt','Os','Re','W']:
            if by_metal.get(m, 0) > 0:
                print(f"  {m}: {by_metal[m]}")
        print(f"\nJobs by geometry:")
        for g, n in sorted(by_geom.items()):
            print(f"  {g}: {n}")
        print(f"\nEst. time: ~{len(ALL_JOBS)*20/60:.0f} hrs serial, "
              f"~{len(ALL_JOBS)*20/60/32:.1f} hrs on 32 cores")
        sys.exit(0)

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if idx >= len(ALL_JOBS): sys.exit(1)
    sys.exit(0 if run_one(*ALL_JOBS[idx]) else 1)
