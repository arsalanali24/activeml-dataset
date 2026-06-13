"""
gen_bidentate.py
================
Paper 2 dataset: bidentate and tridentate ligand complexes.
Uses CASSCF(14,14) for accuracy — larger active space captures
ligand-based orbitals that monodentate CASSCF(10,10) misses.

Bidentate ligands (2 donor atoms, chelate ring):
  bipy   — 2,2'-bipyridine, N-N donor, bite ~78°
  en     — ethylenediamine, N-N donor, bite ~85°
  ox     — oxalate, O-O donor, bite ~80°
  acac   — acetylacetonate, O-O donor, bite ~90°
  phen   — 1,10-phenanthroline, N-N donor, bite ~79°
  dtc    — dithiocarbamate, S-S donor, bite ~73°

Tridentate:
  tpy    — terpyridine, N-N-N donor (meridional)
  PNP    — pincer P-N-P donor (meridional)

Geometry: simplified chelate — two donor atoms at correct
bite angle and M-donor distance. No organic scaffold.
This is physically meaningful for CASSCF because the
donor atom positions determine orbital splitting.

Output: ~/activeml/data/generated_bidentate/

n_active_e and casscf_size = 14 (stored in JSON)
This distinguishes these structures from (10,10) systems
and allows ML to learn when larger active spaces are needed.

Usage:
  python gen_bidentate.py summary
  python gen_bidentate.py <idx>
  sbatch --array=0-N job_bidentate.sh
"""
import numpy as np, json, os, sys, logging, math
from pyscf import gto, scf, mcscf

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

OUTPUT_DIR = os.path.expanduser('~/activeml/data/generated_bidentate')

METAL_CONSTANTS = {
    'Ti': {'z_eff':  8.14, 'zeta_so_cm1':  121, 'metal_row': '3d'},
    'V':  {'z_eff':  8.98, 'zeta_so_cm1':  208, 'metal_row': '3d'},
    'Cr': {'z_eff':  9.76, 'zeta_so_cm1':  273, 'metal_row': '3d'},
    'Mn': {'z_eff': 10.53, 'zeta_so_cm1':  355, 'metal_row': '3d'},
    'Fe': {'z_eff': 11.18, 'zeta_so_cm1':  460, 'metal_row': '3d'},
    'Co': {'z_eff': 12.00, 'zeta_so_cm1':  533, 'metal_row': '3d'},
    'Ni': {'z_eff': 12.78, 'zeta_so_cm1':  669, 'metal_row': '3d'},
    'Cu': {'z_eff': 13.20, 'zeta_so_cm1':  831, 'metal_row': '3d'},
    'Zn': {'z_eff': 13.57, 'zeta_so_cm1': 1042, 'metal_row': '3d'},
    'Mo': {'z_eff': 10.97, 'zeta_so_cm1':  467, 'metal_row': '4d'},
    'Ru': {'z_eff': 12.33, 'zeta_so_cm1':  880, 'metal_row': '4d'},
    'Rh': {'z_eff': 12.67, 'zeta_so_cm1': 1097, 'metal_row': '4d'},
    'Pd': {'z_eff': 13.00, 'zeta_so_cm1': 1334, 'metal_row': '4d'},
    'Ir': {'z_eff': 17.00, 'zeta_so_cm1': 3909, 'metal_row': '5d'},
    'Pt': {'z_eff': 17.33, 'zeta_so_cm1': 4146, 'metal_row': '5d'},
}

ECP_METALS = {'Mo','Ru','Rh','Pd','Ir','Pt'}

METAL_Z = {
    'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,
    'Ni':28,'Cu':29,'Zn':30,'Mo':42,'Ru':44,'Rh':45,
    'Pd':46,'Ir':77,'Pt':78,
}
LIG_Z = {'Cl':17,'Br':35,'F':9,'N':7,'O':8,'S':16,'P':15}


def bidentate_positions(metal_pos, dist, bite_angle_deg):
    """
    Generate two donor atom positions for a bidentate ligand.
    Placed symmetrically around the x-axis in the xy-plane.
    bite_angle_deg: the chelate bite angle (angle D-M-D).
    Returns two (x,y,z) tuples.
    """
    half = math.radians(bite_angle_deg / 2.0)
    x1 = dist * math.cos(half)
    y1 = dist * math.sin(half)
    x2 = dist * math.cos(half)
    y2 = -dist * math.sin(half)
    return (x1, y1, 0.0), (x2, y2, 0.0)


def make_oct_bidentate(metal, donor_sym, dist, bite_angle,
                       axial_sym, axial_dist, n_axial=2, n_bidentate=2):
    """
    Build octahedral complex with n_bidentate bidentate ligands
    and n_axial monodentate axial ligands.

    For M(bidentate)2(axial)2 octahedral:
      - bidentate 1 in xy-plane (bite angle around x-axis)
      - bidentate 2 in xz-plane (bite angle around x-axis, rotated 90°)
      - axial ligands along y-axis

    For M(bidentate)(axial)4 octahedral:
      - one bidentate in xy-plane
      - 4 monodentate fill remaining positions
    """
    half   = math.radians(bite_angle / 2.0)
    cos_h  = math.cos(half)
    sin_h  = math.sin(half)

    atom_str = f"{metal}  0.000000  0.000000  0.000000\n"

    if n_bidentate == 2:
        # bipy/en/phen type: M(L-L)2X2
        # Bidentate 1: in xy-plane
        atom_str += f"{donor_sym}  {dist*cos_h:.4f}  {dist*sin_h:.4f}  0.0000\n"
        atom_str += f"{donor_sym}  {dist*cos_h:.4f} -{dist*sin_h:.4f}  0.0000\n"
        # Bidentate 2: rotated 90° into xz-plane
        atom_str += f"{donor_sym} -{dist*cos_h:.4f}  0.0000  {dist*sin_h:.4f}\n"
        atom_str += f"{donor_sym} -{dist*cos_h:.4f}  0.0000 -{dist*sin_h:.4f}\n"
        # Axial
        atom_str += f"{axial_sym}  0.0000  {axial_dist:.4f}  0.0000\n"
        atom_str += f"{axial_sym}  0.0000 -{axial_dist:.4f}  0.0000\n"

    elif n_bidentate == 3:
        # tris-bidentate: M(L-L)3
        for angle in [0, 120, 240]:
            rad = math.radians(angle)
            # Rotate bite positions by angle around z-axis
            x1 =  dist*cos_h*math.cos(rad) - dist*sin_h*math.sin(rad)
            y1 =  dist*cos_h*math.sin(rad) + dist*sin_h*math.cos(rad)
            x2 =  dist*cos_h*math.cos(rad) + dist*sin_h*math.sin(rad)
            y2 =  dist*cos_h*math.sin(rad) - dist*sin_h*math.cos(rad)
            atom_str += f"{donor_sym}  {x1:.4f}  {y1:.4f}  0.0000\n"
            atom_str += f"{donor_sym}  {x2:.4f}  {y2:.4f}  0.0000\n"

    elif n_bidentate == 1:
        # mono-bidentate: M(L-L)X4
        atom_str += f"{donor_sym}  {dist*cos_h:.4f}  {dist*sin_h:.4f}  0.0000\n"
        atom_str += f"{donor_sym}  {dist*cos_h:.4f} -{dist*sin_h:.4f}  0.0000\n"
        # 4 monodentate fill remaining oct positions
        atom_str += f"{axial_sym} -{axial_dist:.4f}  0.0000  0.0000\n"
        atom_str += f"{axial_sym}  0.0000  {axial_dist:.4f}  0.0000\n"
        atom_str += f"{axial_sym}  0.0000 -{axial_dist:.4f}  0.0000\n"
        atom_str += f"{axial_sym}  0.0000  0.0000  {axial_dist:.4f}\n"

    return atom_str.strip()


def make_sqpl_bidentate(metal, donor_sym, dist, bite_angle,
                        axial_sym, axial_dist):
    """
    Square planar with one bidentate + two monodentate.
    M(L-L)X2 square planar.
    """
    half = math.radians(bite_angle / 2.0)
    atom_str  = f"{metal}  0.000000  0.000000  0.000000\n"
    atom_str += f"{donor_sym}  {dist*math.cos(half):.4f}  {dist*math.sin(half):.4f}  0.0000\n"
    atom_str += f"{donor_sym}  {dist*math.cos(half):.4f} -{dist*math.sin(half):.4f}  0.0000\n"
    atom_str += f"{axial_sym} -{axial_dist:.4f}  0.0000  0.0000\n"
    atom_str += f"{axial_sym}  0.0000  {axial_dist:.4f}  0.0000\n"
    return atom_str.strip()


def make_pincer(metal, donor_sym_p, dist_p, donor_sym_n, dist_n):
    """
    PNP or NNN pincer — meridional tridentate.
    Central N/donor on x-axis, two P/N donors at ±y.
    Remaining 3 positions filled with Cl.
    """
    atom_str  = f"{metal}  0.000000  0.000000  0.000000\n"
    # Central donor (N) on +x
    atom_str += f"{donor_sym_n}  {dist_n:.4f}  0.0000  0.0000\n"
    # Two side donors (P) at ±y
    atom_str += f"{donor_sym_p}  0.0000  {dist_p:.4f}  0.0000\n"
    atom_str += f"{donor_sym_p}  0.0000 -{dist_p:.4f}  0.0000\n"
    # Fill remaining 3 oct positions
    atom_str += f"Cl -{2.400:.4f}  0.0000  0.0000\n"
    atom_str += f"Cl  0.0000  0.0000  {2.400:.4f}\n"
    atom_str += f"Cl  0.0000  0.0000 -{2.400:.4f}\n"
    return atom_str.strip()


# ══════════════════════════════════════════════════════════════
# BIDENTATE STRUCTURE DEFINITIONS
# Format: (struct_name, metal, charge, atoms_string, ligand_label)
# ══════════════════════════════════════════════════════════════

def build_structures():
    structs = []

    # ── bipy (2,2'-bipyridyl) M-N ~2.05 Å Ru/Fe, bite ~78° ──
    # M(bipy)2Cl2 octahedral — Ru, Fe, Co, Cr
    for metal, dist, charge in [
        ('Ru', 2.057, 0), ('Fe', 2.080, 0),
        ('Co', 2.120, 1), ('Cr', 2.060, 0),
        ('Ir', 2.050, 0), ('Rh', 2.070, 1),
    ]:
        name     = f"{metal}bipy2Cl2_oct"
        atom_str = make_oct_bidentate(metal, 'N', dist, 78.0, 'Cl', 2.380, n_bidentate=2)
        structs.append((name, metal, charge, atom_str, 'bipy'))

    # M(bipy)3 tris-bidentate
    for metal, dist, charge in [
        ('Ru', 2.057, 2), ('Fe', 2.080, 2),
        ('Co', 2.120, 3), ('Ir', 2.050, 3),
    ]:
        name     = f"{metal}bipy3_oct"
        atom_str = make_oct_bidentate(metal, 'N', dist, 78.0, 'N', dist, n_bidentate=3)
        structs.append((name, metal, charge, atom_str, 'bipy'))

    # M(bipy)Cl4 mono-bidentate
    for metal, dist, charge in [
        ('Ru', 2.057, -2), ('Fe', 2.080, -2),
        ('Mo', 2.220, -2), ('Mn', 2.200, -2),
    ]:
        name     = f"{metal}bipyCl4_oct"
        atom_str = make_oct_bidentate(metal, 'N', dist, 78.0, 'Cl', 2.380, n_bidentate=1)
        structs.append((name, metal, charge, atom_str, 'bipy'))

    # ── en (ethylenediamine) M-N ~2.10 Å, bite ~85° ──
    for metal, dist, charge in [
        ('Co', 2.110, 3), ('Cr', 2.070, 3),
        ('Fe', 2.090, 2), ('Ni', 2.120, 2),
        ('Cu', 2.040, 2), ('Zn', 2.150, 2),
        ('Ru', 2.130, 2), ('Pd', 2.050, 2),
    ]:
        name     = f"{metal}en2Cl2_oct"
        atom_str = make_oct_bidentate(metal, 'N', dist, 85.0, 'Cl', 2.350, n_bidentate=2)
        structs.append((name, metal, charge, atom_str, 'en'))

    # ── oxalate M-O ~2.00 Å, bite ~80° ──
    for metal, dist, charge in [
        ('Fe', 2.010, -3), ('Co', 2.000, -3),
        ('Cr', 1.990, -3), ('Mn', 2.050, -3),
        ('Ru', 2.020, -3), ('V',  2.000, -3),
    ]:
        name     = f"{metal}ox3_oct"
        atom_str = make_oct_bidentate(metal, 'O', dist, 80.0, 'O', dist, n_bidentate=3)
        structs.append((name, metal, charge, atom_str, 'ox'))

    # M(ox)2Cl2
    for metal, dist, charge in [
        ('Cu', 2.000, -4), ('Ni', 1.990, -4),
        ('Pd', 2.000, -2), ('Pt', 2.000, -2),
    ]:
        name     = f"{metal}ox2Cl2_oct"
        atom_str = make_oct_bidentate(metal, 'O', dist, 80.0, 'Cl', 2.300, n_bidentate=2)
        structs.append((name, metal, charge, atom_str, 'ox'))

    # ── acac (acetylacetonate) M-O ~1.95 Å, bite ~90° ──
    for metal, dist, charge in [
        ('Fe', 1.990, 0), ('Co', 1.970, 0),
        ('Cr', 1.950, 0), ('Mn', 2.000, 0),
        ('V',  1.960, 0), ('Ti', 2.000, 0),
        ('Ru', 2.000, 0), ('Ir', 2.000, 0),
    ]:
        name     = f"{metal}acac3_oct"
        atom_str = make_oct_bidentate(metal, 'O', dist, 90.0, 'O', dist, n_bidentate=3)
        structs.append((name, metal, charge, atom_str, 'acac'))

    # M(acac)2Cl2
    for metal, dist, charge in [
        ('Cu', 1.950, 0), ('Ni', 1.960, 0),
        ('Pd', 1.980, 0), ('Pt', 1.970, 0),
    ]:
        name     = f"{metal}acac2Cl2_oct"
        atom_str = make_oct_bidentate(metal, 'O', dist, 90.0, 'Cl', 2.300, n_bidentate=2)
        structs.append((name, metal, charge, atom_str, 'acac'))

    # ── phen (1,10-phenanthroline) M-N ~2.06 Å, bite ~79° ──
    for metal, dist, charge in [
        ('Fe', 1.970, 2), ('Co', 2.090, 2),
        ('Ni', 2.090, 2), ('Cu', 2.020, 2),
        ('Ru', 2.060, 2), ('Ir', 2.040, 3),
    ]:
        name     = f"{metal}phen2Cl2_oct"
        atom_str = make_oct_bidentate(metal, 'N', dist, 79.0, 'Cl', 2.350, n_bidentate=2)
        structs.append((name, metal, charge, atom_str, 'phen'))

    # ── dtc (dithiocarbamate) M-S ~2.25-2.35 Å, bite ~73° ──
    # Non-innocent ligand — important for Paper 2
    for metal, dist, charge in [
        ('Ni', 2.200, 0), ('Pd', 2.300, 0),
        ('Pt', 2.310, 0), ('Cu', 2.270, 0),
        ('Fe', 2.350, 0), ('Co', 2.300, 0),
        ('Mo', 2.400, 0), ('Zn', 2.340, 0),
    ]:
        name     = f"{metal}dtc2_sqpl"
        atom_str = make_sqpl_bidentate(metal, 'S', dist, 73.0, 'S', dist)
        structs.append((name, metal, charge, atom_str, 'dtc'))

    # ── Square planar bidentate (Pd, Pt, Ni, Rh) ──
    # M(bipy)Cl2 square planar
    for metal, dist, charge in [
        ('Pd', 2.040, 0), ('Pt', 2.030, 0),
        ('Ni', 2.090, 0), ('Rh', 2.070, 1),
    ]:
        name     = f"{metal}bipyCl2_sqpl"
        atom_str = make_sqpl_bidentate(metal, 'N', dist, 78.0, 'Cl', 2.300)
        structs.append((name, metal, charge, atom_str, 'bipy'))

    # ── PNP pincer — Fe, Ru, Ir (Microsoft target chemistry) ──
    # M-P ~2.25 Å, M-N ~2.05 Å
    for metal, dist_p, dist_n, charge in [
        ('Fe', 2.250, 2.050, 0), ('Ru', 2.290, 2.080, 0),
        ('Ir', 2.270, 2.070, 1), ('Co', 2.220, 2.060, 1),
        ('Rh', 2.250, 2.060, 1),
    ]:
        name     = f"{metal}PNPCl3_mer"
        atom_str = make_pincer(metal, 'P', dist_p, 'N', dist_n)
        structs.append((name, metal, charge, atom_str, 'PNP'))

    # ── NNN pincer — Fe, Mn, Co (spin crossover relevance) ──
    for metal, dist_n, charge in [
        ('Fe', 2.080, 2), ('Mn', 2.200, 2),
        ('Co', 2.090, 2), ('Ni', 2.090, 2),
    ]:
        name     = f"{metal}NNNCl3_mer"
        atom_str = make_pincer(metal, 'N', dist_n, 'N', dist_n)
        structs.append((name, metal, charge, atom_str, 'NNN'))

    return structs


CSD_STRUCTURES = build_structures()

# ── SPIN STATES ───────────────────────────────────────────────
SPIN_STATES = {
    'Ti': [0, 2],      'V':  [0, 2, 4],
    'Cr': [0, 2, 4],   'Mn': [1, 3, 5],
    'Fe': [0, 2, 4],   'Co': [1, 3],
    'Ni': [0, 2],      'Cu': [1],
    'Zn': [0],
    'Mo': [0, 1, 2, 3], 'Ru': [0, 2, 4],
    'Rh': [0, 2],      'Pd': [0, 2],
    'Ir': [0, 2],      'Pt': [0, 2],
}


def count_electrons(atoms_str, charge):
    total = 0
    for line in atoms_str.strip().split('\n'):
        sym = line.strip().split()[0]
        total += METAL_Z.get(sym, LIG_Z.get(sym, 0))
    return total - charge


def get_nact_14(n_total):
    """Find active electrons for CASSCF(14,14)."""
    for n in [14, 13, 15, 12, 16, 11, 10, 9]:
        if (n_total - n) >= 0 and (n_total - n) % 2 == 0:
            return n
    return 14


def run_uhf(mol):
    for settings in [
        dict(max_cycle=300, conv_tol=1e-10, damp=0.0, level_shift=0.0),
        dict(max_cycle=500, conv_tol=1e-9,  damp=0.3, level_shift=0.2),
        dict(max_cycle=800, conv_tol=1e-8,  damp=0.5, level_shift=0.5),
    ]:
        mf = scf.UHF(mol)
        for k, v in settings.items(): setattr(mf, k, v)
        mf.verbose = 0; mf.run()
        if mf.converged: return mf
    return mf


def run_casscf_14(mf, mol, n_act):
    """CASSCF(14,14) — expanded active space for bidentate systems."""
    e_m = (mf.mo_energy[0] + mf.mo_energy[1]) / 2
    occ = mf.mo_occ[0] + mf.mo_occ[1]
    occ_idx  = np.where(occ > 0.5)[0]
    virt_idx = np.where(occ < 0.5)[0]
    if len(occ_idx) == 0 or len(virt_idx) == 0: return None, False

    homo_e = float(e_m[occ_idx[-1]])
    lumo_e = float(e_m[virt_idx[0]])
    gap    = (homo_e + lumo_e) / 2

    # 18-orbital window for CASSCF(14,14): 9 occ + 9 virt
    w18 = sorted(np.argsort(np.abs(e_m - gap))[:18], key=lambda i: e_m[i])

    best_mc = None; best_e = 0.0
    for window in [w18[:14], w18[2:16], w18[1:15], w18[4:18]]:
        for shift in [1e-3, 1e-2, 5e-2, 1e-1, 2e-1]:
            try:
                mc = mcscf.CASSCF(mf, 14, n_act)
                mc.max_cycle_macro = 600
                mc.conv_tol        = 1e-8
                mc.ah_level_shift  = shift
                mc.verbose         = 0
                mc.kernel(mc.sort_mo(window, base=0))
                ec = mc.e_tot - mf.e_tot
                if ec < 0 and ec < best_e:
                    best_mc = mc; best_e = ec
                if mc.converged and ec < -0.01:
                    return mc, True
            except: continue

    if best_mc: return best_mc, best_mc.converged
    return None, False


def run_one(struct_name, metal, charge, atoms_str, spin, ligand_label):
    name    = f"BIDENT_{struct_name}_spin{spin}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outfile = os.path.join(OUTPUT_DIR, f"{name}.json")

    if os.path.exists(outfile):
        r = json.load(open(outfile))
        if r.get('converged') and r.get('corr_energy', 0) < -0.001:
            log.info(f"SKIP: {name}"); return True

    n_e = count_electrons(atoms_str, charge)
    if (n_e % 2) != (spin % 2):
        json.dump({'name': name, 'status': 'skipped',
                   'reason': 'parity', 'geometry': 'bidentate'},
                  open(outfile, 'w')); return True

    consts = METAL_CONSTANTS.get(metal, {'z_eff': 10.0,
                                          'zeta_so_cm1': 400,
                                          'metal_row': '3d'})
    log.info(f"Start: {name}  n_e={n_e}  ligand={ligand_label}")

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

        # CASSCF(14,14) — key difference from Paper 1 pipeline
        n_act = get_nact_14(mol.nelectron)
        mf    = run_uhf(mol)
        mc, conv = run_casscf_14(mf, mol, n_act)

        if mc is None:
            json.dump({'name': name, 'status': 'failed',
                       'geometry': 'bidentate', 'metal': metal},
                      open(outfile, 'w')); return False

        ec = mc.e_tot - mf.e_tot
        if ec >= 0:
            json.dump({'name': name, 'status': 'unphysical',
                       'corr_energy': float(ec),
                       'geometry': 'bidentate'},
                      open(outfile, 'w')); return False

        casdm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
        no, _  = np.linalg.eigh(casdm1)
        no     = np.sort(no)[::-1]
        # n_active: orbitals with fractional occupation
        n_active = sum(1 for n in no if 0.02 < n < 1.98)

        # UHF orbital features (same as Paper 1)
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
            'ligand':            ligand_label,
            'n_ligands':         2,  # bidentate counts as 2 donors
            'charge':            charge,
            'spin':              spin,
            'mult':              spin + 1,
            'geometry':          'bidentate',
            'struct_name':       struct_name,
            'n_electrons':       mol.nelectron,
            'n_active_e':        n_act,
            'casscf_size':       14,   # KEY: distinguishes from (10,10) data
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
        log.info(f"  OK: n_active={n_active} Ec={ec:.4f} "
                 f"converged={conv} ligand={ligand_label}")
        return True

    except Exception as e:
        log.error(f"  Error {name}: {e}")
        json.dump({'name': name, 'status': 'error', 'reason': str(e),
                   'geometry': 'bidentate'}, open(outfile, 'w'))
        return False


# ── BUILD JOB LIST ─────────────────────────────────────────────
ALL_JOBS = []
seen     = set()
os.makedirs(OUTPUT_DIR, exist_ok=True)
existing = set(os.path.basename(f)
               for f in __import__('glob').glob(f'{OUTPUT_DIR}/*.json'))

for struct_name, metal, charge, atoms_str, ligand_label in CSD_STRUCTURES:
    spins = SPIN_STATES.get(metal, [0, 2])
    n_e   = count_electrons(atoms_str, charge)
    for spin in spins:
        if (n_e % 2) != (spin % 2): continue
        fname = f"BIDENT_{struct_name}_spin{spin}.json"
        if fname in existing: continue
        key = (struct_name, spin)
        if key in seen: continue
        seen.add(key)
        ALL_JOBS.append((struct_name, metal, charge,
                         atoms_str, spin, ligand_label))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        from collections import defaultdict
        by_metal  = defaultdict(int)
        by_ligand = defaultdict(int)
        for s, m, c, a, sp, lig in ALL_JOBS:
            by_metal[m]   += 1
            by_ligand[lig] += 1
        print(f"\nTotal bidentate jobs: {len(ALL_JOBS)}")
        print(f"Structures:           {len(CSD_STRUCTURES)}")
        print(f"\nJobs by ligand:")
        for lig, n in sorted(by_ligand.items(), key=lambda x: -x[1]):
            print(f"  {lig}: {n}")
        print(f"\nJobs by metal (top 10):")
        for m, n in sorted(by_metal.items(), key=lambda x: -x[1])[:10]:
            print(f"  {m}: {n}")
        print(f"\nNote: CASSCF(14,14) — ~3x slower than Paper 1 jobs")
        print(f"Est. time: ~{len(ALL_JOBS)*60/60:.0f} hrs serial, "
              f"~{len(ALL_JOBS)*60/60/64:.1f} hrs on 64 cores")
        sys.exit(0)

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if idx >= len(ALL_JOBS): sys.exit(1)
    sys.exit(0 if run_one(*ALL_JOBS[idx]) else 1)
