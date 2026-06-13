"""
gen_orbital_features_real.py
============================
ML2 orbital features for real geometry structures.
Handles generated_csd (hard-coded CSD geometries) and
generated_cod (true experimental CIF coordinates).

Output format is IDENTICAL to gen_orbital_features_v3.py:
  - Same field names, same order, same types
  - Same 14-orbital window (7 occ + 7 virt)
  - Same CASCI(14, n_act_e) entropy calculation
  - Same true_label mapping from CASSCF no_occ

Key difference from v3:
  - build_mol() handles 'csd_real' via hard-coded atom strings
  - build_mol() handles 'cod_real' via CIF file re-parsing
  - Uses ECP for all 4d/5d metals (Pd, Ru, Rh, Mo, Ir, Pt)

Usage:
  python gen_orbital_features_real.py summary      # count jobs
  python gen_orbital_features_real.py <chunk_idx> <n_chunks>
  sbatch --array=0-N job_orbital_real.sh

Validation:
  python gen_orbital_features_real.py validate     # compare one output
                                                    # against v3 format
"""
import sys, os, json, glob, logging, math, re, urllib.request
import numpy as np
from pyscf import gto, scf, mcscf

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

OUTDIR   = os.path.expanduser('~/activeml/data/orbital_features_real')
CSD_DIR  = os.path.expanduser('~/activeml/data/generated_csd')
COD_DIR  = os.path.expanduser('~/activeml/data/generated_cod')
CIF_DIR  = os.path.expanduser('~/activeml/data/cod_cifs')
BASIS    = 'def2-svp'
ECP_METALS = {'Pd','Ru','Rh','Mo','Ir','Pt'}

os.makedirs(OUTDIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# CSD HARD-CODED ATOM STRINGS
# Copied verbatim from gen_csd_4d5d.py — must stay in sync
# ══════════════════════════════════════════════════════════════

CSD_ATOM_STRINGS = {
    "PdCl4_2m_sqpl": """Pd  0.000  0.000  0.000
        Cl  2.295  0.000  0.000
        Cl -2.295  0.000  0.000
        Cl  0.000  2.295  0.000
        Cl  0.000 -2.295  0.000""",
    "PdCl4_2m_sqpl_dist": """Pd  0.000  0.000  0.000
        Cl  2.305  0.080 -0.040
        Cl -2.285  0.050  0.060
        Cl  0.060  2.295 -0.030
        Cl -0.080 -2.300  0.020""",
    "PdBr4_2m_sqpl": """Pd  0.000  0.000  0.000
        Br  2.440  0.000  0.000
        Br -2.440  0.000  0.000
        Br  0.000  2.440  0.000
        Br  0.000 -2.440  0.000""",
    "PdCl6_2m_oct": """Pd  0.000  0.000  0.000
        Cl  2.310  0.000  0.000
        Cl -2.310  0.000  0.000
        Cl  0.000  2.310  0.000
        Cl  0.000 -2.310  0.000
        Cl  0.000  0.000  2.310
        Cl  0.000  0.000 -2.310""",
    "PdCl2N2_sqpl": """Pd  0.000  0.000  0.000
        Cl  2.295  0.000  0.000
        Cl -2.295  0.000  0.000
        N   0.000  2.035  0.000
        N   0.000 -2.035  0.000""",
    "PdCl4_4m_tet": """Pd  0.000  0.000  0.000
        Cl  2.380  0.000  0.000
        Cl -2.380  0.000  0.000
        Cl  0.000  2.380  0.000
        Cl  0.000 -2.380  0.000""",
    "RuCl6_3m_oct": """Ru  0.000  0.000  0.000
        Cl  2.340  0.000  0.000
        Cl -2.340  0.000  0.000
        Cl  0.000  2.340  0.000
        Cl  0.000 -2.340  0.000
        Cl  0.000  0.000  2.340
        Cl  0.000  0.000 -2.340""",
    "RuCl6_4m_oct": """Ru  0.000  0.000  0.000
        Cl  2.375  0.000  0.000
        Cl -2.375  0.000  0.000
        Cl  0.000  2.375  0.000
        Cl  0.000 -2.375  0.000
        Cl  0.000  0.000  2.375
        Cl  0.000  0.000 -2.375""",
    "RuCl6_3m_dist": """Ru  0.000  0.000  0.000
        Cl  2.330  0.100 -0.060
        Cl -2.350  0.070  0.080
        Cl  0.080  2.340 -0.050
        Cl -0.060 -2.345  0.070
        Cl  0.040  0.030  2.360
        Cl -0.050 -0.040 -2.355""",
    "RuCl6_2m_oct": """Ru  0.000  0.000  0.000
        Cl  2.320  0.000  0.000
        Cl -2.320  0.000  0.000
        Cl  0.000  2.320  0.000
        Cl  0.000 -2.320  0.000
        Cl  0.000  0.000  2.320
        Cl  0.000  0.000 -2.320""",
    "RuCl4N2_trans": """Ru  0.000  0.000  0.000
        Cl  2.340  0.000  0.000
        Cl -2.340  0.000  0.000
        Cl  0.000  2.340  0.000
        Cl  0.000 -2.340  0.000
        N   0.000  0.000  2.130
        N   0.000  0.000 -2.130""",
    "RuBr6_3m_oct": """Ru  0.000  0.000  0.000
        Br  2.480  0.000  0.000
        Br -2.480  0.000  0.000
        Br  0.000  2.480  0.000
        Br  0.000 -2.480  0.000
        Br  0.000  0.000  2.480
        Br  0.000  0.000 -2.480""",
    "RhCl4_3m_sqpl": """Rh  0.000  0.000  0.000
        Cl  2.375  0.000  0.000
        Cl -2.375  0.000  0.000
        Cl  0.000  2.375  0.000
        Cl  0.000 -2.375  0.000""",
    "RhCl6_3m_oct": """Rh  0.000  0.000  0.000
        Cl  2.325  0.000  0.000
        Cl -2.325  0.000  0.000
        Cl  0.000  2.325  0.000
        Cl  0.000 -2.325  0.000
        Cl  0.000  0.000  2.325
        Cl  0.000  0.000 -2.325""",
    "RhCl6_3m_dist": """Rh  0.000  0.000  0.000
        Cl  2.315  0.090 -0.050
        Cl -2.335  0.060  0.070
        Cl  0.070  2.325 -0.045
        Cl -0.055 -2.330  0.065
        Cl  0.035  0.025  2.345
        Cl -0.045 -0.035 -2.340""",
    "RhCl2N2_1m_sqpl": """Rh  0.000  0.000  0.000
        Cl  2.375  0.000  0.000
        Cl -2.375  0.000  0.000
        N   0.000  2.080  0.000
        N   0.000 -2.080  0.000""",
    "RhBr6_3m_oct": """Rh  0.000  0.000  0.000
        Br  2.465  0.000  0.000
        Br -2.465  0.000  0.000
        Br  0.000  2.465  0.000
        Br  0.000 -2.465  0.000
        Br  0.000  0.000  2.465
        Br  0.000  0.000 -2.465""",
    "MoCl6_3m_oct": """Mo  0.000  0.000  0.000
        Cl  2.480  0.000  0.000
        Cl -2.480  0.000  0.000
        Cl  0.000  2.480  0.000
        Cl  0.000 -2.480  0.000
        Cl  0.000  0.000  2.480
        Cl  0.000  0.000 -2.480""",
    "MoCl6_2m_oct": """Mo  0.000  0.000  0.000
        Cl  2.460  0.000  0.000
        Cl -2.460  0.000  0.000
        Cl  0.000  2.460  0.000
        Cl  0.000 -2.460  0.000
        Cl  0.000  0.000  2.460
        Cl  0.000  0.000 -2.460""",
    "MoCl6_1m_oct": """Mo  0.000  0.000  0.000
        Cl  2.440  0.000  0.000
        Cl -2.440  0.000  0.000
        Cl  0.000  2.440  0.000
        Cl  0.000 -2.440  0.000
        Cl  0.000  0.000  2.440
        Cl  0.000  0.000 -2.440""",
    "MoO4_2m_tet": """Mo  0.000  0.000  0.000
        O   1.930  0.000  0.000
        O  -1.930  0.000  0.000
        O   0.000  1.930  0.000
        O   0.000 -1.930  0.000""",
    "MoCl6_3m_dist": """Mo  0.000  0.000  0.000
        Cl  2.470  0.110 -0.070
        Cl -2.490  0.080  0.090
        Cl  0.090  2.480 -0.060
        Cl -0.070 -2.485  0.080
        Cl  0.050  0.040  2.495
        Cl -0.060 -0.050 -2.500""",
    "MoCl4O2_trans": """Mo  0.000  0.000  0.000
        Cl  2.450  0.000  0.000
        Cl -2.450  0.000  0.000
        Cl  0.000  2.450  0.000
        Cl  0.000 -2.450  0.000
        O   0.000  0.000  1.680
        O   0.000  0.000 -1.680""",
    "IrCl6_3m_oct": """Ir  0.000  0.000  0.000
        Cl  2.355  0.000  0.000
        Cl -2.355  0.000  0.000
        Cl  0.000  2.355  0.000
        Cl  0.000 -2.355  0.000
        Cl  0.000  0.000  2.355
        Cl  0.000  0.000 -2.355""",
    "IrCl4_3m_sqpl": """Ir  0.000  0.000  0.000
        Cl  2.375  0.000  0.000
        Cl -2.375  0.000  0.000
        Cl  0.000  2.375  0.000
        Cl  0.000 -2.375  0.000""",
    "IrCl6_3m_dist": """Ir  0.000  0.000  0.000
        Cl  2.345  0.100 -0.060
        Cl -2.365  0.070  0.080
        Cl  0.080  2.355 -0.050
        Cl -0.060 -2.360  0.070
        Cl  0.040  0.030  2.370
        Cl -0.050 -0.040 -2.365""",
    "IrCl6_2m_oct": """Ir  0.000  0.000  0.000
        Cl  2.340  0.000  0.000
        Cl -2.340  0.000  0.000
        Cl  0.000  2.340  0.000
        Cl  0.000 -2.340  0.000
        Cl  0.000  0.000  2.340
        Cl  0.000  0.000 -2.340""",
    "IrCl4N2_3m_trans": """Ir  0.000  0.000  0.000
        Cl  2.355  0.000  0.000
        Cl -2.355  0.000  0.000
        Cl  0.000  2.355  0.000
        Cl  0.000 -2.355  0.000
        N   0.000  0.000  2.090
        N   0.000  0.000 -2.090""",
    "IrBr6_3m_oct": """Ir  0.000  0.000  0.000
        Br  2.490  0.000  0.000
        Br -2.490  0.000  0.000
        Br  0.000  2.490  0.000
        Br  0.000 -2.490  0.000
        Br  0.000  0.000  2.490
        Br  0.000  0.000 -2.490""",
    "PtCl4_2m_sqpl": """Pt  0.000  0.000  0.000
        Cl  2.305  0.000  0.000
        Cl -2.305  0.000  0.000
        Cl  0.000  2.305  0.000
        Cl  0.000 -2.305  0.000""",
    "PtCl4_2m_sqpl_dist": """Pt  0.000  0.000  0.000
        Cl  2.315  0.075 -0.035
        Cl -2.295  0.050  0.055
        Cl  0.055  2.305 -0.028
        Cl -0.075 -2.310  0.018""",
    "PtCl2N2_sqpl": """Pt  0.000  0.000  0.000
        Cl  2.305  0.000  0.000
        Cl -2.305  0.000  0.000
        N   0.000  2.025  0.000
        N   0.000 -2.025  0.000""",
    "PtCl6_2m_oct": """Pt  0.000  0.000  0.000
        Cl  2.325  0.000  0.000
        Cl -2.325  0.000  0.000
        Cl  0.000  2.325  0.000
        Cl  0.000 -2.325  0.000
        Cl  0.000  0.000  2.325
        Cl  0.000  0.000 -2.325""",
    "PtCl6_2m_dist": """Pt  0.000  0.000  0.000
        Cl  2.315  0.095 -0.055
        Cl -2.335  0.065  0.075
        Cl  0.075  2.325 -0.048
        Cl -0.058 -2.330  0.068
        Cl  0.038  0.028  2.340
        Cl -0.048 -0.038 -2.335""",
    "PtBr4_2m_sqpl": """Pt  0.000  0.000  0.000
        Br  2.445  0.000  0.000
        Br -2.445  0.000  0.000
        Br  0.000  2.445  0.000
        Br  0.000 -2.445  0.000""",
}


# ══════════════════════════════════════════════════════════════
# CIF PARSER (same logic as gen_cod_real.py)
# ══════════════════════════════════════════════════════════════

def parse_cif_to_atom_str(cif_path, metal):
    """
    Re-parse CIF file to get atom string.
    Returns atom_str (centered on metal) or None.
    Identical to gen_cod_real.py parse_cif_coordinates().
    """
    ALLOWED = {'Cl','Br','F','N','O','S','P','H','C','B'}
    try:
        with open(cif_path, 'r', errors='ignore') as f:
            content = f.read()
    except:
        return None

    def get_val(key):
        m = re.search(rf'{key}\s+([\d.]+)', content)
        return float(m.group(1)) if m else None

    a = get_val('_cell_length_a')
    b = get_val('_cell_length_b')
    c = get_val('_cell_length_c')
    if not all([a, b, c]):
        return None

    alpha = get_val('_cell_angle_alpha') or 90.0
    beta  = get_val('_cell_angle_beta')  or 90.0
    gamma = get_val('_cell_angle_gamma') or 90.0

    al = math.radians(alpha); be = math.radians(beta); ga = math.radians(gamma)
    cos_al = math.cos(al); cos_be = math.cos(be)
    cos_ga = math.cos(ga); sin_ga = math.sin(ga)
    v = math.sqrt(max(0, 1 - cos_al**2 - cos_be**2 - cos_ga**2
                      + 2*cos_al*cos_be*cos_ga))
    M = np.array([
        [a, b*cos_ga, c*cos_be],
        [0, b*sin_ga, c*(cos_al - cos_be*cos_ga)/max(sin_ga, 1e-10)],
        [0, 0,        c*v/max(sin_ga, 1e-10)],
    ])

    loop_match = re.search(
        r'loop_\s*((?:_atom_site_\S+\s*)+)((?:(?!loop_|_\w)[\s\S])*)',
        content)
    if not loop_match:
        return None

    headers    = re.findall(r'_atom_site_\S+', loop_match.group(1))
    data_block = loop_match.group(2)
    col        = {h: i for i, h in enumerate(headers)}

    use_frac = ('_atom_site_fract_x' in col and
                '_atom_site_fract_y' in col and
                '_atom_site_fract_z' in col)
    label_col = col.get('_atom_site_label',
                        col.get('_atom_site_type_symbol', None))
    if label_col is None or not use_frac:
        return None

    atoms = []
    metal_positions = []
    for line in data_block.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('_') or line.startswith('#'):
            continue
        line = re.sub(r'\([\d]+\)', '', line)
        parts = line.split()
        if len(parts) < len(headers):
            continue
        try:
            sym_m = re.match(r'([A-Z][a-z]?)', parts[label_col])
            if not sym_m:
                continue
            sym = sym_m.group(1)
            fx  = float(parts[col['_atom_site_fract_x']])
            fy  = float(parts[col['_atom_site_fract_y']])
            fz  = float(parts[col['_atom_site_fract_z']])
            pos = M @ np.array([fx, fy, fz])
            atoms.append((sym, pos))
            if sym == metal:
                metal_positions.append(len(atoms) - 1)
        except (ValueError, IndexError):
            continue

    if not atoms or not metal_positions:
        return None

    metal_idx = metal_positions[0]
    metal_sym, metal_pos = atoms[metal_idx]

    coord_atoms = [(metal_sym, metal_pos)]
    for i, (sym, pos) in enumerate(atoms):
        if i == metal_idx or sym == 'H':
            continue
        dist = float(np.linalg.norm(pos - metal_pos))
        if 1.5 <= dist <= 2.8:
            coord_atoms.append((sym, pos))

    n_ligands = len(coord_atoms) - 1
    if n_ligands < 2 or n_ligands > 8:
        return None

    ligand_syms = set(s for s, _ in coord_atoms[1:])
    if not ligand_syms.issubset(ALLOWED):
        return None

    center   = coord_atoms[0][1].copy()
    atom_str = ''
    for sym, pos in coord_atoms:
        p = pos - center
        atom_str += f"{sym}  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}\n"

    return atom_str.strip()


# ══════════════════════════════════════════════════════════════
# MOLECULE BUILDER
# ══════════════════════════════════════════════════════════════

def build_mol(d):
    """
    Build PySCF mol from JSON record.
    Handles csd_real (lookup from CSD_ATOM_STRINGS)
    and cod_real (re-parse CIF file).
    """
    geom   = d.get('geometry', 'oct')
    metal  = d['metal']
    charge = d['charge']
    spin   = d['spin']

    # ── CSD real geometry ─────────────────────────────────────
    if geom == 'csd_real':
        struct_name = d.get('struct_name', '')
        atom_str    = CSD_ATOM_STRINGS.get(struct_name)
        if atom_str is None:
            log.warning(f"CSD struct_name not found: {struct_name}")
            return None

    # ── COD real geometry ─────────────────────────────────────
    elif geom == 'cod_real':
        cod_id   = d.get('cod_id', '')
        cif_path = os.path.join(CIF_DIR, metal, f"{cod_id}.cif")
        if not os.path.exists(cif_path):
            log.warning(f"CIF not found: {cif_path}")
            return None
        atom_str = parse_cif_to_atom_str(cif_path, metal)
        if atom_str is None:
            log.warning(f"CIF parse failed: {cif_path}")
            return None

    else:
        log.warning(f"Unexpected geometry: {geom}")
        return None

    mol = gto.Mole()
    mol.atom      = atom_str
    mol.basis     = BASIS
    mol.charge    = charge
    mol.spin      = spin
    mol.verbose   = 0
    mol.max_memory = 28000
    if metal in ECP_METALS:
        mol.ecp = BASIS
    mol.build()
    return mol


# ══════════════════════════════════════════════════════════════
# UHF (identical to v3)
# ══════════════════════════════════════════════════════════════

def run_uhf(mol):
    for s in [
        dict(max_cycle=300, conv_tol=1e-10, damp=0.0, level_shift=0.0),
        dict(max_cycle=500, conv_tol=1e-9,  damp=0.3, level_shift=0.2),
        dict(max_cycle=800, conv_tol=1e-8,  damp=0.5, level_shift=0.5),
    ]:
        mf = scf.UHF(mol)
        mf.max_memory = 28000
        for k, v in s.items():
            setattr(mf, k, v)
        mf.verbose = 0
        mf.run()
        if mf.converged:
            return mf
    return mf


def get_active_electrons(n_total):
    for n in [10, 9, 11, 8, 12, 7, 13, 6, 14]:
        if (n_total - n) >= 0 and (n_total - n) % 2 == 0:
            return n
    return 10


# ══════════════════════════════════════════════════════════════
# MAIN PROCESSING (identical logic to v3.process())
# ══════════════════════════════════════════════════════════════

def process(src_filepath):
    try:
        d = json.load(open(src_filepath))
    except:
        return False

    if d.get('status') != 'ok':
        return True

    name     = d['name']
    n_active = d.get('n_active', 0)
    no_occ   = d.get('no_occ', [])

    if n_active == 0 or not no_occ:
        return True

    outfile = os.path.join(OUTDIR, f'{name}.json')
    if os.path.exists(outfile):
        try:
            r = json.load(open(outfile))
            if r.get('status') == 'done':
                log.info(f"SKIP: {name}")
                return True
        except:
            pass

    # Step 1: True active ranks from CASSCF no_occ
    # Identical to v3 — positions 0-9 in no_occ where 0.02 < n < 1.98
    true_ranks_casscf = set(i for i, n in enumerate(no_occ) if 0.02 < n < 1.98)

    # Step 2: Build molecule from real geometry
    try:
        mol = build_mol(d)
        if mol is None:
            return True
    except Exception as e:
        log.warning(f"Build failed {name}: {e}")
        return False

    if (mol.nelectron % 2) != (d['spin'] % 2):
        return True

    # Step 3: UHF
    try:
        mf   = run_uhf(mol)
        E_HF = float(mf.e_tot)
        log.info(f"  UHF: E={E_HF:.6f} converged={mf.converged}")
    except Exception as e:
        log.error(f"UHF failed {name}: {e}")
        json.dump({'name': name, 'status': 'failed', 'error': str(e)},
                  open(outfile, 'w'), indent=2)
        return False

    # Step 4: Build 14-orbital window (identical to v3)
    occ_total = mf.mo_occ[0] + mf.mo_occ[1]
    e_mean    = (mf.mo_energy[0] + mf.mo_energy[1]) / 2
    occ_idx   = np.where(occ_total > 0.5)[0]
    virt_idx  = np.where(occ_total < 0.5)[0]

    if len(occ_idx) < 7 or len(virt_idx) < 7:
        return True

    homo_idx = int(occ_idx[-1])
    homo_e   = float(e_mean[homo_idx])
    lumo_e   = float(e_mean[virt_idx[0]])
    gap_cen  = (homo_e + lumo_e) / 2

    # 14-orbital window: 7 occ + 7 virt
    window14 = sorted(list(occ_idx[-7:]) + list(virt_idx[:7]))

    # Step 5: Label mapping (identical to v3)
    # For real geometry structures, no_occ has exactly 10 entries
    # (from CASSCF(10,10)), so positions map directly to window
    w14_sorted = sorted(window14, key=lambda i: e_mean[i])
    w14_rank   = {orb: rank for rank, orb in enumerate(w14_sorted)}

    # Real geometry no_occ always has <= 10 entries — direct mapping
    true_active_14 = true_ranks_casscf

    # Step 6: CASCI on 14-window (identical to v3)
    s_i_14       = None
    no_occ_cas14 = None
    prec_entropy = None
    perf_entropy = False
    E_CASCI      = None

    try:
        n_act_e = get_active_electrons(mol.nelectron)
        mo_avg  = (mf.mo_coeff[0] + mf.mo_coeff[1]) / 2
        mc      = mcscf.CASCI(mf, 14, n_act_e)
        mc.verbose = 0
        mo = mc.sort_mo(window14, mo_coeff=mo_avg, base=0)
        mc.kernel(mo)
        E_CASCI  = float(mc.e_tot)
        casdm1   = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
        no_occ_raw, _ = np.linalg.eigh(casdm1)
        no_occ_cas14  = np.sort(no_occ_raw)[::-1]
        eps    = 1e-12
        n_clip = np.clip(no_occ_cas14 / 2, eps, 1 - eps)
        s_i_14 = -(n_clip * np.log(n_clip) + (1 - n_clip) * np.log(1 - n_clip))
        log.info(f"  CASCI done: E={E_CASCI:.6f}")
    except Exception as e:
        log.warning(f"CASCI failed {name}: {e}")

    # Step 7: Build per-orbital data (identical field names/order to v3)
    win_dist   = np.array([abs(e_mean[i] - gap_cen) for i in w14_sorted])
    energy_sel = set(np.argsort(win_dist)[:n_active])
    prec_energy = len(energy_sel & true_active_14) / max(n_active, 1)
    perf_energy = (energy_sel == true_active_14)

    orbital_data = []
    for pos, orb_idx in enumerate(w14_sorted):
        hf_occ = float(occ_total[orb_idx])
        hf_e   = float(e_mean[orb_idx])
        dist_g = float(abs(hf_e - gap_cen))
        dist_h = float(orb_idx - homo_idx)

        if s_i_14 is not None:
            orb_s_i    = float(s_i_14[pos])
            orb_no_occ = float(no_occ_cas14[pos])
            orb_nf     = float(min(orb_no_occ, 2.0 - orb_no_occ))
        else:
            orb_s_i = orb_no_occ = orb_nf = float('nan')

        # Field names and order identical to v3
        orbital_data.append({
            'window_pos':  pos,
            'orbital_idx': int(orb_idx),
            'dist_homo':   dist_h,
            'hf_energy':   hf_e,
            'hf_occ':      hf_occ,
            'is_occupied': 1 if hf_occ > 0.5 else 0,
            'dist_gap':    dist_g,
            'no_occ_cas':  orb_no_occ,
            'noon_frac':   orb_nf,
            's_i':         orb_s_i,
            'true_label':  1 if pos in true_active_14 else 0,
        })

    # Entropy precision
    if s_i_14 is not None and n_active > 0:
        s_vals   = [o['s_i'] for o in orbital_data]
        top_si   = set(np.argsort(s_vals)[::-1][:n_active])
        prec_entropy = float(len(top_si & true_active_14) / n_active)
        perf_entropy = bool(top_si == true_active_14)

    # Step 8: Write result — field names/order IDENTICAL to v3
    result = {
        'name':         name,
        'status':       'done',
        'metal':        d['metal'],
        'ligand':       d['ligand'],
        'charge':       d['charge'],
        'spin':         d['spin'],
        'n_active':     n_active,
        'n_active_e':   d.get('n_active_e', 0),
        'E_HF':         E_HF,
        'E_CASCI':      E_CASCI,
        'homo_idx':     homo_idx,
        'window14':     [int(x) for x in window14],
        'prec_energy':  float(prec_energy),
        'perf_energy':  bool(perf_energy),
        'prec_entropy': prec_entropy,
        'perf_entropy': perf_entropy,
        'orbitals':     orbital_data,
    }

    with open(outfile, 'w') as f:
        json.dump(result, f, indent=2)

    prec_str = f"{prec_entropy:.3f}" if prec_entropy is not None else "N/A"
    log.info(f"  {name}: n_active={n_active} "
             f"prec_entropy={prec_str} perf={perf_entropy}")
    return True


# ══════════════════════════════════════════════════════════════
# BUILD INDEX OF ALL REAL GEOMETRY FILES
# ══════════════════════════════════════════════════════════════

def build_index():
    """Return sorted list of all source JSON paths to process."""
    files = []
    for d in [CSD_DIR, COD_DIR]:
        files.extend(sorted(glob.glob(f'{d}/*.json')))
    return files


# ══════════════════════════════════════════════════════════════
# VALIDATION — compare one output against v3 format
# ══════════════════════════════════════════════════════════════

def validate():
    """
    Check that output files match v3 format exactly.
    Compares field names, types, and orbital list structure.
    """
    v3_sample = glob.glob(
        os.path.expanduser('~/activeml/data/orbital_features_v3/*.json'))
    if not v3_sample:
        print("No v3 files found for comparison")
        return

    v3 = json.load(open(v3_sample[0]))
    print("=== v3 top-level keys ===")
    print(sorted(v3.keys()))
    print("\n=== v3 orbital keys ===")
    print(sorted(v3['orbitals'][0].keys()))

    real_files = glob.glob(f'{OUTDIR}/*.json')
    if not real_files:
        print("\nNo real orbital feature files yet — run jobs first")
        return

    real = json.load(open(real_files[0]))
    print("\n=== real top-level keys ===")
    print(sorted(real.keys()))
    print("\n=== real orbital keys ===")
    print(sorted(real['orbitals'][0].keys()))

    # Check key match
    v3_keys   = set(v3.keys())
    real_keys = set(real.keys())
    if v3_keys == real_keys:
        print("\nTop-level keys: MATCH")
    else:
        print(f"\nTop-level keys MISMATCH:")
        print(f"  Only in v3:   {v3_keys - real_keys}")
        print(f"  Only in real: {real_keys - v3_keys}")

    v3_orb   = set(v3['orbitals'][0].keys())
    real_orb = set(real['orbitals'][0].keys())
    if v3_orb == real_orb:
        print("Orbital keys: MATCH")
    else:
        print(f"Orbital keys MISMATCH:")
        print(f"  Only in v3:   {v3_orb - real_orb}")
        print(f"  Only in real: {real_orb - v3_orb}")

    print(f"\nv3 n_orbitals:   {len(v3['orbitals'])}")
    print(f"real n_orbitals: {len(real['orbitals'])}")
    print(f"\nSample real file: {os.path.basename(real_files[0])}")
    print(f"  prec_entropy: {real.get('prec_entropy')}")
    print(f"  perf_entropy: {real.get('perf_entropy')}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':

    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        files   = build_index()
        done    = len(glob.glob(f'{OUTDIR}/*.json'))
        pending = len(files) - done
        print(f"\nTotal source files: {len(files)}")
        print(f"  generated_csd:    "
              f"{len(glob.glob(f'{CSD_DIR}/*.json'))}")
        print(f"  generated_cod:    "
              f"{len(glob.glob(f'{COD_DIR}/*.json'))}")
        print(f"\nAlready done: {done}")
        print(f"Pending:      {pending}")
        print(f"\nEst. time: ~{len(files)*20/60:.0f} hrs serial, "
              f"~{len(files)*20/60/16:.1f} hrs on 16 cores")
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == 'validate':
        validate()
        sys.exit(0)

    # SLURM chunk mode — same interface as v3
    chunk_idx   = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    chunk_total = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    all_files  = build_index()
    chunk_size = len(all_files) // chunk_total + 1
    start      = chunk_idx * chunk_size
    end        = min(start + chunk_size, len(all_files))
    files      = all_files[start:end]

    log.info(f"Chunk {chunk_idx}/{chunk_total}: {len(files)} files")
    done = 0
    for f in files:
        if process(f):
            done += 1
    log.info(f"Done: {done}/{len(files)}")
