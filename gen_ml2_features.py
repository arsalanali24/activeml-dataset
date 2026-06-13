"""
gen_ml2_features.py
===================
Extracts ML2 features from existing UHF calculations for all training data.
Reads existing JSON files, re-runs targeted analysis on the stored
wavefunction information, and writes ML2-enhanced JSON files.

ML2 NEW FEATURES (beyond ML1):
  1. per_ligand_charge    — estimated ligand charge from coordination chemistry
                            rules. Fixes d_count for non-halide systems.
  2. metal_nat_charge     — natural/Mulliken charge on metal atom (recomputed
                            properly from density matrix)
  3. d_occ_strong         — UNO occupations strongly fractional (0.1 < n < 1.9)
                            distinguishes from weakly fractional
  4. d_occ_weak           — weakly fractional (0.02 < n < 0.1 or 1.9 < n < 1.98)
  5. homo_d_char          — d-orbital character of HOMO (0-1)
  6. lumo_d_char          — d-orbital character of LUMO (0-1)
  7. ligand_field_split   — proxy for Δ_oct from orbital energy differences
  8. sigma_pi_ratio       — ratio of sigma to pi bonding ligands
  9. coordination_class   — octahedral/tetrahedral/square_planar encoded
                            with ligand-field-aware numbering
  10. metal_ox_state      — estimated oxidation state from ligand charges
  11. d_count_corrected   — d_electron_count using per-ligand charges (ML2 key)
  12. n_frac_strong       — count of strongly fractional NOs (0.1 < n < 1.9)
  13. t2g_eg_split        — for octahedral: orbital energy gap t2g vs eg

APPROACH:
  - Re-runs PySCF UHF on stored geometry to get full density matrix
  - Extracts all new features in one pass
  - Falls back to estimating features from existing JSON data if rerun fails
  - Saves ML2 features alongside ML1 features in new JSON files

OUTPUT: ~/activeml/data/ml2_features/<folder>/<original_name>_ml2.json

USAGE:
  python gen_ml2_features.py summary
  python gen_ml2_features.py <global_idx>     # for SLURM array
  python gen_ml2_features.py estimate_all     # fast: estimate without rerunning UHF
  sbatch --array=0-N job_ml2_features.sh
"""
import numpy as np, json, os, sys, glob, logging
from pyscf import gto, scf
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

BASE    = os.path.expanduser('~/activeml/data')
OUT_DIR = os.path.join(BASE, 'ml2_features')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Chemistry knowledge for ligand charge estimation ──────────
# Per-ligand charge used to compute correct oxidation state
# This is the key fix for non-halide systems (VOacac2 etc.)
LIGAND_CHARGE = {
    # Halides — anionic
    'Cl': -1, 'Br': -1, 'F': -1, 'I': -1,
    # Oxo/hydroxo — anionic
    'O':  -2, 'OH': -1, 'O2': -2,
    # Nitrido/imido — anionic
    'N':  -3, 'NH': -2, 'NH2': -1,
    # Sulfido
    'S':  -2,
    # Cyanide — anionic
    'CN': -1,
    # Neutral sigma donors
    'NH3':  0, 'H2O': 0, 'PH3': 0, 'CO': 0,
    'P':    0, 'N_neutral': 0,
    # Bidentate — anionic
    'acac': -1, 'ox': -2, 'dtc': -1,
    # Bidentate — neutral
    'bipy': 0, 'en': 0, 'phen': 0, 'NNN': 0,
    # Porphyrin — dianionic
    'N4': -2, 'N4Cl': -2, 'N4Cl2': -2, 'N4O': -2,
    # Pincer — anionic (PNP typically -1)
    'PNP': -1,
    # Carbonyl
    'CO':   0,
}

LIGAND_DENTICITY = {
    'Cl':1,'Br':1,'F':1,'I':1,'O':1,'N':1,'S':1,'CN':1,
    'NH3':1,'H2O':1,'PH3':1,'CO':1,'P':1,
    'bipy':2,'en':2,'phen':2,'acac':2,'ox':2,'dtc':2,
    'NNN':3,'PNP':3,'N4':4,'N4Cl':4,'N4Cl2':4,'N4O':4,
}

LIGAND_TYPE = {
    'Cl':'sigma','Br':'sigma','F':'sigma','I':'sigma',
    'O':'sigma','OH':'sigma','N':'sigma','NH3':'sigma',
    'H2O':'sigma','PH3':'sigma','en':'sigma','NNN':'sigma',
    'CO':'pi_acceptor','CN':'pi_acceptor',
    'bipy':'pi_acceptor','phen':'pi_acceptor',
    'acac':'pi_donor','ox':'pi_donor',
    'N4':'pi_donor','PNP':'mixed',
}

GROUP_NUMBER = {
    'Ti':4,'V':5,'Cr':6,'Mn':7,'Fe':8,'Co':9,'Ni':10,'Cu':11,'Zn':12,
    'Mo':6,'Ru':8,'Rh':9,'Pd':10,'Tc':7,
    'W':6,'Re':7,'Os':8,'Ir':9,'Pt':10,
}
ECP_METALS = {'Mo','Ru','Rh','Pd','W','Re','Os','Ir','Pt'}
ATOM_Z = {
    'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,
    'Cu':29,'Zn':30,'Mo':42,'Ru':44,'Rh':45,'Pd':46,
    'W':74,'Re':75,'Os':76,'Ir':77,'Pt':78,
    'Cl':17,'Br':35,'F':9,'I':53,'N':7,'O':8,'P':15,'C':6,'H':1,
}
METAL_CONSTANTS = {
    'Ti':{'z_eff':7.74, 'zeta_so_cm1':75,   'metal_row':'3d'},
    'V': {'z_eff':8.45, 'zeta_so_cm1':167,  'metal_row':'3d'},
    'Cr':{'z_eff':9.76, 'zeta_so_cm1':273,  'metal_row':'3d'},
    'Mn':{'z_eff':10.53,'zeta_so_cm1':355,  'metal_row':'3d'},
    'Fe':{'z_eff':11.18,'zeta_so_cm1':460,  'metal_row':'3d'},
    'Co':{'z_eff':12.00,'zeta_so_cm1':533,  'metal_row':'3d'},
    'Ni':{'z_eff':12.78,'zeta_so_cm1':669,  'metal_row':'3d'},
    'Cu':{'z_eff':13.20,'zeta_so_cm1':831,  'metal_row':'3d'},
    'Zn':{'z_eff':13.57,'zeta_so_cm1':1042, 'metal_row':'3d'},
    'Mo':{'z_eff':10.97,'zeta_so_cm1':467,  'metal_row':'4d'},
    'Ru':{'z_eff':12.33,'zeta_so_cm1':880,  'metal_row':'4d'},
    'Rh':{'z_eff':12.67,'zeta_so_cm1':1097, 'metal_row':'4d'},
    'Pd':{'z_eff':13.00,'zeta_so_cm1':1334, 'metal_row':'4d'},
    'W': {'z_eff':11.61,'zeta_so_cm1':2748, 'metal_row':'5d'},
    'Re':{'z_eff':17.01,'zeta_so_cm1':2456, 'metal_row':'5d'},
    'Os':{'z_eff':17.17,'zeta_so_cm1':3381, 'metal_row':'5d'},
    'Ir':{'z_eff':17.00,'zeta_so_cm1':3909, 'metal_row':'5d'},
    'Pt':{'z_eff':17.33,'zeta_so_cm1':4146, 'metal_row':'5d'},
}

# ── Training folders to process ───────────────────────────────
TRAIN_FOLDERS = [
    'generated300', 'generated_4d5d', 'generated_polyatomic',
    'generated_csd', 'generated_cod', 'generated_csd_extra',
    'generated_cod_extra', 'generated_bidentate', 'generated_phosphine',
    'generated_row5d_extra', 'generated_porphyrin', 'generated_closed_shell',
    'generated_carbonyl', 'generated_cod_phosphine', 'generated_cod_porphyrin',
    'generated_gap_fill', 'generated_cod_gap_fill',
    'benchmark_test_cases', 'new_test_cases',
]


# ══════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════

def estimate_oxidation_state(metal, ligand, n_ligands, charge,
                              n_halide=0, atom_str=''):
    """
    Estimate metal oxidation state from ligand charges.
    This is the KEY ML2 improvement — correct for non-halide ligands.
    
    metal_ox + sum(ligand_charges) = molecular_charge
    metal_ox = molecular_charge - sum(ligand_charges)
    """
    lig_charge = LIGAND_CHARGE.get(str(ligand), 0)
    denticity  = LIGAND_DENTICITY.get(str(ligand), 1)

    # Count actual coordination sites
    if atom_str:
        # Count from atom string for accuracy
        lines = [l.strip() for l in atom_str.split('\n') if l.strip()]
        coord_atoms = lines[1:]  # skip metal
        n_coord = len(coord_atoms)
    else:
        n_coord = n_ligands * denticity

    total_lig_charge = lig_charge * n_ligands
    metal_ox = charge - total_lig_charge
    group    = GROUP_NUMBER.get(metal, 0)
    d_count  = max(0, group - metal_ox) if group > 0 else 0

    return metal_ox, d_count


def get_sigma_pi_ratio(ligand):
    """Ratio of sigma to pi character. Higher = more sigma donor."""
    ltype = LIGAND_TYPE.get(str(ligand), 'sigma')
    if ltype == 'sigma':       return 1.0
    elif ltype == 'pi_acceptor': return 0.0
    elif ltype == 'pi_donor':    return 0.5
    else:                        return 0.5


def coordination_class_ml2(geometry, n_ligands):
    """
    ML2 coordination class encoding — ligand-field aware.
    Octahedral strong field vs weak field distinguished.
    """
    g = str(geometry).lower()
    if 'oct' in g or g == 'csd_real' or g == 'cod_real':
        return 0  # octahedral
    elif 'sq_pl' in g:
        return 1  # square planar
    elif 'tet' in g:
        return 2  # tetrahedral
    elif 'sqpyr' in g:
        return 3  # square pyramidal
    elif 'tbp' in g:
        return 4  # trigonal bipyramidal
    elif 'linear' in g:
        return 5  # linear
    elif 'bidentate' in g:
        return 6
    elif 'porphyrin' in g:
        return 7
    elif 'closed_shell' in g:
        return 8
    else:
        return 9


def compute_ml2_from_existing(d):
    """
    Compute ML2 features from existing JSON data WITHOUT rerunning UHF.
    Fast path — uses stored features and chemistry knowledge.
    """
    metal    = d.get('metal', '?')
    ligand   = str(d.get('ligand', 'Cl'))
    n_lig    = d.get('n_ligands', 0)
    charge   = d.get('charge', 0)
    spin     = d.get('spin', 0)
    geom     = d.get('geometry', 'oct')
    atom_str = d.get('atom_str', '')

    # 1. Per-ligand charge and corrected oxidation state
    metal_ox, d_count_corr = estimate_oxidation_state(
        metal, ligand, n_lig, charge, atom_str=atom_str)

    # 2. Sigma/pi ratio
    sig_pi = get_sigma_pi_ratio(ligand)

    # 3. Coordination class
    coord_cls = coordination_class_ml2(geom, n_lig)

    # 4. UNO analysis from stored no_occ
    no_occ = d.get('no_occ', [])
    if no_occ:
        no = np.array(no_occ)
        n_strong = int(np.sum((no > 0.10) & (no < 1.90)))
        n_weak   = int(np.sum(
            ((no > 0.02) & (no <= 0.10)) |
            ((no >= 1.90) & (no < 1.98))
        ))
    else:
        n_strong = 0; n_weak = 0

    # 5. Ligand field splitting proxy
    # From HOMO-LUMO gap and orbital energies
    homo = d.get('homo_energy', 0.0)
    lumo = d.get('lumo_energy', 0.0)
    hl   = d.get('homo_lumo_gap', 0.0)

    # Estimate t2g-eg split for octahedral
    # Approximation: for d6 LS, gap correlates with Δ_oct
    if 'oct' in geom and spin == 0:
        lf_split = float(hl) * 27.2114  # convert to eV
    else:
        lf_split = 0.0

    # 6. d-orbital character (approximate from spin contamination)
    # High spin contamination → more d character in frontier orbitals
    sc = d.get('spin_contamination', 0.0)
    homo_d_char = min(1.0, float(sc) / (spin + 0.1)) if spin > 0 else 0.0
    lumo_d_char = 1.0 - homo_d_char  # complement

    # 7. Effective coordination number
    # Accounts for bidentate ligands
    dent = LIGAND_DENTICITY.get(ligand, 1)
    n_coord_eff = n_lig * dent

    # 8. Metal natural charge proxy
    # From Mulliken charge if available, else estimate
    nat_charge = d.get('mulliken_metal_charge', 0.0)
    if nat_charge == 0.0:
        # Estimate from oxidation state with screening
        nat_charge = metal_ox * 0.4  # partial charge ~40% of formal ox

    # 9. Bond order features
    bo_mean = d.get('mayer_bond_order_mean', 0.0)
    bo_std  = d.get('mayer_bond_order_std',  0.0)

    # 10. MP2 fraction features (improved binning)
    mp2_002 = d.get('n_frac_mp2_002', 0)
    mp2_010 = d.get('n_frac_mp2_010', 0)
    mp2_ratio = float(mp2_010) / max(float(mp2_002), 1.0)

    # 11. Spin per d-electron
    d_count_ml1 = d.get('d_electron_count', 0)
    spin_per_d  = float(spin) / max(float(d_count_ml1), 1.0)

    # 12. Row-specific SO coupling
    zeta = d.get('zeta_so_cm1', 0.0)
    row  = d.get('metal_row', '3d')
    so_heavy = 1 if row == '5d' else (0.5 if row == '4d' else 0.0)

    return {
        # Corrected oxidation state features
        'metal_ox_state':      float(metal_ox),
        'd_count_corrected':   float(d_count_corr),
        'per_ligand_charge':   float(LIGAND_CHARGE.get(ligand, 0)),
        'total_ligand_charge': float(LIGAND_CHARGE.get(ligand, 0) * n_lig),

        # Ligand field features
        'sigma_pi_ratio':      float(sig_pi),
        'coord_class_ml2':     float(coord_cls),
        'n_coord_effective':   float(n_coord_eff),
        'ligand_field_split_eV': float(lf_split),

        # Orbital character
        'homo_d_char':         float(homo_d_char),
        'lumo_d_char':         float(lumo_d_char),

        # UNO improved analysis
        'n_frac_strong':       float(n_strong),
        'n_frac_weak':         float(n_weak),
        'n_frac_ratio':        float(n_strong) / max(float(n_weak), 1.0),

        # Metal charge
        'metal_nat_charge':    float(nat_charge),

        # Bond order
        'bo_mean':             float(bo_mean),
        'bo_std':              float(bo_std),

        # MP2 improved
        'mp2_ratio_010_002':   float(mp2_ratio),

        # Spin features
        'spin_per_d_electron': float(spin_per_d),

        # SO coupling
        'so_coupling_heavy':   float(so_heavy),
        'zeta_times_spin':     float(zeta) * float(spin) / 1000.0,
    }


def compute_ml2_with_uhf(d):
    """
    Compute ML2 features by rerunning UHF for accurate density matrix.
    Used when we need accurate metal natural charge and orbital character.
    Falls back to estimate if UHF fails.
    """
    metal    = d.get('metal', '?')
    charge   = d.get('charge', 0)
    spin     = d.get('spin', 0)
    atom_str = d.get('atom_str', '')

    if not atom_str or metal not in METAL_CONSTANTS:
        return compute_ml2_from_existing(d)

    try:
        mol = gto.Mole()
        mol.atom    = atom_str
        mol.basis   = 'def2-SVP'
        mol.charge  = charge
        mol.spin    = spin
        mol.verbose = 0
        if metal in ECP_METALS: mol.ecp = 'def2-SVP'
        mol.build()

        if spin == 0:
            mf = scf.RHF(mol); scf_type = 'RHF'
        else:
            mf = scf.UHF(mol); scf_type = 'UHF'
        mf.max_cycle=300; mf.conv_tol=1e-9; mf.verbose=0
        mf.run()

        if not mf.converged:
            log.warning("UHF not converged, using estimate")
            return compute_ml2_from_existing(d)

        # Get density matrix and overlap
        dm  = mf.make_rdm1()
        ovp = mol.intor('int1e_ovlp')

        # Mulliken population analysis — proper metal charge
        ao_labels = mol.ao_labels()
        if scf_type == 'RHF':
            ps = dm @ ovp
        else:
            ps = (dm[0] + dm[1]) @ ovp

        # Metal AO indices
        metal_aos = [i for i,l in enumerate(ao_labels)
                     if metal in l]
        metal_pop = sum(ps[i,i] for i in metal_aos)

        # Metal formal electrons (core + valence)
        metal_z   = ATOM_Z.get(metal, 0)
        # With ECP, core electrons removed
        core_e    = 28 if metal in {'Mo','Ru','Rh','Pd'} else \
                    60 if metal in {'W','Re','Os','Ir','Pt'} else 0
        metal_ref = metal_z - core_e
        nat_charge = float(metal_ref - metal_pop)

        # d-orbital character of frontier MOs
        d_aos = [i for i,l in enumerate(ao_labels)
                 if metal in l and ('3d' in l or '4d' in l or '5d' in l)]

        if scf_type == 'RHF':
            mo_c  = mf.mo_occ
            occ_i = np.where(mf.mo_occ > 0.5)[0]
            vir_i = np.where(mf.mo_occ < 0.5)[0]
            if len(occ_i) and len(d_aos):
                homo_coeff = mf.mo_coeff[:, occ_i[-1]]
                homo_d     = float(np.sum(homo_coeff[d_aos]**2))
            else:
                homo_d = 0.5
            if len(vir_i) and len(d_aos):
                lumo_coeff = mf.mo_coeff[:, vir_i[0]]
                lumo_d     = float(np.sum(lumo_coeff[d_aos]**2))
            else:
                lumo_d = 0.5
        else:
            occ_a = np.where(mf.mo_occ[0] > 0.5)[0]
            vir_a = np.where(mf.mo_occ[0] < 0.5)[0]
            if len(occ_a) and len(d_aos):
                homo_coeff = mf.mo_coeff[0][:, occ_a[-1]]
                homo_d     = float(np.sum(homo_coeff[d_aos]**2))
            else:
                homo_d = 0.5
            if len(vir_a) and len(d_aos):
                lumo_coeff = mf.mo_coeff[0][:, vir_a[0]]
                lumo_d     = float(np.sum(lumo_coeff[d_aos]**2))
            else:
                lumo_d = 0.5

        # t2g-eg splitting for octahedral complexes
        geom = d.get('geometry','')
        if 'oct' in geom and len(d_aos) >= 5:
            # Get d-orbital energies
            if scf_type == 'RHF':
                d_ener = [mf.mo_energy[i] for i in
                          np.argsort([np.sum(mf.mo_coeff[d_aos,i]**2)
                                      for i in range(len(mf.mo_energy))])[-5:]]
            else:
                em = (mf.mo_energy[0]+mf.mo_energy[1])/2
                d_ener = sorted([em[i] for i in
                          np.argsort([np.sum(((mf.mo_coeff[0][d_aos,i]**2+
                                               mf.mo_coeff[1][d_aos,i]**2)/2)
                                      for i in range(len(em)))])[-5:]])
            if len(d_ener) >= 5:
                t2g_eg = float((d_ener[-1]-d_ener[0])*27.2114)
            else:
                t2g_eg = 0.0
        else:
            t2g_eg = 0.0

        # Get estimate features and override with UHF values
        feats = compute_ml2_from_existing(d)
        feats['metal_nat_charge'] = nat_charge
        feats['homo_d_char']      = min(1.0, homo_d)
        feats['lumo_d_char']      = min(1.0, lumo_d)
        feats['t2g_eg_split_eV']  = t2g_eg
        feats['uhf_rerun']        = True
        return feats

    except Exception as e:
        log.warning(f"UHF failed ({e}), using estimate")
        feats = compute_ml2_from_existing(d)
        feats['uhf_rerun'] = False
        return feats


# ══════════════════════════════════════════════════════════════
# JOB MANAGEMENT
# ══════════════════════════════════════════════════════════════

def build_job_list():
    """Build list of all JSON files needing ML2 features."""
    jobs = []
    for folder in TRAIN_FOLDERS:
        path = os.path.join(BASE, folder)
        if not os.path.exists(path): continue
        for f in sorted(glob.glob(f'{path}/*.json')):
            try:
                d = json.load(open(f))
                if not isinstance(d, dict): continue
                if d.get('status') != 'ok': continue
                if d.get('n_active', -1) < 0: continue
                # Check if ML2 already done
                rel  = os.path.relpath(f, BASE)
                out  = os.path.join(OUT_DIR,
                                    rel.replace('/', '_').replace('.json','_ml2.json'))
                if not os.path.exists(out):
                    jobs.append((f, out, folder))
            except: pass
    return jobs


def run_one(idx, use_uhf=False):
    """Process one file — estimate ML2 features (fast) or rerun UHF (accurate)."""
    jobs = build_job_list()
    if idx >= len(jobs):
        log.info(f"Index {idx} >= total jobs {len(jobs)} — done")
        return True

    src_file, out_file, folder = jobs[idx]
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    try:
        d = json.load(open(src_file))
        log.info(f"Processing: {os.path.basename(src_file)}")

        # Compute ML2 features
        if use_uhf and d.get('atom_str'):
            ml2 = compute_ml2_with_uhf(d)
        else:
            ml2 = compute_ml2_from_existing(d)

        # Merge ML1 + ML2 features
        result = dict(d)  # copy all ML1 features
        result['ml2_features'] = ml2
        result['ml2_computed'] = True

        # Also add flat ML2 features for easy access
        for k, v in ml2.items():
            result[f'ml2_{k}'] = v

        with open(out_file, 'w') as f:
            json.dump(result, f, indent=2)
        log.info(f"  Saved: {os.path.basename(out_file)}")
        return True

    except Exception as e:
        log.error(f"Error {src_file}: {e}")
        json.dump({'source': src_file, 'status': 'error', 'reason': str(e)},
                  open(out_file, 'w'))
        return False


def estimate_all_fast():
    """
    FAST MODE: Estimate ML2 features for ALL files without rerunning UHF.
    Completes in ~10 minutes for 6,000+ files.
    No HPC needed — runs on login node.
    """
    from collections import defaultdict
    jobs = build_job_list()
    log.info(f"Fast estimation for {len(jobs)} files")

    done = 0; failed = 0
    by_folder = defaultdict(int)

    for src_file, out_file, folder in jobs:
        try:
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            d = json.load(open(src_file))
            ml2 = compute_ml2_from_existing(d)
            result = dict(d)
            result['ml2_features'] = ml2
            result['ml2_computed'] = True
            result['ml2_method']   = 'estimate'
            for k,v in ml2.items():
                result[f'ml2_{k}'] = v
            with open(out_file,'w') as f: json.dump(result,f,indent=2)
            done += 1; by_folder[folder] += 1
            if done % 500 == 0:
                log.info(f"  Progress: {done}/{len(jobs)}")
        except Exception as e:
            failed += 1
            log.warning(f"  Failed: {src_file}: {e}")

    log.info(f"\nDone: {done}  Failed: {failed}")
    log.info("By folder:")
    for folder, n in sorted(by_folder.items()):
        log.info(f"  {folder}: {n}")
    return done


def show_summary():
    jobs = build_job_list()
    done_files = glob.glob(OUT_DIR+'/**/*_ml2.json', recursive=True)
    done_ok    = sum(1 for f in done_files
                     if json.load(open(f)).get('ml2_computed'))

    print(f"\nML2 Feature Extraction Summary")
    print(f"{'='*50}")
    print(f"Total files to process: {len(jobs) + done_ok}")
    print(f"Already done:           {done_ok}")
    print(f"Remaining:              {len(jobs)}")
    print(f"\nOutput dir: {OUT_DIR}")
    print(f"\nML2 new features (13 total):")
    feats = [
        ('metal_ox_state',      'Estimated metal oxidation state'),
        ('d_count_corrected',   'Correct d-count using ligand charges'),
        ('per_ligand_charge',   'Charge of each ligand'),
        ('sigma_pi_ratio',      'Sigma vs pi donor character'),
        ('coord_class_ml2',     'Coordination class (LF-aware)'),
        ('n_coord_effective',   'Effective coordination number'),
        ('ligand_field_split_eV','Ligand field splitting in eV'),
        ('homo_d_char',         'd-orbital character of HOMO'),
        ('lumo_d_char',         'd-orbital character of LUMO'),
        ('n_frac_strong',       'Strongly fractional NOs (0.1-1.9)'),
        ('n_frac_weak',         'Weakly fractional NOs'),
        ('metal_nat_charge',    'Natural charge on metal'),
        ('spin_per_d_electron', 'Spin / d-electron count'),
    ]
    for name, desc in feats:
        print(f"  ml2_{name:<25} {desc}")

    print(f"\nUsage:")
    print(f"  Fast (no UHF):  python gen_ml2_features.py estimate_all")
    print(f"  Batch (UHF):    sbatch --array=0-{len(jobs)-1} job_ml2_features.sh")
    print(f"  Single test:    python gen_ml2_features.py 0")


def verify_ml2_features():
    """Check a few files to verify ML2 features are correct."""
    done_files = glob.glob(OUT_DIR+'/**/*_ml2.json', recursive=True)
    if not done_files:
        print("No ML2 files found. Run estimate_all first.")
        return

    print(f"\nVerification — checking 5 random files:")
    import random
    for f in random.sample(done_files, min(5, len(done_files))):
        d = json.load(open(f))
        print(f"\n  {d.get('name','?')}")
        print(f"    metal={d.get('metal')} ligand={d.get('ligand')} "
              f"charge={d.get('charge')} spin={d.get('spin')}")
        print(f"    n_active={d.get('n_active')} "
              f"d_count_ml1={d.get('d_electron_count')} "
              f"d_count_ml2={d.get('ml2_d_count_corrected','?')}")
        print(f"    metal_ox={d.get('ml2_metal_ox_state','?')} "
              f"sigma_pi={d.get('ml2_sigma_pi_ratio','?'):.2f} "
              f"n_frac_strong={d.get('ml2_n_frac_strong','?')}")


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'summary'

    if cmd == 'summary':
        show_summary()

    elif cmd == 'estimate_all':
        # Fast mode — no UHF, runs on login node in ~10 min
        log.info("Fast estimation mode — no UHF rerun")
        n = estimate_all_fast()
        log.info(f"Completed: {n} files")
        verify_ml2_features()

    elif cmd == 'verify':
        verify_ml2_features()

    else:
        # SLURM array mode — one file per job
        try:
            idx = int(cmd)
            # Use UHF rerun for better accuracy in batch mode
            sys.exit(0 if run_one(idx, use_uhf=True) else 1)
        except ValueError:
            print(f"Unknown command: {cmd}")
            print("Usage: estimate_all | verify | summary | <int>")
            sys.exit(1)
