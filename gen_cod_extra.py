"""
gen_cod_extra.py
================
COD REST API pipeline for missing metals and higher coordination numbers.
Extends gen_cod_real.py with:
  - Ti, V, Zn (zero real geometry coverage currently)
  - 7-coordinate Mo complexes (coordination number 7)

Same pipeline as gen_cod_real.py — same JSON schema, same CASSCF(10,10).
Output: ~/activeml/data/generated_cod_extra/

Usage:
  python gen_cod_extra.py query
  python gen_cod_extra.py download
  python gen_cod_extra.py summary
  python gen_cod_extra.py <idx>
"""
import numpy as np, json, os, sys, glob, time, logging, re, math
import urllib.request, urllib.parse
from pyscf import gto, scf, mcscf

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

CIF_DIR    = os.path.expanduser('~/activeml/data/cod_cifs_extra')
OUTPUT_DIR = os.path.expanduser('~/activeml/data/generated_cod_extra')
CAND_FILE  = os.path.join(CIF_DIR, 'candidates.json')

METAL_CONSTANTS = {
    'Ti': {'z_eff':  8.14, 'zeta_so_cm1':  121, 'metal_row': '3d'},
    'V':  {'z_eff':  8.98, 'zeta_so_cm1':  208, 'metal_row': '3d'},
    'Zn': {'z_eff': 13.57, 'zeta_so_cm1': 1042, 'metal_row': '3d'},
    'Mo': {'z_eff': 10.97, 'zeta_so_cm1':  467, 'metal_row': '4d'},
}

ECP_METALS = {'Mo'}
METAL_Z    = {'Ti':22, 'V':23, 'Zn':30, 'Mo':42}
LIG_Z      = {'Cl':17, 'Br':35, 'F':9, 'N':7, 'O':8, 'S':16}
ALLOWED    = {'Cl','Br','F','N','O','S','P','H','C','B'}

TARGETS = {'Ti': 15, 'V': 15, 'Zn': 15, 'Mo': 10}

SPIN_STATES = {
    'Ti': [0, 2],
    'V':  [0, 2, 4],
    'Zn': [0],
    'Mo': [1, 3, 5],
}


def query_cod(metal, max_results=300):
    base   = "https://www.crystallography.net/cod/result.php"
    params = {'el1': metal, 'nel': '1', 'format': 'json'}
    url    = base + '?' + urllib.parse.urlencode(params)
    log.info(f"  Querying COD for {metal}")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read().decode())
    except Exception as e:
        log.error(f"  Query failed {metal}: {e}"); return []
    log.info(f"  {metal}: {len(data)} raw results")
    return data


def is_good(entry, metal):
    rall = entry.get('Rall')
    if rall is None: return False
    try:
        if float(rall) > 0.08: return False
    except: return False
    zprime = entry.get('Zprime')
    if zprime is not None:
        try:
            if float(zprime) > 1.0: return False
        except: pass
    nel = entry.get('nel')
    if nel is not None:
        try:
            if int(nel) > 5: return False
        except: pass
    formula = (entry.get('formula') or '').replace('-', '').strip()
    if not formula or metal not in formula: return False
    elements = set(re.findall(r'[A-Z][a-z]?', formula))
    elements.discard(metal)
    other_tm = {'Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
                'Mo','Tc','Ru','Rh','Pd','Ag','Cd',
                'W','Re','Os','Ir','Pt','Au','Hg'}
    other_tm.discard(metal)
    if elements & other_tm: return False
    disallowed = elements - ALLOWED - {'Na','K','Li','Ca','Mg',
                                        'Ba','Sr','Cs','Rb'}
    if disallowed: return False
    c_match = re.search(r'C(\d+)', formula or '')
    if c_match and int(c_match.group(1)) > 12: return False
    Z = entry.get('Z')
    if Z is not None:
        try:
            if int(Z) > 8: return False
        except: pass
    return True


def query_all():
    os.makedirs(CIF_DIR, exist_ok=True)
    all_candidates = []
    for metal, target in TARGETS.items():
        log.info(f"\nQuerying {metal} (target: {target})...")
        results = query_cod(metal)
        time.sleep(1)
        good = [r for r in results if is_good(r, metal)]
        log.info(f"  {metal}: {len(good)} candidates after filtering")
        selected = good[:target * 3]
        for r in selected: r['target_metal'] = metal
        all_candidates.extend(selected)
    with open(CAND_FILE, 'w') as f:
        json.dump(all_candidates, f, indent=2)
    log.info(f"\nTotal candidates: {len(all_candidates)}")
    from collections import Counter
    by_metal = Counter(c['target_metal'] for c in all_candidates)
    for m, n in sorted(by_metal.items()): log.info(f"  {m}: {n}")


def download_cif(cod_id, metal):
    metal_dir = os.path.join(CIF_DIR, metal)
    os.makedirs(metal_dir, exist_ok=True)
    cif_path  = os.path.join(metal_dir, f"{cod_id}.cif")
    if os.path.exists(cif_path) and os.path.getsize(cif_path) > 100:
        return cif_path
    url = f"https://www.crystallography.net/cod/{cod_id}.cif"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            content = r.read()
        with open(cif_path, 'wb') as f: f.write(content)
        time.sleep(0.5)
        return cif_path
    except Exception as e:
        log.error(f"  Download failed {cod_id}: {e}"); return None


def download_all():
    if not os.path.exists(CAND_FILE):
        log.error("Run query first"); return
    candidates = json.load(open(CAND_FILE))
    log.info(f"Downloading {len(candidates)} CIF files...")
    ok = failed = 0
    for i, c in enumerate(candidates):
        path = download_cif(c['file'], c['target_metal'])
        if path: ok += 1
        else: failed += 1
        if (i+1) % 20 == 0:
            log.info(f"  {i+1}/{len(candidates)} ok={ok} failed={failed}")
    log.info(f"Done: {ok} ok, {failed} failed")


def parse_cif(cif_path, metal):
    try:
        with open(cif_path, 'r', errors='ignore') as f:
            content = f.read()
    except: return None

    def get_val(key):
        m = re.search(rf'{key}\s+([\d.]+)', content)
        return float(m.group(1)) if m else None

    a = get_val('_cell_length_a'); b = get_val('_cell_length_b')
    c = get_val('_cell_length_c')
    if not all([a, b, c]): return None

    alpha = get_val('_cell_angle_alpha') or 90.0
    beta  = get_val('_cell_angle_beta')  or 90.0
    gamma = get_val('_cell_angle_gamma') or 90.0
    al = math.radians(alpha); be = math.radians(beta); ga = math.radians(gamma)
    cos_al=math.cos(al); cos_be=math.cos(be); cos_ga=math.cos(ga)
    sin_ga=math.sin(ga)
    v = math.sqrt(max(0, 1-cos_al**2-cos_be**2-cos_ga**2+2*cos_al*cos_be*cos_ga))
    M = np.array([[a, b*cos_ga, c*cos_be],
                  [0, b*sin_ga, c*(cos_al-cos_be*cos_ga)/max(sin_ga,1e-10)],
                  [0, 0,        c*v/max(sin_ga,1e-10)]])

    loop_match = re.search(
        r'loop_\s*((?:_atom_site_\S+\s*)+)((?:(?!loop_|_\w)[\s\S])*)',
        content)
    if not loop_match: return None
    headers    = re.findall(r'_atom_site_\S+', loop_match.group(1))
    col        = {h: i for i, h in enumerate(headers)}
    data_block = loop_match.group(2)

    use_frac = '_atom_site_fract_x' in col
    label_col = col.get('_atom_site_label',
                        col.get('_atom_site_type_symbol', None))
    if label_col is None or not use_frac: return None

    atoms = []; metal_positions = []
    for line in data_block.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('_') or line.startswith('#'): continue
        line = re.sub(r'\([\d]+\)', '', line)
        parts = line.split()
        if len(parts) < len(headers): continue
        try:
            sym_m = re.match(r'([A-Z][a-z]?)', parts[label_col])
            if not sym_m: continue
            sym = sym_m.group(1)
            fx  = float(parts[col['_atom_site_fract_x']])
            fy  = float(parts[col['_atom_site_fract_y']])
            fz  = float(parts[col['_atom_site_fract_z']])
            pos = M @ np.array([fx, fy, fz])
            atoms.append((sym, pos))
            if sym == metal: metal_positions.append(len(atoms)-1)
        except (ValueError, IndexError): continue

    if not atoms or not metal_positions: return None
    metal_idx = metal_positions[0]
    metal_sym, metal_pos = atoms[metal_idx]

    coord_atoms = [(metal_sym, metal_pos)]
    for i, (sym, pos) in enumerate(atoms):
        if i == metal_idx or sym == 'H': continue
        dist = float(np.linalg.norm(pos - metal_pos))
        if 1.5 <= dist <= 2.8: coord_atoms.append((sym, pos))

    n_ligands = len(coord_atoms) - 1
    if n_ligands < 2 or n_ligands > 8: return None
    if not set(s for s, _ in coord_atoms[1:]).issubset(ALLOWED): return None

    center   = coord_atoms[0][1].copy()
    atom_str = ''
    for sym, pos in coord_atoms:
        p = pos - center
        atom_str += f"{sym}  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}\n"
    return atom_str.strip(), n_ligands


def guess_charge(formula, n_ligands):
    m = re.search(r'\](\d*)([+-])', formula or '')
    if m:
        n = int(m.group(1)) if m.group(1) else 1
        return (-1 if m.group(2) == '-' else 1) * n
    cl = len(re.findall(r'Cl', formula or ''))
    br = len(re.findall(r'Br', formula or ''))
    total = cl + br
    if total == 4: return -2
    if total == 6: return -3
    if total == 2: return 0
    return 0


def build_job_list():
    if not os.path.exists(CAND_FILE): return []
    candidates = json.load(open(CAND_FILE))
    cand_by_id = {c['file']: c for c in candidates}
    jobs = []; seen = set()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing = set(os.path.basename(f)
                   for f in glob.glob(f'{OUTPUT_DIR}/*.json'))
    from collections import defaultdict
    metal_count = defaultdict(int)

    for metal in TARGETS:
        metal_dir = os.path.join(CIF_DIR, metal)
        if not os.path.exists(metal_dir): continue
        for cif_path in sorted(glob.glob(f'{metal_dir}/*.cif')):
            if metal_count[metal] >= TARGETS[metal]: break
            cod_id = os.path.basename(cif_path).replace('.cif', '')
            result = parse_cif(cif_path, metal)
            if result is None: continue
            atom_str, n_ligands = result
            key = (metal, cod_id)
            if key in seen: continue
            seen.add(key)
            cand   = cand_by_id.get(cod_id, {})
            charge = guess_charge(cand.get('formula', ''), n_ligands)
            n_e    = sum(METAL_Z.get(l.strip().split()[0],
                                     LIG_Z.get(l.strip().split()[0], 0))
                         for l in atom_str.split('\n') if l.strip()) - charge
            struct_label = f"COD{cod_id}_{metal}"
            added = False
            for spin in SPIN_STATES.get(metal, [0, 2]):
                if (n_e % 2) != (spin % 2): continue
                fname = f"COD_{struct_label}_spin{spin}.json"
                if fname in existing: continue
                jobs.append((cod_id, metal, charge,
                             atom_str, spin, struct_label))
                added = True
            if added: metal_count[metal] += 1
    return jobs


def get_nact(n_total):
    for n in [10, 9, 11, 8, 12, 7, 13, 6, 14]:
        if (n_total - n) >= 0 and (n_total - n) % 2 == 0: return n
    return 10


def run_uhf(mol):
    for s in [dict(max_cycle=300, conv_tol=1e-10, damp=0.0, level_shift=0.0),
              dict(max_cycle=500, conv_tol=1e-9,  damp=0.3, level_shift=0.2),
              dict(max_cycle=800, conv_tol=1e-8,  damp=0.5, level_shift=0.5)]:
        mf = scf.UHF(mol)
        for k, v in s.items(): setattr(mf, k, v)
        mf.verbose = 0; mf.run()
        if mf.converged: return mf
    return mf


def run_casscf(mf, mol, n_act):
    e_m = (mf.mo_energy[0]+mf.mo_energy[1])/2
    occ = mf.mo_occ[0]+mf.mo_occ[1]
    oi  = np.where(occ > 0.5)[0]; vi = np.where(occ < 0.5)[0]
    if len(oi) == 0 or len(vi) == 0: return None, False
    gap = (float(e_m[oi[-1]]) + float(e_m[vi[0]])) / 2
    w14 = sorted(np.argsort(np.abs(e_m-gap))[:14], key=lambda i: e_m[i])
    best_mc = None; best_e = 0.0
    for window in [w14[:10], w14[2:12], w14[1:11], w14[3:13]]:
        for shift in [1e-3, 1e-2, 5e-2, 1e-1]:
            try:
                mc = mcscf.CASSCF(mf, 10, n_act)
                mc.max_cycle_macro = 500; mc.conv_tol = 1e-8
                mc.ah_level_shift = shift; mc.verbose = 0
                mc.kernel(mc.sort_mo(window, base=0))
                ec = mc.e_tot - mf.e_tot
                if ec < 0 and ec < best_e: best_mc = mc; best_e = ec
                if mc.converged and ec < -0.01: return mc, True
            except: continue
    if best_mc: return best_mc, best_mc.converged
    return None, False


def run_one(cod_id, metal, charge, atom_str, spin, struct_label):
    name    = f"COD_{struct_label}_spin{spin}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outfile = os.path.join(OUTPUT_DIR, f"{name}.json")

    if os.path.exists(outfile):
        r = json.load(open(outfile))
        if r.get('converged') and r.get('corr_energy', 0) < -0.001:
            log.info(f"SKIP: {name}"); return True

    n_e = sum(METAL_Z.get(l.strip().split()[0],
                          LIG_Z.get(l.strip().split()[0], 0))
              for l in atom_str.split('\n') if l.strip()) - charge

    if (n_e % 2) != (spin % 2):
        json.dump({'name': name, 'status': 'skipped', 'reason': 'parity',
                   'geometry': 'cod_real'}, open(outfile, 'w')); return True

    consts = METAL_CONSTANTS[metal]
    log.info(f"Start: {name}  n_e={n_e}  cod={cod_id}")

    try:
        mol = gto.Mole()
        mol.atom = atom_str; mol.basis = 'def2-SVP'
        mol.charge = charge; mol.spin = spin; mol.verbose = 0
        if metal in ECP_METALS: mol.ecp = 'def2-SVP'
        mol.build()

        n_act = get_nact(mol.nelectron)
        mf    = run_uhf(mol)
        mc, conv = run_casscf(mf, mol, n_act)

        if mc is None:
            json.dump({'name': name, 'status': 'failed',
                       'geometry': 'cod_real', 'metal': metal,
                       'cod_id': cod_id}, open(outfile, 'w')); return False

        ec = mc.e_tot - mf.e_tot
        if ec >= 0:
            json.dump({'name': name, 'status': 'unphysical',
                       'corr_energy': float(ec),
                       'geometry': 'cod_real'}, open(outfile, 'w')); return False

        casdm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
        no, _  = np.linalg.eigh(casdm1); no = np.sort(no)[::-1]
        n_active = sum(1 for n in no if 0.02 < n < 1.98)

        syms = set(re.findall(r'\b([A-Z][a-z]?)\b', atom_str))
        syms.discard(metal)
        lig = 'Cl'
        for l in ['Cl','Br','F','N','O','S']: 
            if l in syms: lig = l; break

        e_m = (mf.mo_energy[0]+mf.mo_energy[1])/2
        occ = mf.mo_occ[0]+mf.mo_occ[1]
        oi  = np.where(occ > 0.5)[0]; vi = np.where(occ < 0.5)[0]
        homo_e = float(e_m[oi[-1]]) if len(oi) > 0 else 0.0
        lumo_e = float(e_m[vi[0]])  if len(vi) > 0 else 0.0
        oa = np.where(mf.mo_occ[0] > 0.5)[0]
        ob = np.where(mf.mo_occ[1] > 0.5)[0]
        homo_a = float(mf.mo_energy[0][oa[-1]]) if len(oa) > 0 else 0.0
        homo_b = float(mf.mo_energy[1][ob[-1]]) if len(ob) > 0 else 0.0
        S = spin/2.0; spin_contam = float(mf.spin_square()[0]-S*(S+1))

        result = {
            'name': name, 'metal': metal, 'ligand': lig,
            'n_ligands': atom_str.count(lig), 'charge': charge,
            'spin': spin, 'mult': spin+1, 'geometry': 'cod_real',
            'cod_id': cod_id, 'struct_name': struct_label,
            'n_electrons': mol.nelectron, 'n_active_e': n_act,
            'E_HF': float(mf.e_tot), 'E_CASSCF': float(mc.e_tot),
            'corr_energy': float(ec), 'converged': bool(conv),
            'n_active': n_active, 'no_occ': [float(x) for x in no],
            'status': 'ok', 'z_eff': consts['z_eff'],
            'zeta_so_cm1': consts['zeta_so_cm1'],
            'metal_row': consts['metal_row'],
            'spin_contamination': spin_contam,
            'homo_lumo_gap': float(lumo_e-homo_e),
            'homo_energy': homo_e, 'lumo_energy': lumo_e,
            'homo_ab_gap': float(abs(homo_a-homo_b)),
        }
        with open(outfile, 'w') as f: json.dump(result, f, indent=2)
        log.info(f"  OK: n_active={n_active} Ec={ec:.4f}")
        return True
    except Exception as e:
        log.error(f"  Error {name}: {e}")
        json.dump({'name': name, 'status': 'error', 'reason': str(e),
                   'geometry': 'cod_real', 'cod_id': cod_id},
                  open(outfile, 'w'))
        return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: query | download | summary | <idx>"); sys.exit(1)
    cmd = sys.argv[1]
    if cmd == 'query':    query_all()
    elif cmd == 'download': download_all()
    elif cmd == 'summary':
        jobs = build_job_list()
        from collections import defaultdict
        by_metal = defaultdict(int)
        for _, m, *_ in jobs: by_metal[m] += 1
        print(f"\nTotal COD extra jobs: {len(jobs)}")
        for m in ['Ti','V','Zn','Mo']:
            print(f"  {m}: {by_metal.get(m,0)}")
    else:
        jobs = build_job_list()
        idx  = int(cmd)
        if idx >= len(jobs): sys.exit(1)
        sys.exit(0 if run_one(*jobs[idx]) else 1)
