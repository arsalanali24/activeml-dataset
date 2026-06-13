"""
gen_cod_phosphine_porphyrin.py
==============================
COD REST API pipeline for:
  Priority 2: Real experimental phosphine complexes
    Target: Pd, Rh, Ir, Ru with P-donor ligands
    ~15 structures per metal = ~60 total
    Output: ~/activeml/data/generated_cod_phosphine/

  Priority 3: Real experimental porphyrin/heme structures
    Target: Fe-porphyrin CIF files from COD
    ~40-60 structures
    Output: ~/activeml/data/generated_cod_porphyrin/

Why real geometry matters here:
  - Phosphine complexes have significant M-P-C bond angle variation
    that model geometries cannot capture
  - Fe-porphyrin has Jahn-Teller distortions, saddling, ruffling
    that affect orbital ordering
  - Real geometries validate the simplified model approach

CIF parsing strategy:
  Phosphine: find M + P atoms within 2.8 Ang of metal
             P-C bonds ignored (only P donor position matters)
  Porphyrin: find M + 4 N atoms in square plane within 2.2 Ang
             Axial ligands at > 2.2 Ang also included

Usage:
  python gen_cod_phosphine_porphyrin.py query_phosphine
  python gen_cod_phosphine_porphyrin.py query_porphyrin
  python gen_cod_phosphine_porphyrin.py download_phosphine
  python gen_cod_phosphine_porphyrin.py download_porphyrin
  python gen_cod_phosphine_porphyrin.py summary
  python gen_cod_phosphine_porphyrin.py run_phosphine <idx>
  python gen_cod_phosphine_porphyrin.py run_porphyrin <idx>
"""
import numpy as np, json, os, sys, glob, time, logging, re, math
import urllib.request, urllib.parse
from pyscf import gto, scf, mcscf

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

BASE_DIR   = os.path.expanduser('~/activeml/data')
CIF_DIR_PH = os.path.join(BASE_DIR, 'cod_cifs_phosphine')
CIF_DIR_PP = os.path.join(BASE_DIR, 'cod_cifs_porphyrin')
OUT_DIR_PH = os.path.join(BASE_DIR, 'generated_cod_phosphine')
OUT_DIR_PP = os.path.join(BASE_DIR, 'generated_cod_porphyrin')
CAND_PH    = os.path.join(CIF_DIR_PH, 'candidates.json')
CAND_PP    = os.path.join(CIF_DIR_PP, 'candidates.json')

for d in [CIF_DIR_PH, CIF_DIR_PP, OUT_DIR_PH, OUT_DIR_PP]:
    os.makedirs(d, exist_ok=True)

# ── Metal constants ───────────────────────────────────────────
METAL_CONSTANTS = {
    'Fe': {'z_eff':11.18,'zeta_so_cm1': 460,'metal_row':'3d'},
    'Ru': {'z_eff':12.33,'zeta_so_cm1': 880,'metal_row':'4d'},
    'Rh': {'z_eff':12.67,'zeta_so_cm1':1097,'metal_row':'4d'},
    'Pd': {'z_eff':13.00,'zeta_so_cm1':1334,'metal_row':'4d'},
    'Ir': {'z_eff':17.00,'zeta_so_cm1':3909,'metal_row':'5d'},
    'Pt': {'z_eff':17.33,'zeta_so_cm1':4146,'metal_row':'5d'},
    'Mn': {'z_eff':10.53,'zeta_so_cm1': 355,'metal_row':'3d'},
    'Co': {'z_eff':12.00,'zeta_so_cm1': 533,'metal_row':'3d'},
}
ECP_METALS = {'Ru','Rh','Pd','Ir','Pt'}
METAL_Z    = {'Fe':26,'Ru':44,'Rh':45,'Pd':46,'Ir':77,'Pt':78,
              'Mn':25,'Co':27}
LIG_Z      = {'P':15,'N':7,'Cl':17,'Br':35,'F':9,'O':8,'C':6}
ALLOWED    = {'P','N','Cl','Br','F','O','S','C','H'}

PHOSPHINE_TARGETS = {'Pd':15,'Rh':15,'Ir':15,'Ru':15}
PORPHYRIN_TARGETS = {'Fe':40,'Mn':10,'Co':10}

SPIN_MAP = {
    'Fe':[0,2,4],'Ru':[0,2],'Rh':[0,1],'Pd':[0],
    'Ir':[0,2],'Pt':[0],'Mn':[1,3,5],'Co':[1,3],
}


# ══════════════════════════════════════════════════════════════
# COD QUERY
# ══════════════════════════════════════════════════════════════

def query_cod(metal, max_results=500):
    base   = "https://www.crystallography.net/cod/result.php"
    params = {'el1': metal, 'nel': '1', 'format': 'json'}
    url    = base + '?' + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read().decode())
        log.info(f"  {metal}: {len(data)} raw results")
        return data
    except Exception as e:
        log.error(f"  Query failed {metal}: {e}"); return []


def filter_phosphine(entry, metal):
    """Filter for phosphine-containing complexes."""
    try:
        if float(entry.get('Rall', 1.0)) > 0.07: return False
    except: return False
    formula = (entry.get('formula') or '').strip()
    if metal not in formula: return False
    if 'P' not in formula: return False
    # Must have P in formula (phosphine donor)
    elements = set(re.findall(r'[A-Z][a-z]?', formula))
    elements.discard(metal)
    # No other transition metals
    other_tm = {'Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
                'Mo','Ru','Rh','Pd','Ir','Pt','W','Re','Os'}
    other_tm.discard(metal)
    if elements & other_tm: return False
    # Carbon count reasonable (ligand scaffold)
    c_m = re.search(r'C(\d+)', formula)
    if c_m and int(c_m.group(1)) > 30: return False
    return True


def filter_porphyrin(entry, metal):
    """Filter for porphyrin-type complexes (N4 coordination)."""
    try:
        if float(entry.get('Rall', 1.0)) > 0.07: return False
    except: return False
    formula = (entry.get('formula') or '').strip()
    if metal not in formula: return False
    if 'N' not in formula: return False
    # Porphyrin has significant carbon content
    c_m = re.search(r'C(\d+)', formula)
    n_m = re.search(r'N(\d+)', formula)
    if not c_m or not n_m: return False
    n_c = int(c_m.group(1)); n_n = int(n_m.group(1))
    # Porphyrin has C20N4 core — require at least N>=4 and C>=10
    if n_n < 4 or n_c < 10: return False
    # No other transition metals
    elements = set(re.findall(r'[A-Z][a-z]?', formula))
    elements.discard(metal)
    other_tm = {'Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
                'Mo','Ru','Rh','Pd','Ir','Pt'}
    other_tm.discard(metal)
    if elements & other_tm: return False
    return True


def query_all_phosphine():
    candidates = []
    for metal, target in PHOSPHINE_TARGETS.items():
        log.info(f"\nQuerying COD for {metal} phosphine...")
        results = query_cod(metal)
        time.sleep(1)
        good = [r for r in results if filter_phosphine(r, metal)]
        log.info(f"  {metal}: {len(good)} phosphine candidates")
        for r in good[:target*4]: r['target_metal'] = metal
        candidates.extend(good[:target*4])
    with open(CAND_PH, 'w') as f: json.dump(candidates, f, indent=2)
    from collections import Counter
    by_m = Counter(c['target_metal'] for c in candidates)
    log.info(f"\nTotal phosphine candidates: {len(candidates)}")
    for m, n in sorted(by_m.items()): log.info(f"  {m}: {n}")


def query_all_porphyrin():
    candidates = []
    for metal, target in PORPHYRIN_TARGETS.items():
        log.info(f"\nQuerying COD for {metal} porphyrin...")
        results = query_cod(metal)
        time.sleep(1)
        good = [r for r in results if filter_porphyrin(r, metal)]
        log.info(f"  {metal}: {len(good)} porphyrin candidates")
        for r in good[:target*3]: r['target_metal'] = metal
        candidates.extend(good[:target*3])
    with open(CAND_PP, 'w') as f: json.dump(candidates, f, indent=2)
    from collections import Counter
    by_m = Counter(c['target_metal'] for c in candidates)
    log.info(f"\nTotal porphyrin candidates: {len(candidates)}")
    for m, n in sorted(by_m.items()): log.info(f"  {m}: {n}")


# ══════════════════════════════════════════════════════════════
# CIF DOWNLOAD
# ══════════════════════════════════════════════════════════════

def download_cifs(cand_file, cif_dir):
    if not os.path.exists(cand_file):
        log.error(f"Run query first: {cand_file}"); return
    candidates = json.load(open(cand_file))
    log.info(f"Downloading {len(candidates)} CIF files...")
    ok = failed = 0
    for i, c in enumerate(candidates):
        metal     = c['target_metal']
        metal_dir = os.path.join(cif_dir, metal)
        os.makedirs(metal_dir, exist_ok=True)
        cif_path  = os.path.join(metal_dir, f"{c['file']}.cif")
        if os.path.exists(cif_path) and os.path.getsize(cif_path) > 100:
            ok += 1; continue
        try:
            url = f"https://www.crystallography.net/cod/{c['file']}.cif"
            with urllib.request.urlopen(url, timeout=20) as r:
                with open(cif_path, 'wb') as f: f.write(r.read())
            ok += 1; time.sleep(0.4)
        except Exception as e:
            failed += 1
            log.error(f"  Failed {c['file']}: {e}")
        if (i+1) % 20 == 0:
            log.info(f"  {i+1}/{len(candidates)} ok={ok} failed={failed}")
    log.info(f"Done: {ok} ok, {failed} failed")


# ══════════════════════════════════════════════════════════════
# CIF PARSER — phosphine version
# Finds M + P donors (and other simple ligand atoms)
# ══════════════════════════════════════════════════════════════

def parse_cif(cif_path, metal, mode='phosphine'):
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
    al=math.radians(alpha); be=math.radians(beta); ga=math.radians(gamma)
    cos_al=math.cos(al); cos_be=math.cos(be); cos_ga=math.cos(ga)
    sin_ga=math.sin(ga)
    v=math.sqrt(max(0,1-cos_al**2-cos_be**2-cos_ga**2+2*cos_al*cos_be*cos_ga))
    M=np.array([[a,b*cos_ga,c*cos_be],
                [0,b*sin_ga,c*(cos_al-cos_be*cos_ga)/max(sin_ga,1e-10)],
                [0,0,c*v/max(sin_ga,1e-10)]])

    loop_m = re.search(
        r'loop_\s*((?:_atom_site_\S+\s*)+)((?:(?!loop_|_\w)[\s\S])*)',
        content)
    if not loop_m: return None
    headers = re.findall(r'_atom_site_\S+', loop_m.group(1))
    col     = {h: i for i, h in enumerate(headers)}
    block   = loop_m.group(2)

    use_frac   = '_atom_site_fract_x' in col
    label_col  = col.get('_atom_site_label',
                          col.get('_atom_site_type_symbol', None))
    if label_col is None or not use_frac: return None

    atoms = []; metal_pos_list = []
    for line in block.strip().split('\n'):
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
            if sym == metal: metal_pos_list.append(len(atoms)-1)
        except: continue

    if not atoms or not metal_pos_list: return None
    metal_idx  = metal_pos_list[0]
    metal_sym, metal_pos = atoms[metal_idx]

    if mode == 'phosphine':
        # Include P donors AND simple ligand atoms (Cl, N, O etc)
        # but NOT C or H (we don't want the phosphine carbon chain)
        include_syms = {'P', 'Cl', 'Br', 'F', 'N', 'O', 'S'}
        max_dist = 2.8
    else:  # porphyrin
        # Include N donors (porphyrin) and axial ligands
        include_syms = {'N', 'Cl', 'Br', 'O', 'S'}
        max_dist = 2.4

    coord_atoms = [(metal_sym, metal_pos)]
    for i, (sym, pos) in enumerate(atoms):
        if i == metal_idx or sym == 'H': continue
        dist = float(np.linalg.norm(pos - metal_pos))
        if sym in include_syms and 1.5 <= dist <= max_dist:
            coord_atoms.append((sym, pos))

    n_lig = len(coord_atoms) - 1
    if n_lig < 2 or n_lig > 8: return None

    # For phosphine mode: must have at least one P
    if mode == 'phosphine':
        has_p = any(s == 'P' for s, _ in coord_atoms[1:])
        if not has_p: return None

    # For porphyrin mode: must have at least 4 N
    if mode == 'porphyrin':
        n_count = sum(1 for s, _ in coord_atoms[1:] if s == 'N')
        if n_count < 4: return None

    center   = coord_atoms[0][1].copy()
    atom_str = ''
    for sym, pos in coord_atoms:
        p = pos - center
        atom_str += f"{sym}  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}\n"
    return atom_str.strip(), n_lig


# ══════════════════════════════════════════════════════════════
# CASSCF PIPELINE (same as existing scripts)
# ══════════════════════════════════════════════════════════════

def get_nact(n, size=10):
    for k in [size,size-1,size+1,size-2,size+2,size-3]:
        if k>0 and (n-k)>=0 and (n-k)%2==0: return k
    return size

def run_uhf(mol):
    for s in [dict(max_cycle=300,conv_tol=1e-10,damp=0.0,level_shift=0.0),
              dict(max_cycle=500,conv_tol=1e-9, damp=0.3,level_shift=0.2),
              dict(max_cycle=800,conv_tol=1e-8, damp=0.5,level_shift=0.5)]:
        mf=scf.UHF(mol)
        for k,v in s.items(): setattr(mf,k,v)
        mf.verbose=0; mf.run()
        if mf.converged: return mf
    return mf

def run_casscf(mf, mol, n_act, cas_n=10):
    e_m=(mf.mo_energy[0]+mf.mo_energy[1])/2
    occ=mf.mo_occ[0]+mf.mo_occ[1]
    oi=np.where(occ>0.5)[0]; vi=np.where(occ<0.5)[0]
    if not len(oi) or not len(vi): return None,False
    gap=(float(e_m[oi[-1]])+float(e_m[vi[0]]))/2
    ws=cas_n+4
    w=sorted(np.argsort(np.abs(e_m-gap))[:ws],key=lambda i:e_m[i])
    best_mc=None; best_e=0.0
    for win in [w[:cas_n],w[2:cas_n+2],w[1:cas_n+1],w[4:cas_n+4]]:
        for sh in [1e-3,1e-2,5e-2,1e-1,2e-1]:
            try:
                mc=mcscf.CASSCF(mf,cas_n,n_act)
                mc.max_cycle_macro=600; mc.conv_tol=1e-8
                mc.ah_level_shift=sh; mc.verbose=0
                mc.kernel(mc.sort_mo(win,base=0))
                ec=mc.e_tot-mf.e_tot
                if ec<0 and ec<best_e: best_mc=mc; best_e=ec
                if mc.converged and ec<-0.01: return mc,True
            except: continue
    if best_mc: return best_mc,best_mc.converged
    return None,False


def run_one(cod_id, metal, atom_str, spin, out_dir,
            struct_label, lig_label, cas_n=10, mode='phosphine'):
    name    = f"COD_{struct_label}_spin{spin}"
    outfile = os.path.join(out_dir, f"{name}.json")

    if os.path.exists(outfile):
        r = json.load(open(outfile))
        if r.get('status') == 'ok':
            log.info(f"SKIP: {name}"); return True

    n_e = sum(METAL_Z.get(l.strip().split()[0],
                          LIG_Z.get(l.strip().split()[0], 0))
              for l in atom_str.split('\n') if l.strip())
    if (n_e % 2) != (spin % 2):
        json.dump({'name':name,'status':'skipped','reason':'parity'},
                  open(outfile,'w')); return True

    consts = METAL_CONSTANTS.get(metal,
             {'z_eff':10,'zeta_so_cm1':400,'metal_row':'3d'})
    log.info(f"Start: {name}  n_e={n_e}")

    try:
        mol = gto.Mole()
        mol.atom = atom_str; mol.basis = 'def2-SVP'
        mol.charge = 0; mol.spin = spin; mol.verbose = 0
        if metal in ECP_METALS: mol.ecp = 'def2-SVP'
        mol.build()

        n_act = get_nact(mol.nelectron, cas_n)
        mf    = run_uhf(mol)
        mc, conv = run_casscf(mf, mol, n_act, cas_n)

        if mc is None or mc.e_tot - mf.e_tot >= 0:
            json.dump({'name':name,'status':'failed','metal':metal,
                       'cod_id':cod_id}, open(outfile,'w')); return False

        ec = mc.e_tot - mf.e_tot
        casdm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
        no,_ = np.linalg.eigh(casdm1); no=np.sort(no)[::-1]
        n_active = sum(1 for n in no if 0.02 < n < 1.98)

        e_m=(mf.mo_energy[0]+mf.mo_energy[1])/2
        occ=mf.mo_occ[0]+mf.mo_occ[1]
        oi=np.where(occ>0.5)[0]; vi=np.where(occ<0.5)[0]
        S=spin/2.0

        result = {
            'name':name,'metal':metal,'ligand':lig_label,
            'charge':0,'spin':spin,'mult':spin+1,
            'geometry':'cod_real','cod_id':cod_id,
            'struct_name':struct_label,'mode':mode,
            'casscf_size':cas_n,'n_electrons':mol.nelectron,
            'n_active_e':n_act,'E_HF':float(mf.e_tot),
            'E_CASSCF':float(mc.e_tot),'corr_energy':float(ec),
            'converged':bool(conv),'n_active':n_active,
            'no_occ':[float(x) for x in no],'status':'ok',
            'z_eff':consts['z_eff'],'zeta_so_cm1':consts['zeta_so_cm1'],
            'metal_row':consts['metal_row'],
            'spin_contamination':float(mf.spin_square()[0]-S*(S+1)),
            'homo_lumo_gap':float(e_m[vi[0]]-e_m[oi[-1]]) if len(oi) and len(vi) else 0.0,
            'homo_energy':float(e_m[oi[-1]]) if len(oi) else 0.0,
            'lumo_energy':float(e_m[vi[0]]) if len(vi) else 0.0,
            'homo_ab_gap':0.0,
        }
        with open(outfile,'w') as f: json.dump(result,f,indent=2)
        log.info(f"  OK: n_active={n_active} Ec={ec:.4f}")
        return True
    except Exception as e:
        log.error(f"  Error: {e}")
        json.dump({'name':name,'status':'error','reason':str(e),
                   'cod_id':cod_id}, open(outfile,'w'))
        return False


# ══════════════════════════════════════════════════════════════
# JOB LIST BUILDERS
# ══════════════════════════════════════════════════════════════

def build_phosphine_jobs():
    if not os.path.exists(CAND_PH): return []
    candidates = json.load(open(CAND_PH))
    jobs = []; seen = set()
    existing = set(os.path.basename(f)
                   for f in glob.glob(f'{OUT_DIR_PH}/*.json'))
    from collections import defaultdict
    metal_count = defaultdict(int)

    for c in candidates:
        metal = c['target_metal']
        if metal_count[metal] >= PHOSPHINE_TARGETS[metal]: continue
        cod_id = c['file']
        cif_path = os.path.join(CIF_DIR_PH, metal, f"{cod_id}.cif")
        if not os.path.exists(cif_path): continue
        result = parse_cif(cif_path, metal, mode='phosphine')
        if result is None: continue
        atom_str, n_lig = result
        struct_label = f"{cod_id}_{metal}_ph"
        key = (metal, cod_id)
        if key in seen: continue
        seen.add(key)
        n_e = sum(METAL_Z.get(l.strip().split()[0],
                              LIG_Z.get(l.strip().split()[0],0))
                  for l in atom_str.split('\n') if l.strip())
        added = False
        for spin in SPIN_MAP.get(metal, [0,2]):
            if (n_e%2)!=(spin%2): continue
            fname = f"COD_{struct_label}_spin{spin}.json"
            if fname in existing: continue
            jobs.append((cod_id, metal, atom_str, spin,
                         OUT_DIR_PH, struct_label, 'PH3', 10, 'phosphine'))
            added = True
        if added: metal_count[metal] += 1
    return jobs


def build_porphyrin_jobs():
    if not os.path.exists(CAND_PP): return []
    candidates = json.load(open(CAND_PP))
    jobs = []; seen = set()
    existing = set(os.path.basename(f)
                   for f in glob.glob(f'{OUT_DIR_PP}/*.json'))
    from collections import defaultdict
    metal_count = defaultdict(int)

    for c in candidates:
        metal = c['target_metal']
        if metal_count[metal] >= PORPHYRIN_TARGETS[metal]: continue
        cod_id = c['file']
        cif_path = os.path.join(CIF_DIR_PP, metal, f"{cod_id}.cif")
        if not os.path.exists(cif_path): continue
        result = parse_cif(cif_path, metal, mode='porphyrin')
        if result is None: continue
        atom_str, n_lig = result
        struct_label = f"{cod_id}_{metal}_pp"
        key = (metal, cod_id)
        if key in seen: continue
        seen.add(key)
        n_e = sum(METAL_Z.get(l.strip().split()[0],
                              LIG_Z.get(l.strip().split()[0],0))
                  for l in atom_str.split('\n') if l.strip())
        added = False
        # Porphyrin uses CASSCF(14,14)
        for spin in SPIN_MAP.get(metal, [0,2,4]):
            if (n_e%2)!=(spin%2): continue
            fname = f"COD_{struct_label}_spin{spin}.json"
            if fname in existing: continue
            jobs.append((cod_id, metal, atom_str, spin,
                         OUT_DIR_PP, struct_label, 'N4', 14, 'porphyrin'))
            added = True
        if added: metal_count[metal] += 1
    return jobs


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'summary'

    if cmd == 'query_phosphine':
        query_all_phosphine()

    elif cmd == 'query_porphyrin':
        query_all_porphyrin()

    elif cmd == 'download_phosphine':
        download_cifs(CAND_PH, CIF_DIR_PH)

    elif cmd == 'download_porphyrin':
        download_cifs(CAND_PP, CIF_DIR_PP)

    elif cmd == 'summary':
        ph_jobs = build_phosphine_jobs()
        pp_jobs = build_porphyrin_jobs()
        from collections import defaultdict
        ph_by_m = defaultdict(int)
        pp_by_m = defaultdict(int)
        for j in ph_jobs: ph_by_m[j[1]] += 1
        for j in pp_jobs: pp_by_m[j[1]] += 1
        print(f"\nPhosphine COD jobs: {len(ph_jobs)}")
        for m in ['Pd','Rh','Ir','Ru']:
            print(f"  {m}: {ph_by_m.get(m,0)}")
        print(f"\nPorphyrin COD jobs: {len(pp_jobs)}")
        for m in ['Fe','Mn','Co']:
            print(f"  {m}: {pp_by_m.get(m,0)}")

    elif cmd == 'run_phosphine':
        jobs = build_phosphine_jobs()
        idx  = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        if idx < len(jobs):
            sys.exit(0 if run_one(*jobs[idx]) else 1)

    elif cmd == 'run_porphyrin':
        jobs = build_porphyrin_jobs()
        idx  = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        if idx < len(jobs):
            sys.exit(0 if run_one(*jobs[idx]) else 1)
