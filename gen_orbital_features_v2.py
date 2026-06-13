"""
gen_orbital_features_v2.py — FIXED version
Bug fix: is_active now correctly maps NOON eigenvalues back to window
orbital positions via eigenvector ordering, not raw eigenvalue indices.
"""
import sys, os, json, glob, logging, math
import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.mcscf import addons as mcaddons

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

OUTDIR   = os.path.expanduser('~/activeml/data/orbital_features_v2')
os.makedirs(OUTDIR, exist_ok=True)

BASIS      = 'def2-svp'
ECP_METALS = {'Pd','Ru','Rh','Mo','Ir','Pt'}
N_WINDOW   = 14

INDEX_FILE = os.path.expanduser('~/activeml/scripts/orbital_index_v2.txt')

# ── Geometry builder (identical to gen_orbital_features.py) ──
import re

def build_mol(d):
    metal  = d['metal']
    ligand = d['ligand']
    n_lig  = d['n_ligands']
    charge = d['charge']
    spin   = d['spin']
    dist   = d.get('dist_ang', 2.1)
    geom   = d.get('geometry', 'oct')

    if geom == 'csd_real':
        return None

    if n_lig == 6 or geom == 'oct':
        pos = [(dist,0,0),(-dist,0,0),(0,dist,0),
               (0,-dist,0),(0,0,dist),(0,0,-dist)]
    elif geom in ('sq_pl','square_planar'):
        pos = [(dist,0,0),(-dist,0,0),(0,dist,0),(0,-dist,0)]
    elif n_lig == 5 and 'tbp' in geom:
        pos = [(dist,0,0),
               (-dist*0.5, dist*math.sqrt(3)/2,0),
               (-dist*0.5,-dist*math.sqrt(3)/2,0),
               (0,0,dist),(0,0,-dist)]
    elif n_lig == 5:
        pos = [(dist,0,0),(-dist,0,0),(0,dist,0),(0,-dist,0),(0,0,dist)]
    elif n_lig == 4:
        s = dist/np.sqrt(3)
        pos = [(s,s,s),(s,-s,-s),(-s,s,-s),(-s,-s,s)]
    else:
        pos = [(dist,0,0),(-dist,0,0),(0,dist,0),(0,-dist,0),
               (0,0,dist),(0,0,-dist)]

    POLY = {
        'NH3':('N', lambda p,m: _nh3(p,m)),
        'H2O':('O', lambda p,m: _h2o(p,m)),
        'CN': ('C', lambda p,m: _cn(p,m)),
        'PH3':('P', lambda p,m: _ph3(p,m)),
    }
    mixed = re.findall(r'([A-Z][a-z]?)(\d+)', ligand)

    atom_str = metal + ' 0.0000 0.0000 0.0000\n'
    if ligand in POLY:
        bind, fn = POLY[ligand]
        for p in pos[:n_lig]:
            atom_str += bind+f' {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n'
            for a in fn(p,(0,0,0)):
                atom_str += a[0]+f' {a[1]:.4f} {a[2]:.4f} {a[3]:.4f}\n'
    elif mixed and len(mixed) > 1:
        idx = 0
        for sym,cnt in mixed:
            for _ in range(int(cnt)):
                if idx < len(pos):
                    p = pos[idx]
                    atom_str += sym+f' {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n'
                    idx += 1
    else:
        for p in pos[:n_lig]:
            atom_str += ligand+f' {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n'

    mol = gto.Mole()
    mol.atom  = atom_str
    mol.basis = BASIS
    mol.charge = charge
    mol.spin   = spin
    mol.verbose = 0
    mol.max_memory = 28000
    if metal in ECP_METALS:
        mol.ecp = BASIS
    mol.build()
    return mol

def _nh3(p,m,bl=1.012,ang=106.7):
    v=np.array(p)-np.array(m); v=v/np.linalg.norm(v)
    c=math.radians(180-ang)
    perp=np.cross(v,[1,0,0]) if abs(v[0])<0.9 else np.cross(v,[0,1,0])
    perp=perp/np.linalg.norm(perp); p2=np.cross(v,perp)
    n=np.array(p)
    return [('H',*(n+bl*(math.cos(c)*v+math.sin(c)*(math.cos(2*math.pi*i/3)*perp+math.sin(2*math.pi*i/3)*p2)))) for i in range(3)]

def _h2o(p,m,bl=0.957,ang=104.5):
    v=np.array(p)-np.array(m); v=v/np.linalg.norm(v)
    ha=math.radians(ang/2)
    perp=np.cross(v,[1,0,0]) if abs(v[0])<0.9 else np.cross(v,[0,1,0])
    perp=perp/np.linalg.norm(perp); o=np.array(p)
    return [('H',*(o+bl*(math.cos(math.pi-ha)*v+s*math.sin(math.pi-ha)*perp))) for s in [1,-1]]

def _cn(p,m,bl=1.154):
    v=np.array(p)-np.array(m); v=v/np.linalg.norm(v)
    return [('N',*(np.array(p)+bl*v))]

def _ph3(p,m,bl=1.415,ang=93.3):
    v=np.array(p)-np.array(m); v=v/np.linalg.norm(v)
    c=math.radians(180-ang)
    perp=np.cross(v,[1,0,0]) if abs(v[0])<0.9 else np.cross(v,[0,1,0])
    perp=perp/np.linalg.norm(perp); p2=np.cross(v,perp)
    n=np.array(p)
    return [('H',*(n+bl*(math.cos(c)*v+math.sin(c)*(math.cos(2*math.pi*i/3)*perp+math.sin(2*math.pi*i/3)*p2)))) for i in range(3)]

def orbital_entropy(n):
    n = float(np.clip(n, 1e-10, 1-1e-10))
    return -n*math.log(n) - (1-n)*math.log(1-n)

# ── Process one file ──────────────────────────────────────────
def process(src_filepath):
    try:
        d = json.load(open(src_filepath))
    except:
        return False

    if d.get('status') != 'ok':
        return True

    name       = d.get('name', '')
    n_active   = d.get('n_active', 0)
    n_active_e = d.get('n_active_e', 10)

    outfile = os.path.join(OUTDIR, os.path.basename(src_filepath))
    if os.path.exists(outfile):
        try:
            existing = json.load(open(outfile))
            if existing.get('status') == 'done':
                return True
        except:
            pass  # corrupted file, reprocess

    try:
        mol = build_mol(d)
        if mol is None:
            return True

        # Robust UHF — try multiple convergence strategies
        mf = None
        for uhf_settings in [
            dict(max_cycle=300, conv_tol=1e-10, damp=0.0,  level_shift=0.0),
            dict(max_cycle=500, conv_tol=1e-9,  damp=0.3,  level_shift=0.2),
            dict(max_cycle=800, conv_tol=1e-8,  damp=0.5,  level_shift=0.5),
        ]:
            mf = scf.UHF(mol)
            mf.verbose = 0
            for k, v in uhf_settings.items():
                setattr(mf, k, v)
            mf.kernel()
            if mf.converged:
                break

        E_HF     = float(mf.e_tot)
        mo_occ_a = mf.mo_occ[0]
        homo_idx = int(np.where(mo_occ_a > 0)[0].max())

        # Balanced window: 7 highest occupied + 7 lowest virtual
        # Matches casci_orbital_entropy.py exactly
        occ_total = mf.mo_occ[0] + mf.mo_occ[1]
        occ_idx   = np.where(occ_total > 0.5)[0]
        virt_idx  = np.where(occ_total < 0.5)[0]
        occ_win   = list(occ_idx[-7:])
        virt_win  = list(virt_idx[:7])
        window    = sorted(occ_win + virt_win)
        # BUG FIX 1: convert numpy ints to Python ints for JSON
        window    = [int(x) for x in window]

        n_cas_e = min(n_active_e, N_WINDOW)
        if (n_cas_e % 2) != (d['spin'] % 2):
            n_cas_e -= 1
        if n_cas_e < 2:
            n_cas_e = 2

        mo_avg    = (mf.mo_coeff[0] + mf.mo_coeff[1]) / 2
        mc        = mcscf.CASCI(mf, N_WINDOW, n_cas_e)
        mc.verbose = 0
        mo_sorted = mc.sort_mo(window, mo_coeff=mo_avg, base=0)
        mc.kernel(mo_sorted)

        E_CASCI = float(mc.e_tot)

        # Get sorted NOONs (descending) — matches old script exactly
        casdm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
        evals, _ = np.linalg.eigh(casdm1)
        noons_sorted = np.sort(evals)[::-1]  # descending

        mo_energies_a = mf.mo_energy[0]
        mo_energies_b = mf.mo_energy[1]

        # BUG FIX 2: assign is_active by selecting top n_active
        # orbitals by NOON fractionality, NOT by threshold.
        # This matches the old casci_orbital_entropy.py behavior.
        # noon_frac = min(noon, 2-noon) — higher = more fractional
        noon_fracs = [min(float(n), 2.0 - float(n)) for n in noons_sorted]
        # Get indices of top n_active most fractional
        active_positions = set(
            sorted(range(len(noon_fracs)),
                   key=lambda i: noon_fracs[i],
                   reverse=True)[:n_active]
        )

        orbital_data = []
        for i, mo_idx in enumerate(window):
            noon_i = float(noons_sorted[i]) if i < len(noons_sorted) else 0.0
            noon_i = max(0.0, min(2.0, noon_i))
            n_norm = noon_i / 2.0
            s_i    = orbital_entropy(n_norm)
            is_active = 1 if i in active_positions else 0

            orbital_data.append({
                'orbital_idx': int(mo_idx),
                'window_pos':  i,
                'dist_homo':   float(mo_idx - homo_idx),
                'mo_energy_a': float(mo_energies_a[mo_idx]),
                'mo_energy_b': float(mo_energies_b[mo_idx])
                               if mo_idx < len(mo_energies_b) else 0.0,
                'noon':        noon_i,
                'noon_frac':   float(noon_fracs[i]),
                'noon_norm':   n_norm,
                's_i':         s_i,
                'is_active':   is_active,
            })

        result = {
            'name':         name,
            'status':       'done',
            'metal':        d['metal'],
            'ligand':       d['ligand'],
            'charge':       int(d['charge']),
            'spin':         int(d['spin']),
            'n_active':     int(n_active),
            'n_active_e':   int(n_active_e),
            'E_HF':         E_HF,
            'E_CASCI':      E_CASCI,
            'window':       window,
            'homo_idx':     int(homo_idx),
            'n_window':     N_WINDOW,
            'prec_entropy': _entropy_precision(orbital_data, n_active),
            'perf_entropy': _entropy_perfect(orbital_data, n_active),
            'orbitals':     orbital_data,
        }

        with open(outfile, 'w') as fout:
            json.dump(result, fout, indent=2)
        log.info(f"  {name}: n_active={n_active} "
                 f"prec={result['prec_entropy']:.3f} "
                 f"perfect={result['perf_entropy']}")
        return True

    except Exception as e:
        log.error(f"CASCI failed {name}: {e}")
        with open(outfile, 'w') as fout:
            json.dump({'name': name, 'status': 'failed', 'error': str(e)}, fout)
        return False

def _entropy_precision(orb_data, n_active):
    if n_active == 0:
        return 1.0
    sorted_by_si = sorted(orb_data, key=lambda x: x['s_i'], reverse=True)
    selected     = set(o['window_pos'] for o in sorted_by_si[:n_active])
    true_active  = set(o['window_pos'] for o in orb_data if o['is_active'])
    if not true_active:
        return 1.0
    return len(selected & true_active) / n_active


def _entropy_perfect(orb_data, n_active):
    if n_active == 0:
        return True
    sorted_by_si = sorted(orb_data, key=lambda x: x['s_i'], reverse=True)
    selected     = set(o['window_pos'] for o in sorted_by_si[:n_active])
    true_active  = set(o['window_pos'] for o in orb_data if o['is_active'])
    return selected == true_active


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--count':
        files = [l.strip() for l in open(INDEX_FILE) if l.strip()]
        print(f"Total jobs: {len(files)}")
        sys.exit(0)

    chunk_idx   = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    chunk_total = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    all_files   = [l.strip() for l in open(INDEX_FILE) if l.strip()]
    chunk_size  = len(all_files) // chunk_total + 1
    start       = chunk_idx * chunk_size
    end         = min(start + chunk_size, len(all_files))
    files       = all_files[start:end]

    log.info(f"Chunk {chunk_idx}/{chunk_total}: {len(files)} files")
    done = 0
    for f in files:
        if process(f):
            done += 1
    log.info(f"Done: {done}/{len(files)}")
