"""
gen_irc_casscf.py
=================
Computes CASSCF(14,14) labels for each IRC snapshot.
This is the Track A core result — shows active space
changes dynamically along the Pd oxidative addition pathway.

Input:  ~/activeml/data/irc_pd/snapshot_00.xyz ... snapshot_11.xyz
Output: ~/activeml/data/irc_pd/casscf_snapshot_00.json ... 

Key result for paper:
  Reactant (snap 0):  Pd(0) d10 — n_active should be small (2-4)
  Near TS  (snap 5):  Pd oxidative addition TS — n_active larger (8-12)
  Product  (snap 11): Pd(II) d8 — n_active moderate (4-8)

This dynamic variation is what AutoRXN misses by freezing
the active space at the first geometry.

Usage:
  python gen_irc_casscf.py <snap_idx>   (0-11)
  python gen_irc_casscf.py summary
  sbatch --array=0-11 job_irc_casscf.sh
"""
import numpy as np, json, os, sys, logging
from pyscf import gto, scf, mcscf

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

IRC_DIR = os.path.expanduser('~/activeml/data/irc_pd')
OUT_DIR = IRC_DIR  # save alongside XYZ files

def read_xyz(path):
    """Read XYZ file, return (symbols, coords_angstrom)."""
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    syms, coords = [], []
    for line in lines[2:2+n]:
        parts = line.split()
        syms.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return syms, np.array(coords)

def build_atom_str(syms, coords):
    return '\n'.join(f"{s}  {c[0]:.6f}  {c[1]:.6f}  {c[2]:.6f}"
                     for s, c in zip(syms, coords))

def run_uhf(mol):
    for s in [
        dict(max_cycle=300, conv_tol=1e-10, damp=0.0, level_shift=0.0),
        dict(max_cycle=500, conv_tol=1e-9,  damp=0.3, level_shift=0.2),
        dict(max_cycle=800, conv_tol=1e-8,  damp=0.5, level_shift=0.5),
    ]:
        mf = scf.UHF(mol)
        for k,v in s.items(): setattr(mf,k,v)
        mf.verbose=0; mf.run()
        if mf.converged: return mf
    return mf

def run_casscf14(mf, mol):
    """CASSCF(14,14) with 18-orbital window."""
    e_m = (mf.mo_energy[0]+mf.mo_energy[1])/2
    occ = mf.mo_occ[0]+mf.mo_occ[1]
    oi  = np.where(occ>0.5)[0]
    vi  = np.where(occ<0.5)[0]
    if not len(oi) or not len(vi): return None, False

    gap = (float(e_m[oi[-1]])+float(e_m[vi[0]]))/2
    w18 = sorted(np.argsort(np.abs(e_m-gap))[:18],
                 key=lambda i: e_m[i])

    # Determine n_active electrons for (14,14)
    n_tot = mol.nelectron
    for n in [14,13,15,12,16,11,10]:
        if n>0 and (n_tot-n)>=0 and (n_tot-n)%2==0:
            n_act=n; break
    else:
        n_act=14

    best_mc=None; best_e=0.0
    for win in [w18[:14],w18[2:16],w18[1:15],w18[4:18]]:
        for sh in [1e-3,1e-2,5e-2,1e-1,2e-1]:
            try:
                mc=mcscf.CASSCF(mf,14,n_act)
                mc.max_cycle_macro=600; mc.conv_tol=1e-8
                mc.ah_level_shift=sh; mc.verbose=0
                mc.kernel(mc.sort_mo(win,base=0))
                ec=mc.e_tot-mf.e_tot
                if ec<0 and ec<best_e: best_mc=mc; best_e=ec
                if mc.converged and ec<-0.01: return mc,True
            except: continue

    if best_mc: return best_mc,best_mc.converged
    return None,False

def run_one(snap_idx):
    xyz_file = os.path.join(IRC_DIR, f'snapshot_{snap_idx:02d}.xyz')
    out_file = os.path.join(OUT_DIR,  f'casscf_snapshot_{snap_idx:02d}.json')

    if not os.path.exists(xyz_file):
        log.error(f"XYZ not found: {xyz_file}"); return False

    if os.path.exists(out_file):
        r = json.load(open(out_file))
        if r.get('status')=='ok':
            log.info(f"SKIP: snap {snap_idx:02d}"); return True

    # Read geometry
    syms, coords = read_xyz(xyz_file)
    atom_str = build_atom_str(syms, coords)

    # Get IRC metadata from summary
    summary = json.load(open(os.path.join(IRC_DIR,'irc_summary.json')))
    snap_data = next((s for s in summary['snapshots']
                      if s['irc_index']==snap_idx), {})
    phase    = snap_data.get('phase','unknown')
    pd_c     = snap_data.get('pd_c_dist', 0.0)
    pd_cl    = snap_data.get('pd_cl_dist', 0.0)
    c_cl     = snap_data.get('c_cl_dist', 0.0)
    e_dft    = snap_data.get('energy', 0.0)

    log.info(f"Snap {snap_idx:02d} [{phase}]: "
             f"Pd-C={pd_c:.3f} C-Cl={c_cl:.3f}")

    try:
        mol = gto.Mole()
        mol.atom    = atom_str
        mol.basis   = 'def2-SVP'
        mol.ecp     = {'Pd': 'def2-SVP'}
        mol.charge  = 0
        mol.spin    = 0
        mol.verbose = 0
        mol.build()

        log.info(f"  n_electrons={mol.nelectron}")

        # UHF
        mf = run_uhf(mol)
        log.info(f"  UHF: E={mf.e_tot:.6f} conv={mf.converged}")

        # CASSCF(14,14)
        mc, conv = run_casscf14(mf, mol)

        if mc is None:
            log.warning(f"  CASSCF failed — saving UHF result")
            ec=0.0; n_active=0; no=[]
        else:
            ec = float(mc.e_tot - mf.e_tot)
            casdm1=mc.fcisolver.make_rdm1(mc.ci,mc.ncas,mc.nelecas)
            no_arr,_=np.linalg.eigh(casdm1)
            no=list(np.sort(no_arr)[::-1])
            n_active=sum(1 for n in no if 0.02<n<1.98)
            log.info(f"  CASSCF: Ec={ec:.4f} n_active={n_active} "
                     f"conv={conv}")

        # HF features for ML
        e_a,e_b=mf.mo_energy[0],mf.mo_energy[1]
        occ=mf.mo_occ[0]+mf.mo_occ[1]
        oi=np.where(occ>0.5)[0]; vi=np.where(occ<0.5)[0]
        em=(e_a+e_b)/2
        homo=float(em[oi[-1]]) if len(oi) else 0.0
        lumo=float(em[vi[0]])  if len(vi) else 0.0
        S=0.0
        sc=float(mf.spin_square()[0]-S*(S+1))

        result = {
            'snap_index':    snap_idx,
            'phase':         phase,
            'xyz_file':      f'snapshot_{snap_idx:02d}.xyz',
            'status':        'ok',
            'pd_c_dist':     float(pd_c),
            'pd_cl_dist':    float(pd_cl),
            'c_cl_dist':     float(c_cl),
            'E_DFT':         float(e_dft),
            'E_HF':          float(mf.e_tot),
            'E_CASSCF':      float(mc.e_tot) if mc else float(mf.e_tot),
            'corr_energy':   ec,
            'converged':     bool(conv if mc else mf.converged),
            'n_active':      n_active,
            'no_occ':        [float(x) for x in no],
            'n_electrons':   mol.nelectron,
            'spin_contamination': sc,
            'homo_lumo_gap': float(lumo-homo),
            'homo_energy':   homo,
            'lumo_energy':   lumo,
            # Reaction coordinate features
            'metal':         'Pd',
            'ligand':        'P',
            'n_ligands':     2,
            'charge':        0,
            'spin':          0,
            'mult':          1,
            'dist_ang':      float(pd_c),
            'geometry':      'irc',
            'z_eff':         13.00,
            'zeta_so_cm1':   1334,
            'metal_row':     '4d',
        }

        with open(out_file,'w') as f:
            json.dump(result, f, indent=2)
        log.info(f"  Saved: n_active={n_active} phase={phase}")
        return True

    except Exception as e:
        log.error(f"  Error snap {snap_idx}: {e}")
        json.dump({'snap_index':snap_idx,'status':'error',
                   'reason':str(e)}, open(out_file,'w'))
        return False

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='summary':
        print(f"\nIRC CASSCF jobs: 12 snapshots (0-11)")
        summary = json.load(
            open(os.path.join(IRC_DIR,'irc_summary.json')))
        print(f"Reaction: {summary['reaction']}")
        print(f"Barrier:  {summary['barrier_kcal']:.1f} kcal/mol")
        print(f"\nSnap  Phase              Pd-C   C-Cl  Done?")
        print("-"*55)
        for s in summary['snapshots']:
            done = os.path.exists(
                os.path.join(OUT_DIR,
                             f"casscf_snapshot_{s['irc_index']:02d}.json"))
            status = "✓" if done else "pending"
            print("  %2d  %-18s %5.3f  %5.3f  %s" % (
                s['irc_index'],s['phase'],
                s['pd_c_dist'],s['c_cl_dist'],status))
        sys.exit(0)

    idx = int(sys.argv[1]) if len(sys.argv)>1 else 0
    if idx>11: sys.exit(0)
    sys.exit(0 if run_one(idx) else 1)
