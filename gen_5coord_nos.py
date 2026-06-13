"""
More 5-coordinate (square pyramidal) systems
specifically targeting N, O, S ligands.
These are the weakest group (70.9% within±1).
Target: +200 systems for N/O/S at 5-coord.
"""
import numpy as np, json, os, sys, logging
from pyscf import gto, scf, mcscf

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

METAL_Z  = {'Fe':26,'Mn':25,'Cr':24,
             'Co':27,'Ni':28,'Cu':29}
LIG_Z    = {'N':7,'O':8,'S':16}
LIG_CHG  = {'N':-3,'O':-2,'S':-2}

EQ = {
    'Fe':{'N':2.10,'O':2.05,'S':2.35},
    'Mn':{'N':2.20,'O':2.15,'S':2.45},
    'Cr':{'N':2.10,'O':2.05,'S':2.40},
    'Co':{'N':2.00,'O':1.95,'S':2.30},
    'Ni':{'N':2.05,'O':2.00,'S':2.28},
    'Cu':{'N':2.05,'O':1.98,'S':2.32},
}

def parity_ok(metal, lig, charge, spin):
    n_e = METAL_Z[metal] + 5*LIG_Z[lig] - charge
    return (n_e%2)==(spin%2), n_e

def build_sqpyr(metal, ligand, d_eq, d_ax=None):
    """Square pyramidal: 4 equatorial + 1 axial."""
    if d_ax is None: d_ax = d_eq * 1.08
    s = f"{metal}  0.000  0.000  0.000\n"
    for p in [(d_eq,0,0),(-d_eq,0,0),
              (0,d_eq,0),(0,-d_eq,0)]:
        s += f"{ligand}  {p[0]:.3f}  {p[1]:.3f}  0.000\n"
    s += f"{ligand}  0.000  0.000  {d_ax:.3f}\n"
    return s

def build_tbp(metal, ligand, d_eq, d_ax=None):
    """Trigonal bipyramidal: 3 equatorial + 2 axial."""
    import math
    if d_ax is None: d_ax = d_eq * 1.05
    s = f"{metal}  0.000  0.000  0.000\n"
    for i in range(3):
        a = i * 2 * math.pi / 3
        s += (f"{ligand}  {d_eq*math.cos(a):.3f}  "
              f"{d_eq*math.sin(a):.3f}  0.000\n")
    s += f"{ligand}  0.000  0.000  {d_ax:.3f}\n"
    s += f"{ligand}  0.000  0.000  {-d_ax:.3f}\n"
    return s

def get_nact(n_total):
    for n in [10,9,11,8,12,7,13,6,14]:
        if (n_total-n)>=0 and (n_total-n)%2==0: return n
    return 10

def run_uhf(mol):
    for s in [
        dict(max_cycle=300,conv_tol=1e-10,
             damp=0.0,level_shift=0.0),
        dict(max_cycle=500,conv_tol=1e-9,
             damp=0.3,level_shift=0.2),
        dict(max_cycle=800,conv_tol=1e-8,
             damp=0.5,level_shift=0.5),
    ]:
        mf=scf.UHF(mol)
        for k,v in s.items(): setattr(mf,k,v)
        mf.verbose=0; mf.run()
        if mf.converged: return mf
    return mf

def run_casscf(mf, mol, n_act):
    e_m=(mf.mo_energy[0]+mf.mo_energy[1])/2
    occ=mf.mo_occ[0]+mf.mo_occ[1]
    occ_idx  = np.where(occ>0.5)[0]
    virt_idx = np.where(occ<0.5)[0]
    if len(occ_idx)==0 or len(virt_idx)==0:
        return None, False
    homo_e = float(e_m[occ_idx[-1]])
    lumo_e = float(e_m[virt_idx[0]])
    gap    = (homo_e+lumo_e)/2
    w14    = sorted(
        np.argsort(np.abs(e_m-gap))[:14],
        key=lambda i: e_m[i])
    best_mc=None; best_e=0.0
    for window in [w14[:10],w14[2:12],w14[1:11],w14[3:13]]:
        for shift in [1e-3,1e-2,5e-2,1e-1]:
            try:
                mc=mcscf.CASSCF(mf,10,n_act)
                mc.max_cycle_macro=500
                mc.conv_tol=1e-8
                mc.ah_level_shift=shift
                mc.verbose=0
                mc.kernel(mc.sort_mo(window,base=0))
                ec=mc.e_tot-mf.e_tot
                if ec<0 and ec<best_e:
                    best_mc=mc; best_e=ec
                if mc.converged and ec<-0.01:
                    return mc,True
            except: continue
    if best_mc: return best_mc,best_mc.converged
    return None,False

def run_one(metal, ligand, charge, spin, dist, geom):
    geom_tag = 'sqpyr' if geom=='sq' else 'tbp'
    name = (f"{metal}_{ligand}5{geom_tag}"
            f"_chg{charge}_spin{spin}")
    outdir  = os.path.expanduser(
        '~/activeml/data/generated300')
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"{name}.json")

    if os.path.exists(outfile):
        r = json.load(open(outfile))
        if (r.get('converged') and
                r.get('corr_energy',0)<-0.001):
            log.info(f"SKIP: {name}"); return True

    ok,n_e = parity_ok(metal,ligand,charge,spin)
    if not ok:
        json.dump({'name':name,'status':'skipped',
                   'reason':'parity','ligand':ligand},
                  open(outfile,'w'))
        return True

    log.info(f"Start: {name} n_e={n_e}")
    try:
        if geom == 'sq':
            atom_str = build_sqpyr(metal, ligand, dist)
        else:
            atom_str = build_tbp(metal, ligand, dist)

        mol = gto.Mole()
        mol.atom    = atom_str
        mol.basis   = 'def2-SVP'
        mol.charge  = charge
        mol.spin    = spin
        mol.verbose = 0
        mol.build()

        n_act = get_nact(mol.nelectron)
        mf    = run_uhf(mol)
        mc, conv = run_casscf(mf, mol, n_act)

        if mc is None:
            json.dump({'name':name,'status':'failed',
                       'ligand':ligand,'geometry':geom_tag},
                      open(outfile,'w'))
            return False

        ec = mc.e_tot - mf.e_tot
        if ec >= 0:
            json.dump({'name':name,'status':'unphysical',
                       'corr_energy':float(ec),
                       'ligand':ligand},
                      open(outfile,'w'))
            return False

        casdm1 = mc.fcisolver.make_rdm1(
            mc.ci, mc.ncas, mc.nelecas)
        no,_ = np.linalg.eigh(casdm1)
        no   = np.sort(no)[::-1]
        n_active = sum(1 for n in no if 0.02<n<1.98)

        json.dump({
            'name'       : name,
            'metal'      : metal,
            'ligand'     : ligand,
            'n_ligands'  : 5,
            'charge'     : charge,
            'spin'       : spin,
            'mult'       : spin+1,
            'dist_ang'   : dist,
            'geometry'   : geom_tag,
            'n_electrons': mol.nelectron,
            'n_active_e' : n_act,
            'E_HF'       : float(mf.e_tot),
            'E_CASSCF'   : float(mc.e_tot),
            'corr_energy': float(ec),
            'converged'  : bool(conv),
            'n_active'   : n_active,
            'no_occ'     : [float(x) for x in no],
            'status'     : 'ok',
        }, open(outfile,'w'), indent=2)

        log.info(f"  OK: n_active={n_active} Ec={ec:.4f}")
        return True

    except Exception as e:
        log.error(f"  Error: {e}")
        json.dump({'name':name,'status':'error',
                   'reason':str(e),'ligand':ligand},
                  open(outfile,'w'))
        return False

# ── BUILD JOB LIST ─────────────────────────────────────────────
ALL_JOBS = []
seen     = set()
gen300   = os.path.expanduser('~/activeml/data/generated300')
existing = set(os.path.basename(f)
               for f in __import__('glob').glob(
                   f'{gen300}/*.json'))

# Extended charge ranges and both geometries for N/O/S
for ligand in ['N','O','S']:
    for metal in ['Fe','Mn','Cr','Co','Ni','Cu']:
        eq = EQ[metal][ligand]
        for charge in [-1,-2,-3,-4,-5,-6,0,1]:
            for frac in [0.93,0.97,1.00,1.03,1.07]:
                dist = round(eq*frac, 3)
                for geom in ['sq','tbp']:
                    for spin in range(0,7):
                        ok,_ = parity_ok(
                            metal,ligand,charge,spin)
                        if not ok: continue
                        gtag = ('sqpyr' if geom=='sq'
                                else 'tbp')
                        fname = (f"{metal}_{ligand}5{gtag}"
                                 f"_chg{charge}_spin{spin}.json")
                        if fname in existing: continue
                        key=(metal,ligand,charge,dist,spin,geom)
                        if key in seen: continue
                        seen.add(key)
                        ALL_JOBS.append(
                            (metal,ligand,charge,spin,dist,geom))

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=='summary':
        from collections import defaultdict
        by_lig   = defaultdict(int)
        by_metal = defaultdict(int)
        by_geom  = defaultdict(int)
        for m,l,c,s,d,g in ALL_JOBS:
            by_lig[l]  += 1
            by_metal[m]+= 1
            by_geom[g] += 1
        print(f"Total 5-coord N/O/S jobs: {len(ALL_JOBS)}")
        print("By ligand:", dict(sorted(by_lig.items())))
        print("By metal: ", dict(sorted(by_metal.items())))
        print("By geom:  ", dict(sorted(by_geom.items())))
        sys.exit(0)
    idx = int(sys.argv[1]) if len(sys.argv)>1 else 0
    if idx >= len(ALL_JOBS): sys.exit(1)
    success = run_one(*ALL_JOBS[idx])
    sys.exit(0 if success else 1)
