"""Batch 1: Hydride (H-) ligand."""
import numpy as np, json, os, sys, logging
from pyscf import gto, scf, mcscf

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

METAL_Z = {'Fe':26,'Mn':25,'Cr':24,'Co':27,'Ni':28,'Cu':29}
EQ_DIST = {
    'Fe':1.65,'Mn':1.75,'Cr':1.72,
    'Co':1.62,'Ni':1.60,'Cu':1.63}

def parity_ok(metal, charge, n_lig, spin):
    n_e = METAL_Z[metal] + n_lig*1 - charge
    return (n_e%2)==(spin%2), n_e

def build_geometry(metal, n_lig, dist):
    pos = {
        4:[(dist,0,0),(-dist,0,0),(0,dist,0),(0,-dist,0)],
        6:[(dist,0,0),(-dist,0,0),(0,dist,0),
           (0,-dist,0),(0,0,dist),(0,0,-dist)]}
    s = f"{metal}  0.000  0.000  0.000\n"
    for p in pos[n_lig]:
        s += f"H  {p[0]:.3f}  {p[1]:.3f}  {p[2]:.3f}\n"
    return s

def get_nact(n_total):
    for n in [10,9,11,8,12,7,13,6,14]:
        if (n_total-n)>=0 and (n_total-n)%2==0:
            return n
    return 10

def run_uhf(mol):
    for s in [
        dict(max_cycle=300,conv_tol=1e-10,damp=0.0,level_shift=0.0),
        dict(max_cycle=500,conv_tol=1e-9, damp=0.3,level_shift=0.2),
        dict(max_cycle=800,conv_tol=1e-8, damp=0.5,level_shift=0.5),
    ]:
        mf=scf.UHF(mol)
        for k,v in s.items(): setattr(mf,k,v)
        mf.verbose=0; mf.run()
        if mf.converged: return mf
    return mf

def run_casscf(mf, mol, n_act):
    e_m=(mf.mo_energy[0]+mf.mo_energy[1])/2
    occ=mf.mo_occ[0]+mf.mo_occ[1]
    homo=float(e_m[occ>0.5].max())
    lumo=float(e_m[occ<0.5].min())
    gap=(homo+lumo)/2
    w14=sorted(np.argsort(np.abs(e_m-gap))[:14],
               key=lambda i:e_m[i])
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
                if mc.converged and ec<-0.01: return mc,True
            except: continue
    if best_mc: return best_mc,best_mc.converged
    return None,False

def run_one(metal,charge,n_lig,dist,spin):
    name=f"{metal}_H{n_lig}_chg{charge}_spin{spin}"
    outdir=os.path.expanduser('~/activeml/data/generated300')
    os.makedirs(outdir,exist_ok=True)
    outfile=os.path.join(outdir,f"{name}.json")
    if os.path.exists(outfile):
        r=json.load(open(outfile))
        if r.get('converged') and r.get('corr_energy',0)<-0.001:
            log.info(f"SKIP: {name}"); return True
    ok,n_e=parity_ok(metal,charge,n_lig,spin)
    if not ok:
        json.dump({'name':name,'status':'skipped',
                   'reason':'parity','ligand':'H'},
                  open(outfile,'w')); return True
    log.info(f"Start: {name} n_e={n_e}")
    try:
        mol=gto.Mole()
        mol.atom=build_geometry(metal,n_lig,dist)
        mol.basis='def2-SVP'; mol.charge=charge
        mol.spin=spin; mol.verbose=0; mol.build()
        n_act=get_nact(mol.nelectron)
        mf=run_uhf(mol)
        mc,conv=run_casscf(mf,mol,n_act)
        if mc is None:
            json.dump({'name':name,'status':'failed','ligand':'H'},
                      open(outfile,'w')); return False
        ec=mc.e_tot-mf.e_tot
        if ec>=0:
            json.dump({'name':name,'status':'unphysical',
                       'corr_energy':float(ec),'ligand':'H'},
                      open(outfile,'w')); return False
        dm1=mc.fcisolver.make_rdm1(mc.ci,mc.ncas,mc.nelecas)
        no,_=np.linalg.eigh(dm1)
        no=np.sort(no)[::-1]
        n_active=sum(1 for n in no if 0.02<n<1.98)
        json.dump({
            'name':name,'metal':metal,'ligand':'H',
            'n_ligands':n_lig,'charge':charge,'spin':spin,
            'mult':spin+1,'dist_ang':dist,
            'n_electrons':mol.nelectron,'n_active_e':n_act,
            'E_HF':float(mf.e_tot),'E_CASSCF':float(mc.e_tot),
            'corr_energy':float(ec),'converged':bool(conv),
            'n_active':n_active,'no_occ':[float(x) for x in no],
            'status':'ok'},open(outfile,'w'),indent=2)
        log.info(f"  OK: n_active={n_active} Ec={ec:.4f}")
        return True
    except Exception as e:
        log.error(f"  Error: {e}")
        json.dump({'name':name,'status':'error',
                   'reason':str(e),'ligand':'H'},
                  open(outfile,'w')); return False

# Build job list
ALL_JOBS=[]
seen=set()
gen300=os.path.expanduser('~/activeml/data/generated300')
existing=set(os.path.basename(f)
             for f in __import__('glob').glob(f'{gen300}/*.json'))

for metal in ['Fe','Mn','Cr','Co','Ni','Cu']:
    eq=EQ_DIST[metal]
    for charge in [-1,-2,-3,-4,-5,0,1,2]:
        for n_lig in [4,6]:
            for frac in [0.95,1.00,1.05]:
                dist=round(eq*frac,3)
                for spin in range(0,7):
                    ok,_=parity_ok(metal,charge,n_lig,spin)
                    if not ok: continue
                    fname=f"{metal}_H{n_lig}_chg{charge}_spin{spin}.json"
                    if fname in existing: continue
                    key=(metal,charge,n_lig,dist,spin)
                    if key in seen: continue
                    seen.add(key)
                    ALL_JOBS.append((metal,charge,n_lig,dist,spin))

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=='summary':
        from collections import defaultdict
        by_metal=defaultdict(int)
        for m,c,n,d,s in ALL_JOBS:
            by_metal[m]+=1
        print(f"Total H ligand jobs: {len(ALL_JOBS)}")
        for k,v in sorted(by_metal.items()):
            print(f"  {k}: {v}")
        sys.exit(0)
    idx=int(sys.argv[1]) if len(sys.argv)>1 else 0
    if idx>=len(ALL_JOBS): sys.exit(1)
    success=run_one(*ALL_JOBS[idx])
    sys.exit(0 if success else 1)
