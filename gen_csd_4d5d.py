"""
4d/5d CSD-style structure dataset.
Direct extension of gen_csd_real.py — same logic, same JSON schema.

Key differences vs gen_csd_real.py:
  - mol.ecp = 'def2-SVP' added for all 4d/5d metals
  - METAL_CONSTANTS provides z_eff, zeta_so_cm1, metal_row
  - Output goes to ~/activeml/data/generated_csd/
  - Structures cover Pd, Ru, Rh, Mo (4d) and Ir, Pt (5d)

Bond lengths from:
  Alvarez S., Dalton Trans. 2013, 42, 8617  (4d/5d M-X survey)
  Cambridge Structural Database mean values
  Orpen A.G. et al., J. Chem. Soc. Dalton Trans. 1989, S1

Usage:
  python gen_csd_4d5d.py summary        # count jobs
  python gen_csd_4d5d.py <idx>          # run job idx
  sbatch --array=0-N job_csd_4d5d.sh    # HPC array
"""
import numpy as np, json, os, sys, logging
from pyscf import gto, scf, mcscf

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

METAL_CONSTANTS = {
    'Mo': {'z_eff': 10.97, 'zeta_so_cm1':  467, 'metal_row': '4d'},
    'Ru': {'z_eff': 12.33, 'zeta_so_cm1':  880, 'metal_row': '4d'},
    'Rh': {'z_eff': 12.67, 'zeta_so_cm1': 1097, 'metal_row': '4d'},
    'Pd': {'z_eff': 13.00, 'zeta_so_cm1': 1334, 'metal_row': '4d'},
    'Ir': {'z_eff': 17.00, 'zeta_so_cm1': 3909, 'metal_row': '5d'},
    'Pt': {'z_eff': 17.33, 'zeta_so_cm1': 4146, 'metal_row': '5d'},
}

ECP_METALS = set(METAL_CONSTANTS.keys())

CSD_STRUCTURES = [
    ("PdCl4_2m_sqpl","Pd",-2,
     """Pd  0.000  0.000  0.000
        Cl  2.295  0.000  0.000
        Cl -2.295  0.000  0.000
        Cl  0.000  2.295  0.000
        Cl  0.000 -2.295  0.000"""),
    ("PdCl4_2m_sqpl_dist","Pd",-2,
     """Pd  0.000  0.000  0.000
        Cl  2.305  0.080 -0.040
        Cl -2.285  0.050  0.060
        Cl  0.060  2.295 -0.030
        Cl -0.080 -2.300  0.020"""),
    ("PdBr4_2m_sqpl","Pd",-2,
     """Pd  0.000  0.000  0.000
        Br  2.440  0.000  0.000
        Br -2.440  0.000  0.000
        Br  0.000  2.440  0.000
        Br  0.000 -2.440  0.000"""),
    ("PdCl6_2m_oct","Pd",-2,
     """Pd  0.000  0.000  0.000
        Cl  2.310  0.000  0.000
        Cl -2.310  0.000  0.000
        Cl  0.000  2.310  0.000
        Cl  0.000 -2.310  0.000
        Cl  0.000  0.000  2.310
        Cl  0.000  0.000 -2.310"""),
    ("PdCl2N2_sqpl","Pd",0,
     """Pd  0.000  0.000  0.000
        Cl  2.295  0.000  0.000
        Cl -2.295  0.000  0.000
        N   0.000  2.035  0.000
        N   0.000 -2.035  0.000"""),
    ("PdCl4_4m_tet","Pd",-4,
     """Pd  0.000  0.000  0.000
        Cl  2.380  0.000  0.000
        Cl -2.380  0.000  0.000
        Cl  0.000  2.380  0.000
        Cl  0.000 -2.380  0.000"""),
    ("RuCl6_3m_oct","Ru",-3,
     """Ru  0.000  0.000  0.000
        Cl  2.340  0.000  0.000
        Cl -2.340  0.000  0.000
        Cl  0.000  2.340  0.000
        Cl  0.000 -2.340  0.000
        Cl  0.000  0.000  2.340
        Cl  0.000  0.000 -2.340"""),
    ("RuCl6_4m_oct","Ru",-4,
     """Ru  0.000  0.000  0.000
        Cl  2.375  0.000  0.000
        Cl -2.375  0.000  0.000
        Cl  0.000  2.375  0.000
        Cl  0.000 -2.375  0.000
        Cl  0.000  0.000  2.375
        Cl  0.000  0.000 -2.375"""),
    ("RuCl6_3m_dist","Ru",-3,
     """Ru  0.000  0.000  0.000
        Cl  2.330  0.100 -0.060
        Cl -2.350  0.070  0.080
        Cl  0.080  2.340 -0.050
        Cl -0.060 -2.345  0.070
        Cl  0.040  0.030  2.360
        Cl -0.050 -0.040 -2.355"""),
    ("RuCl6_2m_oct","Ru",-2,
     """Ru  0.000  0.000  0.000
        Cl  2.320  0.000  0.000
        Cl -2.320  0.000  0.000
        Cl  0.000  2.320  0.000
        Cl  0.000 -2.320  0.000
        Cl  0.000  0.000  2.320
        Cl  0.000  0.000 -2.320"""),
    ("RuCl4N2_trans","Ru",-1,
     """Ru  0.000  0.000  0.000
        Cl  2.340  0.000  0.000
        Cl -2.340  0.000  0.000
        Cl  0.000  2.340  0.000
        Cl  0.000 -2.340  0.000
        N   0.000  0.000  2.130
        N   0.000  0.000 -2.130"""),
    ("RuBr6_3m_oct","Ru",-3,
     """Ru  0.000  0.000  0.000
        Br  2.480  0.000  0.000
        Br -2.480  0.000  0.000
        Br  0.000  2.480  0.000
        Br  0.000 -2.480  0.000
        Br  0.000  0.000  2.480
        Br  0.000  0.000 -2.480"""),
    ("RhCl4_3m_sqpl","Rh",-3,
     """Rh  0.000  0.000  0.000
        Cl  2.375  0.000  0.000
        Cl -2.375  0.000  0.000
        Cl  0.000  2.375  0.000
        Cl  0.000 -2.375  0.000"""),
    ("RhCl6_3m_oct","Rh",-3,
     """Rh  0.000  0.000  0.000
        Cl  2.325  0.000  0.000
        Cl -2.325  0.000  0.000
        Cl  0.000  2.325  0.000
        Cl  0.000 -2.325  0.000
        Cl  0.000  0.000  2.325
        Cl  0.000  0.000 -2.325"""),
    ("RhCl6_3m_dist","Rh",-3,
     """Rh  0.000  0.000  0.000
        Cl  2.315  0.090 -0.050
        Cl -2.335  0.060  0.070
        Cl  0.070  2.325 -0.045
        Cl -0.055 -2.330  0.065
        Cl  0.035  0.025  2.345
        Cl -0.045 -0.035 -2.340"""),
    ("RhCl2N2_1m_sqpl","Rh",-1,
     """Rh  0.000  0.000  0.000
        Cl  2.375  0.000  0.000
        Cl -2.375  0.000  0.000
        N   0.000  2.080  0.000
        N   0.000 -2.080  0.000"""),
    ("RhBr6_3m_oct","Rh",-3,
     """Rh  0.000  0.000  0.000
        Br  2.465  0.000  0.000
        Br -2.465  0.000  0.000
        Br  0.000  2.465  0.000
        Br  0.000 -2.465  0.000
        Br  0.000  0.000  2.465
        Br  0.000  0.000 -2.465"""),
    ("MoCl6_3m_oct","Mo",-3,
     """Mo  0.000  0.000  0.000
        Cl  2.480  0.000  0.000
        Cl -2.480  0.000  0.000
        Cl  0.000  2.480  0.000
        Cl  0.000 -2.480  0.000
        Cl  0.000  0.000  2.480
        Cl  0.000  0.000 -2.480"""),
    ("MoCl6_2m_oct","Mo",-2,
     """Mo  0.000  0.000  0.000
        Cl  2.460  0.000  0.000
        Cl -2.460  0.000  0.000
        Cl  0.000  2.460  0.000
        Cl  0.000 -2.460  0.000
        Cl  0.000  0.000  2.460
        Cl  0.000  0.000 -2.460"""),
    ("MoCl6_1m_oct","Mo",-1,
     """Mo  0.000  0.000  0.000
        Cl  2.440  0.000  0.000
        Cl -2.440  0.000  0.000
        Cl  0.000  2.440  0.000
        Cl  0.000 -2.440  0.000
        Cl  0.000  0.000  2.440
        Cl  0.000  0.000 -2.440"""),
    ("MoO4_2m_tet","Mo",-2,
     """Mo  0.000  0.000  0.000
        O   1.930  0.000  0.000
        O  -1.930  0.000  0.000
        O   0.000  1.930  0.000
        O   0.000 -1.930  0.000"""),
    ("MoCl6_3m_dist","Mo",-3,
     """Mo  0.000  0.000  0.000
        Cl  2.470  0.110 -0.070
        Cl -2.490  0.080  0.090
        Cl  0.090  2.480 -0.060
        Cl -0.070 -2.485  0.080
        Cl  0.050  0.040  2.495
        Cl -0.060 -0.050 -2.500"""),
    ("MoCl4O2_trans","Mo",-2,
     """Mo  0.000  0.000  0.000
        Cl  2.450  0.000  0.000
        Cl -2.450  0.000  0.000
        Cl  0.000  2.450  0.000
        Cl  0.000 -2.450  0.000
        O   0.000  0.000  1.680
        O   0.000  0.000 -1.680"""),
    ("IrCl6_3m_oct","Ir",-3,
     """Ir  0.000  0.000  0.000
        Cl  2.355  0.000  0.000
        Cl -2.355  0.000  0.000
        Cl  0.000  2.355  0.000
        Cl  0.000 -2.355  0.000
        Cl  0.000  0.000  2.355
        Cl  0.000  0.000 -2.355"""),
    ("IrCl4_3m_sqpl","Ir",-3,
     """Ir  0.000  0.000  0.000
        Cl  2.375  0.000  0.000
        Cl -2.375  0.000  0.000
        Cl  0.000  2.375  0.000
        Cl  0.000 -2.375  0.000"""),
    ("IrCl6_3m_dist","Ir",-3,
     """Ir  0.000  0.000  0.000
        Cl  2.345  0.100 -0.060
        Cl -2.365  0.070  0.080
        Cl  0.080  2.355 -0.050
        Cl -0.060 -2.360  0.070
        Cl  0.040  0.030  2.370
        Cl -0.050 -0.040 -2.365"""),
    ("IrCl6_2m_oct","Ir",-2,
     """Ir  0.000  0.000  0.000
        Cl  2.340  0.000  0.000
        Cl -2.340  0.000  0.000
        Cl  0.000  2.340  0.000
        Cl  0.000 -2.340  0.000
        Cl  0.000  0.000  2.340
        Cl  0.000  0.000 -2.340"""),
    ("IrCl4N2_3m_trans","Ir",-1,
     """Ir  0.000  0.000  0.000
        Cl  2.355  0.000  0.000
        Cl -2.355  0.000  0.000
        Cl  0.000  2.355  0.000
        Cl  0.000 -2.355  0.000
        N   0.000  0.000  2.090
        N   0.000  0.000 -2.090"""),
    ("IrBr6_3m_oct","Ir",-3,
     """Ir  0.000  0.000  0.000
        Br  2.490  0.000  0.000
        Br -2.490  0.000  0.000
        Br  0.000  2.490  0.000
        Br  0.000 -2.490  0.000
        Br  0.000  0.000  2.490
        Br  0.000  0.000 -2.490"""),
    ("PtCl4_2m_sqpl","Pt",-2,
     """Pt  0.000  0.000  0.000
        Cl  2.305  0.000  0.000
        Cl -2.305  0.000  0.000
        Cl  0.000  2.305  0.000
        Cl  0.000 -2.305  0.000"""),
    ("PtCl4_2m_sqpl_dist","Pt",-2,
     """Pt  0.000  0.000  0.000
        Cl  2.315  0.075 -0.035
        Cl -2.295  0.050  0.055
        Cl  0.055  2.305 -0.028
        Cl -0.075 -2.310  0.018"""),
    ("PtCl2N2_sqpl","Pt",0,
     """Pt  0.000  0.000  0.000
        Cl  2.305  0.000  0.000
        Cl -2.305  0.000  0.000
        N   0.000  2.025  0.000
        N   0.000 -2.025  0.000"""),
    ("PtCl6_2m_oct","Pt",-2,
     """Pt  0.000  0.000  0.000
        Cl  2.325  0.000  0.000
        Cl -2.325  0.000  0.000
        Cl  0.000  2.325  0.000
        Cl  0.000 -2.325  0.000
        Cl  0.000  0.000  2.325
        Cl  0.000  0.000 -2.325"""),
    ("PtCl6_2m_dist","Pt",-2,
     """Pt  0.000  0.000  0.000
        Cl  2.315  0.095 -0.055
        Cl -2.335  0.065  0.075
        Cl  0.075  2.325 -0.048
        Cl -0.058 -2.330  0.068
        Cl  0.038  0.028  2.340
        Cl -0.048 -0.038 -2.335"""),
    ("PtBr4_2m_sqpl","Pt",-2,
     """Pt  0.000  0.000  0.000
        Br  2.445  0.000  0.000
        Br -2.445  0.000  0.000
        Br  0.000  2.445  0.000
        Br  0.000 -2.445  0.000"""),
]

SPIN_STATES = {
    'Mo': [1, 3, 5],
    'Ru': [0, 2, 4],
    'Rh': [0, 2],
    'Pd': [0, 2],
    'Ir': [0, 2],
    'Pt': [0, 2],
}

METAL_Z = {'Mo':42,'Ru':44,'Rh':45,'Pd':46,'Ir':77,'Pt':78}
LIG_Z   = {'Cl':17,'Br':35,'F':9,'N':7,'O':8}
OUTPUT_DIR = os.path.expanduser('~/activeml/data/generated_csd')

def count_electrons(atoms_str, charge):
    n_e = 0
    for line in atoms_str.strip().split('\n'):
        sym = line.strip().split()[0]
        n_e += METAL_Z.get(sym, LIG_Z.get(sym, 0))
    return n_e - charge

def get_nact(n_total):
    for n in [10,9,11,8,12,7,13,6,14]:
        if (n_total-n)>=0 and (n_total-n)%2==0: return n
    return 10

def run_uhf(mol):
    for settings in [
        dict(max_cycle=300,conv_tol=1e-10,damp=0.0,level_shift=0.0),
        dict(max_cycle=500,conv_tol=1e-9, damp=0.3,level_shift=0.2),
        dict(max_cycle=800,conv_tol=1e-8, damp=0.5,level_shift=0.5),
    ]:
        mf=scf.UHF(mol)
        for k,v in settings.items(): setattr(mf,k,v)
        mf.verbose=0; mf.run()
        if mf.converged: return mf
    return mf

def run_casscf(mf, mol, n_act):
    e_m=(mf.mo_energy[0]+mf.mo_energy[1])/2
    occ=mf.mo_occ[0]+mf.mo_occ[1]
    occ_idx  = np.where(occ>0.5)[0]
    virt_idx = np.where(occ<0.5)[0]
    if len(occ_idx)==0 or len(virt_idx)==0: return None,False
    homo_e=float(e_m[occ_idx[-1]])
    lumo_e=float(e_m[virt_idx[0]])
    gap=(homo_e+lumo_e)/2
    w14=sorted(np.argsort(np.abs(e_m-gap))[:14],key=lambda i:e_m[i])
    best_mc=None; best_e=0.0
    for window in [w14[:10],w14[2:12],w14[1:11],w14[3:13]]:
        for shift in [1e-3,1e-2,5e-2,1e-1]:
            try:
                mc=mcscf.CASSCF(mf,10,n_act)
                mc.max_cycle_macro=500; mc.conv_tol=1e-8
                mc.ah_level_shift=shift; mc.verbose=0
                mc.kernel(mc.sort_mo(window,base=0))
                ec=mc.e_tot-mf.e_tot
                if ec<0 and ec<best_e: best_mc=mc; best_e=ec
                if mc.converged and ec<-0.01: return mc,True
            except: continue
    if best_mc: return best_mc,best_mc.converged
    return None,False

def run_one(struct_name, metal, charge, atoms_str, spin):
    name=f"CSD_{struct_name}_spin{spin}"
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    outfile=os.path.join(OUTPUT_DIR,f"{name}.json")
    if os.path.exists(outfile):
        r=json.load(open(outfile))
        if r.get('converged') and r.get('corr_energy',0)<-0.001:
            log.info(f"SKIP: {name}"); return True
    n_e=count_electrons(atoms_str,charge)
    if (n_e%2)!=(spin%2):
        json.dump({'name':name,'status':'skipped','reason':'parity','geometry':'csd_real'},open(outfile,'w'))
        return True
    consts=METAL_CONSTANTS[metal]
    log.info(f"Start: {name}  n_e={n_e}  row={consts['metal_row']}")
    try:
        mol=gto.Mole()
        mol.atom=atoms_str; mol.basis='def2-SVP'; mol.ecp='def2-SVP'
        mol.charge=charge; mol.spin=spin; mol.verbose=0
        mol.build()
        n_act=get_nact(mol.nelectron)
        mf=run_uhf(mol)
        mc,conv=run_casscf(mf,mol,n_act)
        if mc is None:
            json.dump({'name':name,'status':'failed','geometry':'csd_real','metal':metal},open(outfile,'w'))
            return False
        ec=mc.e_tot-mf.e_tot
        if ec>=0:
            json.dump({'name':name,'status':'unphysical','corr_energy':float(ec),'geometry':'csd_real'},open(outfile,'w'))
            return False
        casdm1=mc.fcisolver.make_rdm1(mc.ci,mc.ncas,mc.nelecas)
        no,_=np.linalg.eigh(casdm1); no=np.sort(no)[::-1]
        n_active=sum(1 for n in no if 0.02<n<1.98)
        lig='Cl'
        for l in ['Br','F','N','O']:
            if l in atoms_str: lig=l; break
        e_m=(mf.mo_energy[0]+mf.mo_energy[1])/2
        occ=mf.mo_occ[0]+mf.mo_occ[1]
        occ_idx=np.where(occ>0.5)[0]; virt_idx=np.where(occ<0.5)[0]
        homo_e=float(e_m[occ_idx[-1]])  if len(occ_idx)>0  else 0.0
        lumo_e=float(e_m[virt_idx[0]]) if len(virt_idx)>0 else 0.0
        homo_a=float(mf.mo_energy[0][np.where(mf.mo_occ[0]>0.5)[0][-1]]) if len(np.where(mf.mo_occ[0]>0.5)[0])>0 else 0.0
        homo_b=float(mf.mo_energy[1][np.where(mf.mo_occ[1]>0.5)[0][-1]]) if len(np.where(mf.mo_occ[1]>0.5)[0])>0 else 0.0
        S=spin/2.0; spin_contam=float(mf.spin_square()[0]-S*(S+1))
        result={
            'name':name,'metal':metal,'ligand':lig,
            'n_ligands':atoms_str.count(lig),'charge':charge,
            'spin':spin,'mult':spin+1,'geometry':'csd_real',
            'struct_name':struct_name,'n_electrons':mol.nelectron,
            'n_active_e':n_act,'E_HF':float(mf.e_tot),
            'E_CASSCF':float(mc.e_tot),'corr_energy':float(ec),
            'converged':bool(conv),'n_active':n_active,
            'no_occ':[float(x) for x in no],'status':'ok',
            'z_eff':consts['z_eff'],'zeta_so_cm1':consts['zeta_so_cm1'],
            'metal_row':consts['metal_row'],'spin_contamination':spin_contam,
            'homo_lumo_gap':float(lumo_e-homo_e),'homo_energy':homo_e,
            'lumo_energy':lumo_e,'homo_ab_gap':float(abs(homo_a-homo_b)),
        }
        with open(outfile,'w') as f: json.dump(result,f,indent=2)
        log.info(f"  OK: n_active={n_active}  Ec={ec:.4f}  converged={conv}")
        return True
    except Exception as e:
        log.error(f"  Error: {name}: {e}")
        json.dump({'name':name,'status':'error','reason':str(e),'geometry':'csd_real'},open(outfile,'w'))
        return False

ALL_JOBS=[]; seen=set()
os.makedirs(OUTPUT_DIR,exist_ok=True)
existing=set(os.path.basename(f) for f in __import__('glob').glob(f'{OUTPUT_DIR}/*.json'))
for struct_name,metal,charge,atoms_str in CSD_STRUCTURES:
    spins=SPIN_STATES.get(metal,[0,2])
    n_e=count_electrons(atoms_str,charge)
    for spin in spins:
        if (n_e%2)!=(spin%2): continue
        fname=f"CSD_{struct_name}_spin{spin}.json"
        if fname in existing: continue
        key=(struct_name,spin)
        if key in seen: continue
        seen.add(key)
        ALL_JOBS.append((struct_name,metal,charge,atoms_str,spin))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='summary':
        from collections import defaultdict
        by_metal=defaultdict(int)
        for s,m,c,a,sp in ALL_JOBS: by_metal[m]+=1
        print(f"\nTotal jobs:   {len(ALL_JOBS)}")
        print(f"Structures:   {len(CSD_STRUCTURES)}")
        print("\nJobs by metal:")
        for m in ['Mo','Ru','Rh','Pd','Ir','Pt']:
            print(f"  {m}: {by_metal.get(m,0)}")
        print(f"\nEstimated HPC time: ~{len(ALL_JOBS)*25/60:.0f} hrs serial, ~{len(ALL_JOBS)*25/60/32:.1f} hrs on 32 cores")
        sys.exit(0)
    idx=int(sys.argv[1]) if len(sys.argv)>1 else 0
    if idx>=len(ALL_JOBS): sys.exit(1)
    success=run_one(*ALL_JOBS[idx])
    sys.exit(0 if success else 1)
