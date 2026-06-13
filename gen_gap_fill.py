"""
gen_gap_fill.py
===============
Targeted gap-filling for specific d-electron count / geometry gaps
revealed by honest validation analysis.

Gaps:
  1. V(IV) d1 — vanadyl and V(IV) halide complexes
  2. Re(III/IV) d3/d4 — rhenium halide complexes
  3. Mo(III) d3 — molybdenum halide complexes
  4. Ni(II) tetrahedral d8 — NiCl4, NiBr4, Ni(CN)4
  5. Co(III) d6 LS with N donors — Co(NH3)6, Co(en)3

Output: ~/activeml/data/generated_gap_fill/
"""
import numpy as np, json, os, sys, logging
from pyscf import gto, scf, mcscf, mp

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

OUTPUT_DIR = os.path.expanduser('~/activeml/data/generated_gap_fill')
os.makedirs(OUTPUT_DIR, exist_ok=True)

GROUP_NUMBER = {
    'Ti':4,'V':5,'Cr':6,'Mn':7,'Fe':8,'Co':9,'Ni':10,'Cu':11,'Zn':12,
    'Mo':6,'Ru':8,'Rh':9,'Pd':10,'Tc':7,
    'W':6,'Re':7,'Os':8,'Ir':9,'Pt':10,
}
METAL_CONSTANTS = {
    'V': {'z_eff':8.45, 'zeta_so_cm1':167,  'metal_row':'3d'},
    'Cr':{'z_eff':9.76, 'zeta_so_cm1':273,  'metal_row':'3d'},
    'Co':{'z_eff':12.00,'zeta_so_cm1':533,  'metal_row':'3d'},
    'Ni':{'z_eff':12.78,'zeta_so_cm1':669,  'metal_row':'3d'},
    'Mo':{'z_eff':10.97,'zeta_so_cm1':467,  'metal_row':'4d'},
    'Ru':{'z_eff':12.33,'zeta_so_cm1':880,  'metal_row':'4d'},
    'Re':{'z_eff':17.01,'zeta_so_cm1':2456, 'metal_row':'5d'},
    'Os':{'z_eff':17.17,'zeta_so_cm1':3381, 'metal_row':'5d'},
    'Ir':{'z_eff':17.00,'zeta_so_cm1':3909, 'metal_row':'5d'},
}
ECP = {'Mo','Ru','Re','Os','Ir'}
HALIDES = {'Cl','Br','F'}
ATOM_Z = {'V':23,'Cr':24,'Co':27,'Ni':28,'Mo':42,'Ru':44,'Re':75,
          'Os':76,'Ir':77,'Cl':17,'Br':35,'F':9,'N':7,'O':8,'P':15}

def correct_d_count(metal, charge, atom_str):
    group = GROUP_NUMBER.get(metal,0)
    if not group: return 0
    n_hal = sum(1 for l in atom_str.split('\n')
                if l.strip().split()[0:1] and
                l.strip().split()[0] in HALIDES)
    return max(0, group - (charge + n_hal))

# ── STRUCTURES ───────────────────────────────────────────────
# Format: (name, metal, charge, spin, atom_str, ligand, n_lig,
#          dist_ang, geometry, expected_n_active)

STRUCTURES = [

    # ── Gap 1: V(IV) d1 ──────────────────────────────────────
    # V(IV) = d1, spin=1, n_active=1
    ("VCl4_tet_d1",    "V", 0, 1,
     "V  0.000 0.000 0.000\nCl 2.140 0.000 0.000\nCl -2.140 0.000 0.000\nCl 0.000 2.140 0.000\nCl 0.000 -2.140 0.000",
     "Cl", 4, 2.140, "tet", 1),

    ("VCl6_2m_oct_d1", "V", -2, 1,
     "V  0.000 0.000 0.000\nCl 2.300 0.000 0.000\nCl -2.300 0.000 0.000\nCl 0.000 2.300 0.000\nCl 0.000 -2.300 0.000\nCl 0.000 0.000 2.300\nCl 0.000 0.000 -2.300",
     "Cl", 6, 2.300, "oct", 1),

    ("VBr4_tet_d1",    "V", 0, 1,
     "V  0.000 0.000 0.000\nBr 2.280 0.000 0.000\nBr -2.280 0.000 0.000\nBr 0.000 2.280 0.000\nBr 0.000 -2.280 0.000",
     "Br", 4, 2.280, "tet", 1),

    ("VF6_1m_oct_d1",  "V", -1, 1,
     "V  0.000 0.000 0.000\nF  1.840 0.000 0.000\nF -1.840 0.000 0.000\nF  0.000 1.840 0.000\nF  0.000 -1.840 0.000\nF  0.000 0.000 1.840\nF  0.000 0.000 -1.840",
     "F", 6, 1.840, "oct", 1),

    ("VCl5_1m_sqpyr_d1","V",-1, 1,
     "V  0.000 0.000 0.000\nCl 2.180 0.000 0.000\nCl -2.180 0.000 0.000\nCl 0.000 2.180 0.000\nCl 0.000 -2.180 0.000\nCl 0.000 0.000 2.180",
     "Cl", 5, 2.180, "sqpyr", 1),

    ("VN6_3p_oct_d2",  "V", 3, 0,
     "V  0.000 0.000 0.000\nN  2.050 0.000 0.000\nN -2.050 0.000 0.000\nN  0.000 2.050 0.000\nN  0.000 -2.050 0.000\nN  0.000 0.000 2.050\nN  0.000 0.000 -2.050",
     "N", 6, 2.050, "oct", 2),

    # ── Gap 2: Re(III/IV) d3/d4 ──────────────────────────────
    # Re(IV) = d3, spin=3 (HS) or spin=1 (LS)
    ("ReCl6_2m_oct_d3","Re",-2, 3,
     "Re 0.000 0.000 0.000\nCl 2.370 0.000 0.000\nCl -2.370 0.000 0.000\nCl 0.000 2.370 0.000\nCl 0.000 -2.370 0.000\nCl 0.000 0.000 2.370\nCl 0.000 0.000 -2.370",
     "Cl", 6, 2.370, "oct", 3),

    ("ReCl6_2m_LS_d3", "Re",-2, 1,
     "Re 0.000 0.000 0.000\nCl 2.370 0.000 0.000\nCl -2.370 0.000 0.000\nCl 0.000 2.370 0.000\nCl 0.000 -2.370 0.000\nCl 0.000 0.000 2.370\nCl 0.000 0.000 -2.370",
     "Cl", 6, 2.370, "oct", 3),

    ("ReBr6_2m_oct_d3","Re",-2, 3,
     "Re 0.000 0.000 0.000\nBr 2.490 0.000 0.000\nBr -2.490 0.000 0.000\nBr 0.000 2.490 0.000\nBr 0.000 -2.490 0.000\nBr 0.000 0.000 2.490\nBr 0.000 0.000 -2.490",
     "Br", 6, 2.490, "oct", 3),

    ("ReCl6_3m_oct_d4","Re",-3, 4,
     "Re 0.000 0.000 0.000\nCl 2.370 0.000 0.000\nCl -2.370 0.000 0.000\nCl 0.000 2.370 0.000\nCl 0.000 -2.370 0.000\nCl 0.000 0.000 2.370\nCl 0.000 0.000 -2.370",
     "Cl", 6, 2.370, "oct", 4),

    ("ReCl5_1m_sqpyr_d3","Re",-1, 3,
     "Re 0.000 0.000 0.000\nCl 2.340 0.000 0.000\nCl -2.340 0.000 0.000\nCl 0.000 2.340 0.000\nCl 0.000 -2.340 0.000\nCl 0.000 0.000 2.340",
     "Cl", 5, 2.340, "sqpyr", 3),

    # ── Gap 3: Mo(III) d3 ────────────────────────────────────
    ("MoCl6_3m_oct_d3","Mo",-3, 3,
     "Mo 0.000 0.000 0.000\nCl 2.520 0.000 0.000\nCl -2.520 0.000 0.000\nCl 0.000 2.520 0.000\nCl 0.000 -2.520 0.000\nCl 0.000 0.000 2.520\nCl 0.000 0.000 -2.520",
     "Cl", 6, 2.520, "oct", 3),

    ("MoBr6_3m_oct_d3","Mo",-3, 3,
     "Mo 0.000 0.000 0.000\nBr 2.620 0.000 0.000\nBr -2.620 0.000 0.000\nBr 0.000 2.620 0.000\nBr 0.000 -2.620 0.000\nBr 0.000 0.000 2.620\nBr 0.000 0.000 -2.620",
     "Br", 6, 2.620, "oct", 3),

    ("MoCl5_2m_sqpyr_d3","Mo",-2, 3,
     "Mo 0.000 0.000 0.000\nCl 2.480 0.000 0.000\nCl -2.480 0.000 0.000\nCl 0.000 2.480 0.000\nCl 0.000 -2.480 0.000\nCl 0.000 0.000 2.480",
     "Cl", 5, 2.480, "sqpyr", 3),

    ("MoCl6_2m_oct_d4","Mo",-2, 4,
     "Mo 0.000 0.000 0.000\nCl 2.520 0.000 0.000\nCl -2.520 0.000 0.000\nCl 0.000 2.520 0.000\nCl 0.000 -2.520 0.000\nCl 0.000 0.000 2.520\nCl 0.000 0.000 -2.520",
     "Cl", 6, 2.520, "oct", 4),

    # ── Gap 4: Ni(II) tetrahedral d8 ─────────────────────────
    # Tetrahedral Ni(II) = d8, n_active=8 (full d shell active)
    ("NiCl4_2m_tet_d8","Ni",-2, 2,
     "Ni 0.000 0.000 0.000\nCl 2.270 0.000 0.000\nCl -2.270 0.000 0.000\nCl 0.000 2.270 0.000\nCl 0.000 -2.270 0.000",
     "Cl", 4, 2.270, "tet", 8),

    ("NiBr4_2m_tet_d8","Ni",-2, 2,
     "Ni 0.000 0.000 0.000\nBr 2.380 0.000 0.000\nBr -2.380 0.000 0.000\nBr 0.000 2.380 0.000\nBr 0.000 -2.380 0.000",
     "Br", 4, 2.380, "tet", 8),

    ("NiCl4_2m_tet_HS","Ni",-2, 4,
     "Ni 0.000 0.000 0.000\nCl 2.270 0.000 0.000\nCl -2.270 0.000 0.000\nCl 0.000 2.270 0.000\nCl 0.000 -2.270 0.000",
     "Cl", 4, 2.270, "tet", 8),

    ("NiF4_2m_tet_d8", "Ni",-2, 2,
     "Ni 0.000 0.000 0.000\nF  1.930 0.000 0.000\nF -1.930 0.000 0.000\nF  0.000 1.930 0.000\nF  0.000 -1.930 0.000",
     "F", 4, 1.930, "tet", 8),

    ("NiN4_2p_sqpl_d8","Ni", 2, 0,
     "Ni 0.000 0.000 0.000\nN  2.020 0.000 0.000\nN -2.020 0.000 0.000\nN  0.000 2.020 0.000\nN  0.000 -2.020 0.000",
     "N", 4, 2.020, "sq_pl", 8),

    # ── Gap 5: Co(III) d6 LS with N donors ───────────────────
    # Co(III) LS = d6, n_active=6
    ("CoN6_3p_oct_d6", "Co", 3, 0,
     "Co 0.000 0.000 0.000\nN  2.050 0.000 0.000\nN -2.050 0.000 0.000\nN  0.000 2.050 0.000\nN  0.000 -2.050 0.000\nN  0.000 0.000 2.050\nN  0.000 0.000 -2.050",
     "N", 6, 2.050, "oct", 6),

    ("CoCl6_3m_oct_d6","Co",-3, 0,
     "Co 0.000 0.000 0.000\nCl 2.350 0.000 0.000\nCl -2.350 0.000 0.000\nCl 0.000 2.350 0.000\nCl 0.000 -2.350 0.000\nCl 0.000 0.000 2.350\nCl 0.000 0.000 -2.350",
     "Cl", 6, 2.350, "oct", 6),

    ("CoF6_3m_oct_d6", "Co",-3, 0,
     "Co 0.000 0.000 0.000\nF  1.890 0.000 0.000\nF -1.890 0.000 0.000\nF  0.000 1.890 0.000\nF  0.000 -1.890 0.000\nF  0.000 0.000 1.890\nF  0.000 0.000 -1.890",
     "F", 6, 1.890, "oct", 6),

    ("CoN6_3p_LS_v2",  "Co", 3, 0,
     "Co 0.000 0.000 0.000\nN  2.100 0.000 0.000\nN -2.100 0.000 0.000\nN  0.000 2.100 0.000\nN  0.000 -2.100 0.000\nN  0.000 0.000 2.100\nN  0.000 0.000 -2.100",
     "N", 6, 2.100, "oct", 6),
]

def run_uhf(mol):
    for s in [dict(max_cycle=300,conv_tol=1e-10,damp=0.0,level_shift=0.0),
              dict(max_cycle=500,conv_tol=1e-9, damp=0.3,level_shift=0.2),
              dict(max_cycle=800,conv_tol=1e-8, damp=0.5,level_shift=0.5)]:
        mf=scf.UHF(mol)
        for k,v in s.items(): setattr(mf,k,v)
        mf.verbose=0; mf.run()
        if mf.converged: return mf
    return mf

def run_casscf(mf,mol,n_act,cas_n=10):
    if isinstance(mf,scf.rhf.RHF):
        e_m=mf.mo_energy; occ=mf.mo_occ
    else:
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
                if mc.converged and ec<-0.001: return mc,True
            except: continue
    if best_mc: return best_mc,best_mc.converged
    return None,False

def run_one(idx):
    (name,metal,charge,spin,atom_str,ligand,
     n_lig,dist,geom,exp_n) = STRUCTURES[idx]

    fname   = "GAP_%s_spin%d.json" % (name,spin)
    outfile = os.path.join(OUTPUT_DIR,fname)

    if os.path.exists(outfile):
        r=json.load(open(outfile))
        if r.get('status')=='ok':
            log.info("SKIP: %s" % fname); return True

    consts = METAL_CONSTANTS.get(metal,
             {'z_eff':10,'zeta_so_cm1':400,'metal_row':'3d'})
    n_e = sum(ATOM_Z.get(l.strip().split()[0],0)
              for l in atom_str.split('\n') if l.strip()) - charge

    if (n_e%2) != (spin%2):
        json.dump({'name':fname,'status':'skipped','reason':'parity'},
                  open(outfile,'w')); return True

    d_count = correct_d_count(metal, charge, atom_str)
    log.info("Start: %s d=%d exp_n=%d" % (fname,d_count,exp_n))

    try:
        mol=gto.Mole(); mol.atom=atom_str; mol.basis='def2-SVP'
        mol.charge=charge; mol.spin=spin; mol.verbose=0
        if metal in ECP: mol.ecp='def2-SVP'
        mol.build()

        mf=run_uhf(mol)
        log.info("  UHF: E=%.6f conv=%s" % (mf.e_tot,mf.converged))

        # n_act for CASSCF
        for k in [exp_n,exp_n+2,exp_n-2,10,8,6]:
            if k>0 and (mol.nelectron-k)>=0 and (mol.nelectron-k)%2==0:
                n_act=k; break
        else: n_act=exp_n

        mc,conv=run_casscf(mf,mol,n_act)
        if mc is None or mc.e_tot-mf.e_tot>=0:
            n_active=0; ec=0.0; no=[]
        else:
            ec=float(mc.e_tot-mf.e_tot)
            casdm1=mc.fcisolver.make_rdm1(mc.ci,mc.ncas,mc.nelecas)
            no_a,_=np.linalg.eigh(casdm1); no=list(np.sort(no_a)[::-1])
            n_active=sum(1 for n in no if 0.02<n<1.98)

        # MP2
        mp2_corr=0.0; largest_t2=0.0; nfmp=[0,0,0]
        try:
            pt=mp.UMP2(mf).run(); mp2_corr=float(pt.e_corr)
            if pt.t2 is not None:
                arr=np.concatenate([np.abs(x).flatten()
                    for x in (pt.t2 if isinstance(pt.t2,tuple) else [pt.t2])])
                largest_t2=float(np.max(arr))
                nfmp=[int(np.sum(arr>t)) for t in [0.002,0.005,0.010]]
        except: pass

        e_a,e_b=mf.mo_energy[0],mf.mo_energy[1]
        occ=mf.mo_occ[0]+mf.mo_occ[1]
        oi=np.where(occ>0.5)[0]; vi=np.where(occ<0.5)[0]
        em=(e_a+e_b)/2
        homo=float(em[oi[-1]]) if len(oi) else 0.0
        lumo=float(em[vi[0]])  if len(vi) else 0.0
        S=spin/2.0
        sc=float(mf.spin_square()[0]-S*(S+1))

        result={
            'name':fname.replace('.json',''),
            'metal':metal,'ligand':ligand,'n_ligands':n_lig,
            'charge':charge,'spin':spin,'mult':spin+1,
            'dist_ang':dist,'geometry':geom,
            'n_electrons':mol.nelectron,'n_active_e':n_act,
            'E_HF':float(mf.e_tot),
            'E_CASSCF':float(mc.e_tot) if mc else float(mf.e_tot),
            'corr_energy':ec,'mp2_corr':mp2_corr,
            'converged':bool(conv),'n_active':n_active,
            'no_occ':[float(x) for x in no],'status':'ok',
            'z_eff':consts['z_eff'],'zeta_so_cm1':consts['zeta_so_cm1'],
            'metal_row':consts['metal_row'],
            'spin_contamination':sc,
            'homo_lumo_gap':float(lumo-homo),
            'homo_lumo_gap_eV':float((lumo-homo)*27.2114),
            'homo_energy':homo,'lumo_energy':lumo,'homo_ab_gap':0.0,
            'alpha_beta_overlap':0.0,'delta_E_HS_LS':0.0,
            'mulliken_metal_charge':0.0,'mayer_bond_order_mean':0.0,
            'mayer_bond_order_std':0.0,'largest_t2':largest_t2,
            'n_frac_uno_001':0,'n_frac_uno_005':0,
            'n_frac_uno_010':0,'n_frac_uno_020':0,
            'n_frac_mp2_002':nfmp[0],'n_frac_mp2_005':nfmp[1],
            'n_frac_mp2_010':nfmp[2],
            'd_electron_count':d_count,
            'expected_n_active':exp_n,
        }
        with open(outfile,'w') as f: json.dump(result,f,indent=2)
        log.info("  OK: n_active=%d (expected %d)" % (n_active,exp_n))
        return True
    except Exception as e:
        log.error("  Error: %s" % e)
        json.dump({'name':fname,'status':'error','reason':str(e)},
                  open(outfile,'w'))
        return False

ALL_JOBS=[(i,s[3]) for i,s in enumerate(STRUCTURES)]

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='summary':
        from collections import Counter
        by_gap=Counter()
        for i,(name,metal,*_,exp_n) in enumerate(STRUCTURES):
            gap='d1_V' if 'V' in name else \
                'd3_Re' if 'Re' in name else \
                'd3_Mo' if 'Mo' in name else \
                'd8_Ni_tet' if 'Ni' in name else 'Co_d6_LS'
            by_gap[gap]+=1
        print("\nGap-fill structures: %d total" % len(STRUCTURES))
        for gap,n in sorted(by_gap.items()):
            print("  %-20s %d" % (gap,n))
        print("\nExpected n_active values:")
        for name,metal,charge,spin,_,lig,_,_,geom,exp_n in STRUCTURES:
            print("  %-28s %s d%d spin=%d → exp=%d" % (
                name,metal,
                GROUP_NUMBER.get(metal,0)-(charge+atom_str.count('\nCl')+
                atom_str.count('\nBr')+atom_str.count('\nF')),
                spin,exp_n))
        sys.exit(0)
    idx=int(sys.argv[1]) if len(sys.argv)>1 else 0
    if idx>=len(STRUCTURES): sys.exit(0)
    sys.exit(0 if run_one(idx) else 1)
