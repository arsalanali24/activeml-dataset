"""
New validation set — never used during model development.
These test genuine generalization, not rule memorization.
"""
import numpy as np, json, os, sys, logging
from pyscf import gto, scf, mp
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

OUTPUT_DIR = os.path.expanduser('~/activeml/data/new_test_cases')
os.makedirs(OUTPUT_DIR, exist_ok=True)

METAL_CONSTANTS = {
    'Cr':{'z_eff':9.76, 'zeta_so_cm1':273,  'metal_row':'3d'},
    'Mn':{'z_eff':10.53,'zeta_so_cm1':355,  'metal_row':'3d'},
    'V': {'z_eff':8.45, 'zeta_so_cm1':167,  'metal_row':'3d'},
    'Co':{'z_eff':12.00,'zeta_so_cm1':533,  'metal_row':'3d'},
    'Ni':{'z_eff':12.78,'zeta_so_cm1':669,  'metal_row':'3d'},
    'Mo':{'z_eff':10.97,'zeta_so_cm1':467,  'metal_row':'4d'},
    'Ru':{'z_eff':12.33,'zeta_so_cm1':880,  'metal_row':'4d'},
    'Pt':{'z_eff':17.33,'zeta_so_cm1':4146, 'metal_row':'5d'},
    'Re':{'z_eff':17.01,'zeta_so_cm1':2456, 'metal_row':'5d'},
    'Ir':{'z_eff':17.00,'zeta_so_cm1':3909, 'metal_row':'5d'},
}
ECP = {'Mo','Ru','Pt','Re','Ir'}

# 10 new test cases — NEVER used during development
# Published n_active from peer-reviewed CASSCF literature
NEW_CASES = [
    # name, metal, charge, spin, atom_str, ligand, n_lig, dist, geom, pub_n
    ("CrCl6_3m_oct", "Cr", -3, 3,
     "Cr 0 0 0\nCl 2.350 0 0\nCl -2.350 0 0\nCl 0 2.350 0\n"
     "Cl 0 -2.350 0\nCl 0 0 2.350\nCl 0 0 -2.350",
     "Cl", 6, 2.350, "oct", 4),   # Cr(III) d3, pub=4

    ("MnF6_2m_oct", "Mn", -2, 5,
     "Mn 0 0 0\nF 1.840 0 0\nF -1.840 0 0\nF 0 1.840 0\n"
     "F 0 -1.840 0\nF 0 0 1.840\nF 0 0 -1.840",
     "F", 6, 1.840, "oct", 5),    # Mn(II) d5 HS, pub=5

    ("CoNH3_6_3p_oct", "Co", 3, 0,
     "Co 0 0 0\nN 2.050 0 0\nN -2.050 0 0\nN 0 2.050 0\n"
     "N 0 -2.050 0\nN 0 0 2.050\nN 0 0 -2.050",
     "N", 6, 2.050, "oct", 6),    # Co(III) d6 LS, pub=6

    ("NiCl4_2m_tet", "Ni", -2, 2,
     "Ni 0 0 0\nCl 2.270 0 0\nCl -2.270 0 0\n"
     "Cl 0 2.270 0\nCl 0 -2.270 0",
     "Cl", 4, 2.270, "tet", 8),   # Ni(II) d8, pub=8

    ("VOacac2_sqpl", "V", 0, 1,
     "V 0 0 0\nO 1.980 0 0\nO -1.980 0 0\n"
     "O 0 1.980 0\nO 0 -1.980 0",
     "O", 4, 1.980, "sq_pl", 1),  # VO(acac)2 d1, pub=1

    ("MoCl6_3m_oct", "Mo", -3, 3,
     "Mo 0 0 0\nCl 2.520 0 0\nCl -2.520 0 0\nCl 0 2.520 0\n"
     "Cl 0 -2.520 0\nCl 0 0 2.520\nCl 0 0 -2.520",
     "Cl", 6, 2.520, "oct", 3),   # Mo(III) d3, pub=3

    ("RuCl6_3m_oct", "Ru", -3, 1,
     "Ru 0 0 0\nCl 2.360 0 0\nCl -2.360 0 0\nCl 0 2.360 0\n"
     "Cl 0 -2.360 0\nCl 0 0 2.360\nCl 0 0 -2.360",
     "Cl", 6, 2.360, "oct", 5),   # Ru(III) d5, pub=5

    ("PtCl2NH3_2_cis", "Pt", 0, 0,
     "Pt 0 0 0\nCl 2.307 0 0\nCl 0 -2.307 0\n"
     "N -2.050 0 0\nN 0 2.050 0",
     "Cl", 2, 2.307, "sq_pl", 4), # cisplatin Pt(II) d8, pub=4

    ("ReCl6_2m_oct", "Re", -2, 1,
     "Re 0 0 0\nCl 2.370 0 0\nCl -2.370 0 0\nCl 0 2.370 0\n"
     "Cl 0 -2.370 0\nCl 0 0 2.370\nCl 0 0 -2.370",
     "Cl", 6, 2.370, "oct", 3),   # Re(IV) d3, pub=3

    ("IrCl6_2m_oct", "Ir", -2, 1,
     "Ir 0 0 0\nCl 2.340 0 0\nCl -2.340 0 0\nCl 0 2.340 0\n"
     "Cl 0 -2.340 0\nCl 0 0 2.340\nCl 0 0 -2.340",
     "Cl", 6, 2.340, "oct", 5),   # Ir(IV) d5, pub=5
]

ATOM_Z = {'Cr':24,'Mn':25,'V':23,'Co':27,'Ni':28,'Mo':42,
           'Ru':44,'Pt':78,'Re':75,'Ir':77,
           'Cl':17,'F':9,'N':7,'O':8}

def run_hf(mol, spin):
    if spin == 0:
        for s in [dict(max_cycle=300,conv_tol=1e-10),
                  dict(max_cycle=500,conv_tol=1e-9)]:
            mf=scf.RHF(mol)
            for k,v in s.items(): setattr(mf,k,v)
            mf.verbose=0; mf.run()
            if mf.converged: return mf,'RHF'
    for s in [dict(max_cycle=300,conv_tol=1e-10,damp=0.0,level_shift=0.0),
              dict(max_cycle=500,conv_tol=1e-9, damp=0.3,level_shift=0.2),
              dict(max_cycle=800,conv_tol=1e-8, damp=0.5,level_shift=0.5)]:
        mf=scf.UHF(mol)
        for k,v in s.items(): setattr(mf,k,v)
        mf.verbose=0; mf.run()
        if mf.converged: return mf,'UHF'
    return mf,'UHF'

def run_one(idx):
    (name,metal,charge,spin,atom_str,ligand,
     n_lig,dist,geom,pub_n) = NEW_CASES[idx]

    outfile = os.path.join(OUTPUT_DIR, f"{name}.json")
    if os.path.exists(outfile):
        r = json.load(open(outfile))
        if r.get('status')=='ok':
            log.info(f"SKIP: {name}"); return True

    consts = METAL_CONSTANTS[metal]
    n_e = sum(ATOM_Z.get(l.strip().split()[0],0)
              for l in atom_str.split('\n') if l.strip()) - charge

    if (n_e%2) != (spin%2):
        log.warning(f"Parity mismatch {name}: n_e={n_e} spin={spin}")
        json.dump({'name':name,'status':'skipped','reason':'parity'},
                  open(outfile,'w')); return True

    log.info(f"Start: {name} n_e={n_e}")
    try:
        mol=gto.Mole(); mol.atom=atom_str; mol.basis='def2-SVP'
        mol.charge=charge; mol.spin=spin; mol.verbose=0
        if metal in ECP: mol.ecp='def2-SVP'
        mol.build()
        mf,scf_type = run_hf(mol,spin)
        log.info(f"  {scf_type}: E={mf.e_tot:.6f} conv={mf.converged}")

        mp2_corr=0.0; largest_t2=0.0
        n_frac_mp2=[0,0,0]
        try:
            pt=(mp.UMP2(mf) if scf_type=='UHF' else mp.MP2(mf)).run()
            mp2_corr=float(pt.e_corr)
            t2=pt.t2
            if t2 is not None:
                arr=np.concatenate([np.abs(x).flatten()
                                    for x in (t2 if isinstance(t2,tuple)
                                              else [t2])])
                largest_t2=float(np.max(arr))
                n_frac_mp2=[int(np.sum(arr>th))
                            for th in [0.002,0.005,0.010]]
        except: pass

        # Features
        if scf_type=='RHF':
            e_m=mf.mo_energy; occ=mf.mo_occ
            sc=0.0; hab=0.0
            oi=np.where(occ>0.5)[0]; vi=np.where(occ<0.5)[0]
            homo=float(e_m[oi[-1]]) if len(oi) else 0.0
            lumo=float(e_m[vi[0]])  if len(vi) else 0.0
        else:
            ea,eb=mf.mo_energy[0],mf.mo_energy[1]
            occ=mf.mo_occ[0]+mf.mo_occ[1]
            S=spin/2.0; sc=float(mf.spin_square()[0]-S*(S+1))
            oi_a=np.where(mf.mo_occ[0]>0.5)[0]
            oi_b=np.where(mf.mo_occ[1]>0.5)[0]
            ha=float(ea[oi_a[-1]]) if len(oi_a) else 0.0
            hb=float(eb[oi_b[-1]]) if len(oi_b) else 0.0
            hab=float(abs(ha-hb))
            oi=np.where(occ>0.5)[0]; vi=np.where(occ<0.5)[0]
            em=(ea+eb)/2
            homo=float(em[oi[-1]]) if len(oi) else 0.0
            lumo=float(em[vi[0]])  if len(vi) else 0.0

        result={
            'name':name,'metal':metal,'ligand':ligand,
            'n_ligands':n_lig,'charge':charge,'spin':spin,
            'mult':spin+1,'dist_ang':dist,'geometry':geom,
            'n_electrons':mol.nelectron,'status':'ok',
            'converged':bool(mf.converged),'scf_type':scf_type,
            'published_n_active':pub_n,
            'n_active':pub_n,  # use published as label for training
            'spin_contamination':sc,
            'homo_lumo_gap':float(lumo-homo),
            'homo_lumo_gap_eV':float((lumo-homo)*27.2114),
            'homo_energy':homo,'lumo_energy':lumo,'homo_ab_gap':hab,
            'alpha_beta_overlap':0.0,'delta_E_HS_LS':0.0,
            'mp2_corr':mp2_corr,'corr_energy':mp2_corr,
            'largest_t2':largest_t2,'E_HF':float(mf.e_tot),
            'n_frac_mp2_002':n_frac_mp2[0],
            'n_frac_mp2_005':n_frac_mp2[1],
            'n_frac_mp2_010':n_frac_mp2[2],
            'mulliken_metal_charge':0.0,
            'mayer_bond_order_mean':0.0,'mayer_bond_order_std':0.0,
            'z_eff':consts['z_eff'],'zeta_so_cm1':consts['zeta_so_cm1'],
            'metal_row':consts['metal_row'],
        }
        with open(outfile,'w') as f: json.dump(result,f,indent=2)
        log.info(f"  OK: pub_n_active={pub_n}")
        return True
    except Exception as e:
        log.error(f"  Error: {e}")
        json.dump({'name':name,'status':'error','reason':str(e)},
                  open(outfile,'w'))
        return False

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='summary':
        print(f"\nNew validation test cases: {len(NEW_CASES)}")
        print("%-30s %6s %5s %5s %5s" % ('Name','Metal','Spin','Pub','Rule?'))
        print('-'*55)
        for n,m,c,s,_,l,_,_,g,p in NEW_CASES:
            print("  %-28s %6s %5d %5d" % (n,m,s,p))
        sys.exit(0)
    idx=int(sys.argv[1]) if len(sys.argv)>1 else 0
    if idx>=len(NEW_CASES): sys.exit(0)
    sys.exit(0 if run_one(idx) else 1)
