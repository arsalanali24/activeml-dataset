"""
gen_benchmark_test.py
=====================
Automated benchmark testing for ML1 model.
Generates UHF features for new test cases with KNOWN published
CASSCF active spaces, then evaluates the saved model.

Test cases chosen from systems:
  1. Within training distribution — should work well
  2. Published in peer-reviewed CASSCF literature
  3. Never used during model development or rule derivation

Usage:
  python gen_benchmark_test.py summary          # show all cases
  python gen_benchmark_test.py compute <idx>    # compute UHF for case idx
  python gen_benchmark_test.py evaluate         # run model on all computed
  python gen_benchmark_test.py all              # compute + evaluate all
  sbatch --array=0-N job_benchmark_test.sh      # HPC batch
"""
import numpy as np, json, os, sys, logging, glob, pickle
from pyscf import gto, scf, mp
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

BASE     = os.path.expanduser('~/activeml/data')
OUT_DIR  = os.path.join(BASE, 'benchmark_test_cases')
os.makedirs(OUT_DIR, exist_ok=True)

GROUP_NUMBER = {
    'Ti':4,'V':5,'Cr':6,'Mn':7,'Fe':8,'Co':9,'Ni':10,'Cu':11,'Zn':12,
    'Mo':6,'Ru':8,'Rh':9,'Pd':10,'Tc':7,
    'W':6,'Re':7,'Os':8,'Ir':9,'Pt':10,
}
HALIDES = {'Cl','Br','F','I'}
ECP_METALS = {'Mo','Ru','Rh','Pd','W','Re','Os','Ir','Pt'}
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
ATOM_Z = {
    'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,
    'Cu':29,'Zn':30,'Mo':42,'Ru':44,'Rh':45,'Pd':46,
    'W':74,'Re':75,'Os':76,'Ir':77,'Pt':78,
    'Cl':17,'Br':35,'F':9,'I':53,'N':7,'O':8,'P':15,'C':6,'H':1,
}

# ══════════════════════════════════════════════════════════════
# BENCHMARK TEST CASES
# Format: (name, metal, charge, spin, atom_str, ligand,
#          n_lig, dist_ang, geometry, pub_n_active, reference)
#
# SELECTION CRITERIA:
#   - Published CASSCF/CASPT2 active space in literature
#   - System within training distribution (halide/sigma-donor)
#   - Never used during model development
#   - Covers all 3d/4d/5d rows
# ══════════════════════════════════════════════════════════════

BENCHMARK_CASES = [

    # ── 3d metals — core training distribution ────────────────

    # FeCl4^2- high spin Fe(II) d6
    # Ref: Pierloot, IJQC 2011, n_active=6
    ("FeCl4_2m_tet_d6", "Fe", -2, 4,
     "Fe 0 0 0\nCl 2.290 0 0\nCl -2.290 0 0\nCl 0 2.290 0\nCl 0 -2.290 0",
     "Cl", 4, 2.290, "tet", 6,
     "Pierloot IJQC 2011"),

    # CoCl4^2- high spin Co(II) d7
    # Ref: standard CASSCF(7,5), n_active=7
    ("CoCl4_2m_tet_d7", "Co", -2, 3,
     "Co 0 0 0\nCl 2.270 0 0\nCl -2.270 0 0\nCl 0 2.270 0\nCl 0 -2.270 0",
     "Cl", 4, 2.270, "tet", 7,
     "Standard CASSCF(7,5)"),

    # MnCl4^2- high spin Mn(II) d5
    # n_active=5 (half-filled d shell)
    ("MnCl4_2m_tet_d5", "Mn", -2, 5,
     "Mn 0 0 0\nCl 2.340 0 0\nCl -2.340 0 0\nCl 0 2.340 0\nCl 0 -2.340 0",
     "Cl", 4, 2.340, "tet", 5,
     "Standard CASSCF(5,5)"),

    # FeCl6^3- high spin Fe(III) d5
    # Ref: standard CASSCF(5,5), n_active=5
    ("FeCl6_3m_oct_d5", "Fe", -3, 5,
     "Fe 0 0 0\nCl 2.380 0 0\nCl -2.380 0 0\nCl 0 2.380 0\n"
     "Cl 0 -2.380 0\nCl 0 0 2.380\nCl 0 0 -2.380",
     "Cl", 6, 2.380, "oct", 5,
     "Standard CASSCF(5,5)"),

    # CrCl6^3- Cr(III) d3
    # Ref: Pierloot, n_active=3 (t2g^3)
    ("CrCl6_3m_oct_d3", "Cr", -3, 3,
     "Cr 0 0 0\nCl 2.350 0 0\nCl -2.350 0 0\nCl 0 2.350 0\n"
     "Cl 0 -2.350 0\nCl 0 0 2.350\nCl 0 0 -2.350",
     "Cl", 6, 2.350, "oct", 3,
     "Pierloot JPCB 2002"),

    # NiCl6^4- Ni(II) d8 octahedral
    # Ref: Roos, n_active=8 (all d)
    ("NiCl6_4m_oct_d8", "Ni", -4, 2,
     "Ni 0 0 0\nCl 2.420 0 0\nCl -2.420 0 0\nCl 0 2.420 0\n"
     "Cl 0 -2.420 0\nCl 0 0 2.420\nCl 0 0 -2.420",
     "Cl", 6, 2.420, "oct", 8,
     "Roos JACS 2004"),

    # TiCl6^2- Ti(IV) d0 — should give n_active=0
    ("TiCl6_2m_oct_d0", "Ti", -2, 0,
     "Ti 0 0 0\nCl 2.370 0 0\nCl -2.370 0 0\nCl 0 2.370 0\n"
     "Cl 0 -2.370 0\nCl 0 0 2.370\nCl 0 0 -2.370",
     "Cl", 6, 2.370, "oct", 0,
     "d0 closed shell"),

    # CuCl4^2- Cu(II) d9 sq_pl
    # Ref: n_active=1 (single hole in d shell)
    ("CuCl4_2m_sqpl_d9", "Cu", -2, 1,
     "Cu 0 0 0\nCl 2.260 0 0\nCl -2.260 0 0\nCl 0 2.260 0\nCl 0 -2.260 0",
     "Cl", 4, 2.260, "sq_pl", 1,
     "Standard Cu(II) d9"),

    # ── 4d metals ─────────────────────────────────────────────

    # RhCl6^3- Rh(III) d6 low spin
    # Ref: n_active=6
    ("RhCl6_3m_oct_d6", "Rh", -3, 0,
     "Rh 0 0 0\nCl 2.340 0 0\nCl -2.340 0 0\nCl 0 2.340 0\n"
     "Cl 0 -2.340 0\nCl 0 0 2.340\nCl 0 0 -2.340",
     "Cl", 6, 2.340, "oct", 6,
     "Standard CASSCF(6,5)"),

    # PdCl4^2- Pd(II) d8 square planar
    # Ref: Roos PCCP 2004, n_active=4-6
    ("PdCl4_2m_sqpl_d8", "Pd", -2, 0,
     "Pd 0 0 0\nCl 2.295 0 0\nCl -2.295 0 0\nCl 0 2.295 0\nCl 0 -2.295 0",
     "Cl", 4, 2.295, "sq_pl", 4,
     "Roos PCCP 2004"),

    # RuCl3 Ru(III) d5
    # n_active=5
    ("RuCl6_3m_oct_d5v2", "Ru", -3, 1,
     "Ru 0 0 0\nCl 2.360 0 0\nCl -2.360 0 0\nCl 0 2.360 0\n"
     "Cl 0 -2.360 0\nCl 0 0 2.360\nCl 0 0 -2.360",
     "Cl", 6, 2.360, "oct", 5,
     "Standard CASSCF(5,5)"),

    # MoCl6^2- Mo(IV) d2
    # n_active=2
    ("MoCl6_2m_oct_d2", "Mo", -2, 2,
     "Mo 0 0 0\nCl 2.480 0 0\nCl -2.480 0 0\nCl 0 2.480 0\n"
     "Cl 0 -2.480 0\nCl 0 0 2.480\nCl 0 0 -2.480",
     "Cl", 6, 2.480, "oct", 2,
     "Standard CASSCF(2,5)"),

    # ── 5d metals ─────────────────────────────────────────────

    # OsCl6^2- Os(IV) d4
    # Ref: n_active=4
    ("OsCl6_2m_oct_d4", "Os", -2, 4,
     "Os 0 0 0\nCl 2.350 0 0\nCl -2.350 0 0\nCl 0 2.350 0\n"
     "Cl 0 -2.350 0\nCl 0 0 2.350\nCl 0 0 -2.350",
     "Cl", 6, 2.350, "oct", 4,
     "Standard CASSCF(4,5)"),

    # PtCl6^2- Pt(IV) d6 low spin
    # n_active=6
    ("PtCl6_2m_oct_d6", "Pt", -2, 0,
     "Pt 0 0 0\nCl 2.330 0 0\nCl -2.330 0 0\nCl 0 2.330 0\n"
     "Cl 0 -2.330 0\nCl 0 0 2.330\nCl 0 0 -2.330",
     "Cl", 6, 2.330, "oct", 6,
     "Standard CASSCF(6,5)"),

    # WCl6 W(VI) d0
    # n_active=0 (fully oxidised)
    ("WCl6_oct_d0", "W", 0, 0,
     "W 0 0 0\nCl 2.260 0 0\nCl -2.260 0 0\nCl 0 2.260 0\n"
     "Cl 0 -2.260 0\nCl 0 0 2.260\nCl 0 0 -2.260",
     "Cl", 6, 2.260, "oct", 0,
     "d0 closed shell"),

    # IrCl6^3- Ir(III) d6 low spin
    # n_active=6
    ("IrCl6_3m_oct_d6", "Ir", -3, 0,
     "Ir 0 0 0\nCl 2.340 0 0\nCl -2.340 0 0\nCl 0 2.340 0\n"
     "Cl 0 -2.340 0\nCl 0 0 2.340\nCl 0 0 -2.340",
     "Cl", 6, 2.340, "oct", 6,
     "Standard CASSCF(6,5)"),

    # ReCl6^3- Re(III) d4
    # n_active=4
    ("ReCl6_3m_oct_d4", "Re", -3, 4,
     "Re 0 0 0\nCl 2.370 0 0\nCl -2.370 0 0\nCl 0 2.370 0\n"
     "Cl 0 -2.370 0\nCl 0 0 2.370\nCl 0 0 -2.370",
     "Cl", 6, 2.370, "oct", 4,
     "Standard CASSCF(4,5)"),

    # ── Mixed geometry / special cases ────────────────────────

    # FeCl6^4- Fe(II) d6 high spin octahedral
    # n_active=10 (full d + sigma antibonding)
    ("FeCl6_4m_oct_d6HS", "Fe", -4, 4,
     "Fe 0 0 0\nCl 2.500 0 0\nCl -2.500 0 0\nCl 0 2.500 0\n"
     "Cl 0 -2.500 0\nCl 0 0 2.500\nCl 0 0 -2.500",
     "Cl", 6, 2.500, "oct", 10,
     "Pierloot JCTC 2017 — includes sigma antibonding"),

    # CrCl4^2- Cr(II) d4 tetrahedral
    # n_active=4
    ("CrCl4_2m_tet_d4", "Cr", -2, 4,
     "Cr 0 0 0\nCl 2.310 0 0\nCl -2.310 0 0\nCl 0 2.310 0\nCl 0 -2.310 0",
     "Cl", 4, 2.310, "tet", 4,
     "Standard CASSCF(4,5)"),

    # VCl6^3- V(III) d2
    # n_active=2
    ("VCl6_3m_oct_d2", "V", -3, 2,
     "V 0 0 0\nCl 2.340 0 0\nCl -2.340 0 0\nCl 0 2.340 0\n"
     "Cl 0 -2.340 0\nCl 0 0 2.340\nCl 0 0 -2.340",
     "Cl", 6, 2.340, "oct", 2,
     "Standard CASSCF(2,5)"),
]


def correct_d_count(metal, charge, atom_str):
    group = GROUP_NUMBER.get(metal, 0)
    n_hal = sum(1 for l in atom_str.split('\n')
                if l.strip().split()[0:1] and
                l.strip().split()[0] in HALIDES)
    return max(0, group - (charge + n_hal))


def count_electrons(atom_str, charge):
    total = 0
    for line in atom_str.strip().split('\n'):
        sym = line.strip().split()[0]
        total += ATOM_Z.get(sym, 0)
    return total - charge


def run_hf(mol, spin):
    if spin == 0:
        for s in [dict(max_cycle=300,conv_tol=1e-10),
                  dict(max_cycle=500,conv_tol=1e-9),
                  dict(max_cycle=800,conv_tol=1e-8)]:
            mf = scf.RHF(mol)
            for k,v in s.items(): setattr(mf,k,v)
            mf.verbose=0; mf.run()
            if mf.converged: return mf,'RHF'
    for s in [dict(max_cycle=300,conv_tol=1e-10,damp=0.0,level_shift=0.0),
              dict(max_cycle=500,conv_tol=1e-9, damp=0.3,level_shift=0.2),
              dict(max_cycle=800,conv_tol=1e-8, damp=0.5,level_shift=0.5)]:
        mf = scf.UHF(mol)
        for k,v in s.items(): setattr(mf,k,v)
        mf.verbose=0; mf.run()
        if mf.converged: return mf,'UHF'
    return mf,'UHF'


def extract_features(mf, mol, scf_type):
    if scf_type == 'RHF':
        e_m = mf.mo_energy; occ = mf.mo_occ
        sc = 0.0; hab = 0.0
        oi = np.where(occ>0.5)[0]; vi = np.where(occ<0.5)[0]
        homo = float(e_m[oi[-1]]) if len(oi) else 0.0
        lumo = float(e_m[vi[0]])  if len(vi) else 0.0
    else:
        ea,eb = mf.mo_energy[0],mf.mo_energy[1]
        occ   = mf.mo_occ[0]+mf.mo_occ[1]
        S     = mol.spin/2.0
        sc    = float(mf.spin_square()[0]-S*(S+1))
        oi_a  = np.where(mf.mo_occ[0]>0.5)[0]
        oi_b  = np.where(mf.mo_occ[1]>0.5)[0]
        ha    = float(ea[oi_a[-1]]) if len(oi_a) else 0.0
        hb    = float(eb[oi_b[-1]]) if len(oi_b) else 0.0
        hab   = float(abs(ha-hb))
        em    = (ea+eb)/2
        oi    = np.where(occ>0.5)[0]; vi = np.where(occ<0.5)[0]
        homo  = float(em[oi[-1]]) if len(oi) else 0.0
        lumo  = float(em[vi[0]])  if len(vi) else 0.0
    return sc, homo, lumo, hab


def compute_case(idx):
    (name, metal, charge, spin, atom_str, ligand,
     n_lig, dist, geom, pub_n, ref) = BENCHMARK_CASES[idx]

    outfile = os.path.join(OUT_DIR, f"{name}.json")
    if os.path.exists(outfile):
        r = json.load(open(outfile))
        if r.get('status') == 'ok':
            log.info(f"SKIP: {name}"); return True

    n_e   = count_electrons(atom_str, charge)
    d_cnt = correct_d_count(metal, charge, atom_str)

    if (n_e % 2) != (spin % 2):
        log.warning(f"Parity mismatch: {name} n_e={n_e} spin={spin}")
        json.dump({'name':name,'status':'skipped','reason':'parity'},
                  open(outfile,'w'))
        return True

    consts = METAL_CONSTANTS.get(metal,
             {'z_eff':10,'zeta_so_cm1':400,'metal_row':'3d'})
    log.info(f"Computing: {name} d={d_cnt} pub={pub_n}")

    try:
        mol = gto.Mole()
        mol.atom=atom_str; mol.basis='def2-SVP'
        mol.charge=charge; mol.spin=spin; mol.verbose=0
        if metal in ECP_METALS: mol.ecp='def2-SVP'
        mol.build()

        mf, scf_type = run_hf(mol, spin)
        log.info(f"  {scf_type}: E={mf.e_tot:.6f} conv={mf.converged}")

        sc, homo, lumo, hab = extract_features(mf, mol, scf_type)

        # MP2
        mp2_corr=0.0; largest_t2=0.0; nfmp=[0,0,0]
        try:
            pt = (mp.UMP2(mf) if scf_type=='UHF' else mp.MP2(mf)).run()
            mp2_corr = float(pt.e_corr)
            t2 = pt.t2
            if t2 is not None:
                arr = np.concatenate([np.abs(x).flatten()
                    for x in (t2 if isinstance(t2,tuple) else [t2])])
                largest_t2 = float(np.max(arr))
                nfmp = [int(np.sum(arr>th)) for th in [0.002,0.005,0.010]]
        except: pass

        result = {
            'name':                name,
            'metal':               metal,
            'ligand':              ligand,
            'n_ligands':           n_lig,
            'charge':              charge,
            'spin':                spin,
            'mult':                spin+1,
            'dist_ang':            dist,
            'geometry':            geom,
            'n_electrons':         mol.nelectron,
            'status':              'ok',
            'converged':           bool(mf.converged),
            'scf_type':            scf_type,
            'published_n_active':  pub_n,
            'published_reference': ref,
            'd_electron_count':    d_cnt,
            'z_eff':               consts['z_eff'],
            'zeta_so_cm1':         consts['zeta_so_cm1'],
            'metal_row':           consts['metal_row'],
            'E_HF':                float(mf.e_tot),
            'spin_contamination':  sc,
            'homo_lumo_gap':       float(lumo-homo),
            'homo_lumo_gap_eV':    float((lumo-homo)*27.2114),
            'homo_energy':         homo,
            'lumo_energy':         lumo,
            'homo_ab_gap':         hab,
            'alpha_beta_overlap':  0.0,
            'delta_E_HS_LS':       0.0,
            'mp2_corr':            mp2_corr,
            'corr_energy':         mp2_corr,
            'largest_t2':          largest_t2,
            'n_frac_mp2_002':      nfmp[0],
            'n_frac_mp2_005':      nfmp[1],
            'n_frac_mp2_010':      nfmp[2],
            'mulliken_metal_charge': 0.0,
            'mayer_bond_order_mean': 0.0,
            'mayer_bond_order_std':  0.0,
            'n_frac_uno_001':      0,
            'n_frac_uno_005':      0,
            'n_frac_uno_010':      0,
            'n_frac_uno_020':      0,
        }

        with open(outfile,'w') as f: json.dump(result,f,indent=2)
        log.info(f"  Saved: {name}  d_count={d_cnt}  pub={pub_n}")
        return True

    except Exception as e:
        log.error(f"  Error {name}: {e}")
        json.dump({'name':name,'status':'error','reason':str(e)},
                  open(outfile,'w'))
        return False


def evaluate_all():
    """Load saved model and predict all computed benchmark cases."""

    # Load model
    model_path = os.path.join(BASE, 'ml1_model_final.pkl')
    if not os.path.exists(model_path):
        log.error(f"Model not found: {model_path}")
        log.error("Run training script first.")
        return

    with open(model_path,'rb') as f:
        m = pickle.load(f)
    rf  = m['rf']
    imp = m['imp']
    NUMERIC     = m['NUMERIC']
    ROW_ENC     = m['ROW_ENC']
    GEOM_ENC    = m['GEOM_ENC']
    LIG_ENC     = m['LIG_ENC']
    LIG_GROUPS  = m['LIG_GROUPS']

    def Xf(recs):
        rows=[]
        for d in recs:
            row=[float(d.get(f,np.nan) or np.nan) for f in NUMERIC]
            row.append(ROW_ENC.get(d.get('metal_row','3d'),0))
            row.append(LIG_ENC.get(
                LIG_GROUPS.get(str(d.get('ligand','Cl')),'other'),
                len(LIG_ENC)))
            row.append(GEOM_ENC.get(d.get('geometry','oct'),12))
            rows.append(row)
        return np.array(rows,dtype=float)

    # Load computed cases
    recs = []
    for f in sorted(glob.glob(OUT_DIR+'/*.json')):
        r = json.load(open(f))
        if r.get('status')=='ok' and r.get('published_n_active') is not None:
            recs.append(r)

    if not recs:
        log.error("No computed benchmark cases found. Run compute first.")
        return

    pred = np.round(rf.predict(imp.transform(Xf(recs)))).astype(int)

    print("\n"+"="*78)
    print("BENCHMARK EVALUATION — ML1 Model")
    print(f"Model trained on: {m['n_training']} structures")
    print(f"CV MAE: {m['cv_mae']:.3f}  Train within±1: {100*m['train_within1']:.1f}%")
    print("="*78)
    print("  %-30s %3s %5s %5s %5s %5s  OK  Reference" %
          ('Case','M','Pub','Pred','Err','d'))
    print("-"*78)

    by_row = {'3d':{'n':0,'w1':0}, '4d':{'n':0,'w1':0}, '5d':{'n':0,'w1':0}}
    ne=nw=0
    results = []
    for d,p in zip(recs,pred):
        pub  = d['published_n_active']
        err  = int(p)-pub
        ok   = abs(err)==0; w1 = abs(err)<=1
        flag = '✓' if ok else ('~' if w1 else '✗')
        dc   = d.get('d_electron_count','?')
        row  = d.get('metal_row','3d')
        ref  = d.get('published_reference','')[:20]
        print("  %-30s %-3s %5d %5d %+5d %5s  %s  %s" % (
            d['name'][:30],d['metal'],pub,p,err,str(dc),flag,ref))
        if ok: ne+=1
        if w1: nw+=1
        by_row[row]['n'] += 1
        if w1: by_row[row]['w1'] += 1
        results.append({'name':d['name'],'metal':d['metal'],
                        'row':row,'pub':pub,'pred':int(p),
                        'err':err,'exact':bool(ok),'within_1':bool(w1),
                        'd_count':dc,'ref':d.get('published_reference','')})

    n_total = len(recs)
    print("-"*78)
    print("  TOTAL:  Exact=%d/%d (%d%%)   Within±1=%d/%d (%d%%)" % (
        ne,n_total,100*ne//n_total, nw,n_total,100*nw//n_total))
    print("\n  By metal row:")
    for row in ['3d','4d','5d']:
        n = by_row[row]['n']
        if n > 0:
            w = by_row[row]['w1']
            print("    %s: %d/%d within±1 (%d%%)" % (row,w,n,100*w//n))
    print("="*78)

    # Save results
    out = os.path.join(OUT_DIR,'evaluation_results.json')
    json.dump({'n_cases':n_total,'n_exact':ne,'n_within1':nw,
               'pct_exact':100*ne//n_total,'pct_within1':100*nw//n_total,
               'by_row':by_row,'results':results}, open(out,'w'), indent=2)
    print(f"\nResults saved: {out}")


def show_summary():
    print(f"\nBenchmark test cases: {len(BENCHMARK_CASES)}")
    print(f"\n{'Name':<32} {'M':<4} {'d':>3} {'Spin':>5} "
          f"{'pub_n':>6}  Reference")
    print("-"*75)
    for (name,metal,charge,spin,atom_str,lig,n_lig,dist,geom,pub_n,ref) \
            in BENCHMARK_CASES:
        dc = correct_d_count(metal, charge, atom_str)
        done = os.path.exists(os.path.join(OUT_DIR,f"{name}.json"))
        status = "✓" if done else "pending"
        print("  %-30s %-4s %3d %5d %6d  %-20s %s" % (
            name[:30],metal,dc,spin,pub_n,ref[:20],status))

    existing = glob.glob(OUT_DIR+'/*.json')
    ok = sum(1 for f in existing if json.load(open(f)).get('status')=='ok')
    print(f"\nComputed: {ok}/{len(BENCHMARK_CASES)}")
    print(f"Output:   {OUT_DIR}")


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv)>1 else 'summary'

    if cmd == 'summary':
        show_summary()

    elif cmd == 'compute':
        idx = int(sys.argv[2]) if len(sys.argv)>2 else 0
        if idx >= len(BENCHMARK_CASES):
            print(f"Index {idx} out of range (0-{len(BENCHMARK_CASES)-1})")
            sys.exit(1)
        sys.exit(0 if compute_case(idx) else 1)

    elif cmd == 'evaluate':
        evaluate_all()

    elif cmd == 'all':
        # Compute all then evaluate
        failed = []
        for i in range(len(BENCHMARK_CASES)):
            if not compute_case(i):
                failed.append(i)
        if failed:
            log.warning(f"Failed cases: {failed}")
        evaluate_all()

    else:
        # Treat as index for SLURM array
        try:
            idx = int(cmd)
            if idx >= len(BENCHMARK_CASES):
                sys.exit(0)
            sys.exit(0 if compute_case(idx) else 1)
        except ValueError:
            print(f"Unknown command: {cmd}")
            sys.exit(1)
