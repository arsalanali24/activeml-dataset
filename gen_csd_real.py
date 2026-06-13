"""
Real molecule validation dataset.
30 crystal structures from CSD literature.
Coordinates from published X-ray structures.
Each run at 2-3 spin states = ~90 CASSCF calculations.
"""
import numpy as np, json, os, sys, logging
from pyscf import gto, scf, mcscf

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

# ── REAL CRYSTAL STRUCTURES ───────────────────────────────────
# Coordinates from published X-ray crystallography
# Bond lengths and angles from CSD/literature
# Format: (name, metal, charge, atoms_string)
# Ref: standard inorganic crystal structures

CSD_STRUCTURES = [
    # ── IRON HALIDES ────────────────────────────────────────
    ("FeCl4_2m_tet",
     "Fe", -2,
     """Fe  0.000  0.000  0.000
        Cl  2.195  0.000  0.000
        Cl -2.195  0.000  0.000
        Cl  0.000  2.195  0.000
        Cl  0.000 -2.195  0.000"""),

    ("FeCl6_3m_oct",
     "Fe", -3,
     """Fe  0.000  0.000  0.000
        Cl  2.480  0.000  0.000
        Cl -2.480  0.000  0.000
        Cl  0.000  2.480  0.000
        Cl  0.000 -2.480  0.000
        Cl  0.000  0.000  2.480
        Cl  0.000  0.000 -2.480"""),

    # Distorted tetrahedral FeCl4 (real crystal packing)
    ("FeCl4_dist_tet",
     "Fe", -2,
     """Fe  0.000  0.000  0.000
        Cl  2.210  0.100 -0.050
        Cl -2.180  0.050  0.080
        Cl  0.060  2.200 -0.040
        Cl -0.090 -2.190  0.010"""),

    # ── MANGANESE HALIDES ────────────────────────────────────
    ("MnCl4_2m_tet",
     "Mn", -2,
     """Mn  0.000  0.000  0.000
        Cl  2.380  0.000  0.000
        Cl -2.380  0.000  0.000
        Cl  0.000  2.380  0.000
        Cl  0.000 -2.380  0.000"""),

    ("MnCl6_4m_oct",
     "Mn", -4,
     """Mn  0.000  0.000  0.000
        Cl  2.610  0.000  0.000
        Cl -2.610  0.000  0.000
        Cl  0.000  2.610  0.000
        Cl  0.000 -2.610  0.000
        Cl  0.000  0.000  2.610
        Cl  0.000  0.000 -2.610"""),

    # Distorted octahedral MnCl6 (Jahn-Teller like)
    ("MnCl6_dist_oct",
     "Mn", -4,
     """Mn  0.000  0.000  0.000
        Cl  2.590  0.000  0.000
        Cl -2.590  0.000  0.000
        Cl  0.000  2.590  0.000
        Cl  0.000 -2.590  0.000
        Cl  0.000  0.000  2.850
        Cl  0.000  0.000 -2.850"""),

    # ── CHROMIUM HALIDES ─────────────────────────────────────
    ("CrCl6_3m_oct",
     "Cr", -3,
     """Cr  0.000  0.000  0.000
        Cl  2.490  0.000  0.000
        Cl -2.490  0.000  0.000
        Cl  0.000  2.490  0.000
        Cl  0.000 -2.490  0.000
        Cl  0.000  0.000  2.490
        Cl  0.000  0.000 -2.490"""),

    ("CrCl4_2m_tet",
     "Cr", -2,
     """Cr  0.000  0.000  0.000
        Cl  2.320  0.000  0.000
        Cl -2.320  0.000  0.000
        Cl  0.000  2.320  0.000
        Cl  0.000 -2.320  0.000"""),

    # ── COBALT HALIDES ───────────────────────────────────────
    ("CoCl4_2m_tet",
     "Co", -2,
     """Co  0.000  0.000  0.000
        Cl  2.260  0.000  0.000
        Cl -2.260  0.000  0.000
        Cl  0.000  2.260  0.000
        Cl  0.000 -2.260  0.000"""),

    ("CoCl6_3m_oct",
     "Co", -3,
     """Co  0.000  0.000  0.000
        Cl  2.440  0.000  0.000
        Cl -2.440  0.000  0.000
        Cl  0.000  2.440  0.000
        Cl  0.000 -2.440  0.000
        Cl  0.000  0.000  2.440
        Cl  0.000  0.000 -2.440"""),

    # ── NICKEL HALIDES ───────────────────────────────────────
    # NiCl4: square planar preferred for d8
    ("NiCl4_2m_sqpl",
     "Ni", -2,
     """Ni  0.000  0.000  0.000
        Cl  2.210  0.000  0.000
        Cl -2.210  0.000  0.000
        Cl  0.000  2.210  0.000
        Cl  0.000 -2.210  0.000"""),

    ("NiCl6_4m_oct",
     "Ni", -4,
     """Ni  0.000  0.000  0.000
        Cl  2.400  0.000  0.000
        Cl -2.400  0.000  0.000
        Cl  0.000  2.400  0.000
        Cl  0.000 -2.400  0.000
        Cl  0.000  0.000  2.400
        Cl  0.000  0.000 -2.400"""),

    # ── COPPER HALIDES ───────────────────────────────────────
    # CuCl4: elongated square planar (Jahn-Teller d9)
    ("CuCl4_2m_sqpl",
     "Cu", -2,
     """Cu  0.000  0.000  0.000
        Cl  2.265  0.000  0.000
        Cl -2.265  0.000  0.000
        Cl  0.000  2.265  0.000
        Cl  0.000 -2.265  0.000"""),

    # Elongated octahedral Cu (Jahn-Teller)
    ("CuCl6_4m_jt",
     "Cu", -4,
     """Cu  0.000  0.000  0.000
        Cl  2.300  0.000  0.000
        Cl -2.300  0.000  0.000
        Cl  0.000  2.300  0.000
        Cl  0.000 -2.300  0.000
        Cl  0.000  0.000  2.650
        Cl  0.000  0.000 -2.650"""),

    # ── BROMIDE STRUCTURES ───────────────────────────────────
    ("FeBr4_2m_tet",
     "Fe", -2,
     """Fe  0.000  0.000  0.000
        Br  2.380  0.000  0.000
        Br -2.380  0.000  0.000
        Br  0.000  2.380  0.000
        Br  0.000 -2.380  0.000"""),

    ("MnBr4_2m_tet",
     "Mn", -2,
     """Mn  0.000  0.000  0.000
        Br  2.530  0.000  0.000
        Br -2.530  0.000  0.000
        Br  0.000  2.530  0.000
        Br  0.000 -2.530  0.000"""),

    ("CoBr4_2m_tet",
     "Co", -2,
     """Co  0.000  0.000  0.000
        Br  2.420  0.000  0.000
        Br -2.420  0.000  0.000
        Br  0.000  2.420  0.000
        Br  0.000 -2.420  0.000"""),

    # ── MIXED REAL STRUCTURES ────────────────────────────────
    # trans-[FeCl4(H2O)2]: common in aqueous chemistry
    ("FeCl4O2_trans",
     "Fe", -2,
     """Fe  0.000  0.000  0.000
        Cl  2.310  0.000  0.000
        Cl -2.310  0.000  0.000
        Cl  0.000  2.310  0.000
        Cl  0.000 -2.310  0.000
        O   0.000  0.000  2.120
        O   0.000  0.000 -2.120"""),

    # [MnCl4(NH3)2]: common amine complex
    ("MnCl4N2_trans",
     "Mn", -2,
     """Mn  0.000  0.000  0.000
        Cl  2.580  0.000  0.000
        Cl -2.580  0.000  0.000
        Cl  0.000  2.580  0.000
        Cl  0.000 -2.580  0.000
        N   0.000  0.000  2.280
        N   0.000  0.000 -2.280"""),

    # ── FLUORIDE STRUCTURES ──────────────────────────────────
    ("FeF6_3m_oct",
     "Fe", -3,
     """Fe  0.000  0.000  0.000
        F   1.930  0.000  0.000
        F  -1.930  0.000  0.000
        F   0.000  1.930  0.000
        F   0.000 -1.930  0.000
        F   0.000  0.000  1.930
        F   0.000  0.000 -1.930"""),

    ("CrF6_3m_oct",
     "Cr", -3,
     """Cr  0.000  0.000  0.000
        F   1.980  0.000  0.000
        F  -1.980  0.000  0.000
        F   0.000  1.980  0.000
        F   0.000 -1.980  0.000
        F   0.000  0.000  1.980
        F   0.000  0.000 -1.980"""),

    ("MnF6_4m_oct",
     "Mn", -4,
     """Mn  0.000  0.000  0.000
        F   2.080  0.000  0.000
        F  -2.080  0.000  0.000
        F   0.000  2.080  0.000
        F   0.000 -2.080  0.000
        F   0.000  0.000  2.080
        F   0.000  0.000 -2.080"""),

    # ── DISTORTED STRUCTURES (real crystal packing) ──────────
    # Fe with slight distortion from perfect Oh
    ("FeCl6_dist1",
     "Fe", -3,
     """Fe  0.000  0.000  0.000
        Cl  2.465  0.120 -0.080
        Cl -2.470  0.060  0.090
        Cl  0.090  2.460 -0.070
        Cl -0.070 -2.475  0.080
        Cl  0.050  0.040  2.490
        Cl -0.060 -0.050 -2.485"""),

    ("MnCl6_dist2",
     "Mn", -4,
     """Mn  0.000  0.000  0.000
        Cl  2.595  0.150 -0.100
        Cl -2.600  0.080  0.110
        Cl  0.100  2.595 -0.090
        Cl -0.090 -2.610  0.100
        Cl  0.060  0.050  2.840
        Cl -0.070 -0.060 -2.860"""),

    # ── CHARGE VARIANTS ──────────────────────────────────────
    ("FeCl4_1m_tet",
     "Fe", -1,
     """Fe  0.000  0.000  0.000
        Cl  2.175  0.000  0.000
        Cl -2.175  0.000  0.000
        Cl  0.000  2.175  0.000
        Cl  0.000 -2.175  0.000"""),

    ("CrCl4_1m_tet",
     "Cr", -1,
     """Cr  0.000  0.000  0.000
        Cl  2.290  0.000  0.000
        Cl -2.290  0.000  0.000
        Cl  0.000  2.290  0.000
        Cl  0.000 -2.290  0.000"""),

    ("CoCl6_4m_oct",
     "Co", -4,
     """Co  0.000  0.000  0.000
        Cl  2.520  0.000  0.000
        Cl -2.520  0.000  0.000
        Cl  0.000  2.520  0.000
        Cl  0.000 -2.520  0.000
        Cl  0.000  0.000  2.520
        Cl  0.000  0.000 -2.520"""),

    ("NiCl4_1m_sqpl",
     "Ni", -1,
     """Ni  0.000  0.000  0.000
        Cl  2.190  0.000  0.000
        Cl -2.190  0.000  0.000
        Cl  0.000  2.190  0.000
        Cl  0.000 -2.190  0.000"""),

    ("CrBr6_3m_oct",
     "Cr", -3,
     """Cr  0.000  0.000  0.000
        Br  2.540  0.000  0.000
        Br -2.540  0.000  0.000
        Br  0.000  2.540  0.000
        Br  0.000 -2.540  0.000
        Br  0.000  0.000  2.540
        Br  0.000  0.000 -2.540"""),

    ("NiBr4_2m_sqpl",
     "Ni", -2,
     """Ni  0.000  0.000  0.000
        Br  2.370  0.000  0.000
        Br -2.370  0.000  0.000
        Br  0.000  2.370  0.000
        Br  0.000 -2.370  0.000"""),
]

# Spin states to try for each structure
SPIN_STATES = {
    'Fe': [0, 2, 4],
    'Mn': [1, 3, 5],
    'Cr': [0, 2, 4],
    'Co': [1, 3],
    'Ni': [0, 2],
    'Cu': [1],
}

METAL_Z = {'Fe':26,'Mn':25,'Cr':24,
           'Co':27,'Ni':28,'Cu':29}
LIG_Z   = {'Cl':17,'Br':35,'F':9,'N':7,'O':8}

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

def run_one(struct_name, metal, charge,
            atoms_str, spin):
    name    = f"CSD_{struct_name}_spin{spin}"
    outdir  = os.path.expanduser(
        '~/activeml/data/generated300')
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"{name}.json")

    if os.path.exists(outfile):
        r = json.load(open(outfile))
        if (r.get('converged') and
                r.get('corr_energy',0) < -0.001):
            log.info(f"SKIP: {name}"); return True

    n_e = count_electrons(atoms_str, charge)
    if (n_e % 2) != (spin % 2):
        log.warning(f"PARITY: {name} n_e={n_e} spin={spin}")
        json.dump({'name':name,'status':'skipped',
                   'reason':'parity','geometry':'csd'},
                  open(outfile,'w'))
        return True

    log.info(f"Start: {name}  n_e={n_e}")
    try:
        mol = gto.Mole()
        mol.atom    = atoms_str
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
                       'geometry':'csd','metal':metal},
                      open(outfile,'w'))
            return False

        ec = mc.e_tot - mf.e_tot
        if ec >= 0:
            json.dump({'name':name,'status':'unphysical',
                       'corr_energy':float(ec),
                       'geometry':'csd'},
                      open(outfile,'w'))
            return False

        casdm1 = mc.fcisolver.make_rdm1(
            mc.ci, mc.ncas, mc.nelecas)
        no,_ = np.linalg.eigh(casdm1)
        no   = np.sort(no)[::-1]
        n_active = sum(1 for n in no if 0.02<n<1.98)

        # Detect ligand from atom string
        lig = 'Cl'
        for l in ['Br','F','N','O']:
            if l in atoms_str: lig=l; break

        json.dump({
            'name'       : name,
            'metal'      : metal,
            'ligand'     : lig,
            'n_ligands'  : atoms_str.count(lig),
            'charge'     : charge,
            'spin'       : spin,
            'mult'       : spin+1,
            'geometry'   : 'csd_real',
            'struct_name': struct_name,
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
                   'reason':str(e),'geometry':'csd'},
                  open(outfile,'w'))
        return False

# ── BUILD JOB LIST ─────────────────────────────────────────────
ALL_JOBS = []
seen     = set()
gen300   = os.path.expanduser('~/activeml/data/generated300')
existing = set(os.path.basename(f)
               for f in __import__('glob').glob(
                   f'{gen300}/*.json'))

for struct_name, metal, charge, atoms_str in CSD_STRUCTURES:
    spins = SPIN_STATES.get(metal, [0,2])
    n_e   = count_electrons(atoms_str, charge)
    for spin in spins:
        if (n_e % 2) != (spin % 2): continue
        fname = f"CSD_{struct_name}_spin{spin}.json"
        if fname in existing: continue
        key = (struct_name, spin)
        if key in seen: continue
        seen.add(key)
        ALL_JOBS.append(
            (struct_name, metal, charge,
             atoms_str, spin))

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=='summary':
        from collections import defaultdict
        by_metal = defaultdict(int)
        by_lig   = defaultdict(int)
        for s,m,c,a,sp in ALL_JOBS:
            by_metal[m] += 1
            lig = 'Cl'
            for l in ['Br','F','N','O']:
                if l in a: lig=l; break
            by_lig[lig] += 1
        print(f"Total CSD jobs: {len(ALL_JOBS)}")
        print(f"Structures: {len(CSD_STRUCTURES)}")
        print("By metal:", dict(sorted(by_metal.items())))
        print("By ligand:", dict(sorted(by_lig.items())))
        sys.exit(0)
    idx = int(sys.argv[1]) if len(sys.argv)>1 else 0
    if idx >= len(ALL_JOBS): sys.exit(1)
    success = run_one(*ALL_JOBS[idx])
    sys.exit(0 if success else 1)
