"""
New ligand expansion: N, O, S, C donors
matching SC1MC chemical space.
ALL systems verified for parity before submission.
"""
import numpy as np, json, os, sys, logging
from pyscf import gto, scf, mcscf

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ── CONSTANTS ─────────────────────────────────────────────────
METAL_Z = {'Fe':26,'Mn':25,'Cr':24,'Co':27,
           'Ni':28,'Cu':29,'Zn':30}
LIG_Z   = {'Cl':17,'Br':35,'F':9,'I':53,
           'N':7,'O':8,'S':16,'C':6}
LIG_CHARGE = {'Cl':-1,'Br':-1,'F':-1,'I':-1,
              'N':-3,'O':-2,'S':-2,'C':-4}

# Equilibrium M-L distances for new ligands (Angstrom)
# N: represents NH3/CN coordination
# O: represents H2O/OH coordination
# S: represents S2-/SCN coordination
# C: represents CO/CN coordination
EQ_DIST = {
    'Fe':{'Cl':2.18,'Br':2.35,'F':1.85,'I':2.55,
          'N':2.10,'O':2.05,'S':2.35,'C':1.90},
    'Mn':{'Cl':2.35,'Br':2.50,'F':1.98,'I':2.70,
          'N':2.20,'O':2.15,'S':2.45,'C':2.00},
    'Cr':{'Cl':2.31,'Br':2.47,'F':1.94,'I':2.65,
          'N':2.10,'O':2.05,'S':2.40,'C':1.93},
    'Co':{'Cl':2.26,'Br':2.42,'F':1.90,'I':2.60,
          'N':2.00,'O':1.95,'S':2.30,'C':1.85},
    'Ni':{'Cl':2.21,'Br':2.37,'F':1.86,'I':2.55,
          'N':2.05,'O':2.00,'S':2.28,'C':1.85},
    'Cu':{'Cl':2.26,'Br':2.42,'F':1.91,'I':2.60,
          'N':2.05,'O':1.98,'S':2.32,'C':1.90},
}

def n_electrons(metal, ligand, n_lig, charge):
    return (METAL_Z[metal] +
            n_lig * LIG_Z[ligand] - charge)

def parity_ok(metal, ligand, n_lig, charge, spin):
    n_elec = n_electrons(metal, ligand, n_lig, charge)
    return (n_elec % 2) == (spin % 2)

def get_valid_spins(metal, ligand, n_lig, charge,
                   max_spin=5):
    """Return all valid spin values with parity check."""
    valid = []
    for spin in range(0, max_spin+1):
        if parity_ok(metal, ligand, n_lig, charge, spin):
            valid.append(spin)
    return valid

def build_geometry(metal, ligand, n_lig, dist):
    positions = {
        4: [(dist,0,0),(-dist,0,0),
            (0,dist,0),(0,-dist,0)],
        6: [(dist,0,0),(-dist,0,0),(0,dist,0),
            (0,-dist,0),(0,0,dist),(0,0,-dist)]
    }
    atom_str = f"{metal}  0.000  0.000  0.000\n"
    for p in positions[n_lig]:
        atom_str += (f"{ligand}  {p[0]:.3f}  "
                     f"{p[1]:.3f}  {p[2]:.3f}\n")
    return atom_str

def get_active_electrons(n_total):
    for n_act in [10,9,11,8,12,7,13,6,14]:
        if (n_total-n_act)>=0 and (n_total-n_act)%2==0:
            return n_act
    raise ValueError(f"No valid n_active for {n_total}")

def run_hf(mol):
    for s in [
        dict(max_cycle=300,conv_tol=1e-10,
             damp=0.0,level_shift=0.0),
        dict(max_cycle=500,conv_tol=1e-9,
             damp=0.3,level_shift=0.0),
        dict(max_cycle=500,conv_tol=1e-8,
             damp=0.5,level_shift=0.2),
        dict(max_cycle=800,conv_tol=1e-7,
             damp=0.3,level_shift=0.5),
    ]:
        mf = scf.UHF(mol)
        for k,v in s.items(): setattr(mf,k,v)
        mf.verbose=0; mf.run()
        if mf.converged: return mf
    return mf

def run_casscf(mf, mol, n_act):
    homo_idx = int(
        mf.mo_occ[0].sum()+mf.mo_occ[1].sum())//2-1
    center = homo_idx
    windows = [
        list(range(max(0,center-5),center+5)),
        list(range(max(0,center-4),center+6)),
        list(range(max(0,center-6),center+4)),
        list(range(max(0,center-3),center+7)),
        list(range(max(0,center-7),center+3)),
    ]
    best_mc=None; best_ecorr=0.0
    for window in windows:
        if len(window)<10: continue
        for shift in [1e-3,1e-2,5e-2,1e-1]:
            try:
                mc=mcscf.CASSCF(mf,10,n_act)
                mc.max_cycle_macro=500
                mc.conv_tol=1e-8
                mc.ah_level_shift=shift
                mc.verbose=0
                mc.kernel(mc.sort_mo(window[:10],base=0))
                ecorr=mc.e_tot-mf.e_tot
                if ecorr<0 and ecorr<best_ecorr:
                    best_mc=mc; best_ecorr=ecorr
                if mc.converged and ecorr<-0.01:
                    return mc,True
            except: continue
    if best_mc: return best_mc,best_mc.converged
    return None,False

def run_one(metal, charge, n_lig, ligand, dist, spin):
    mult    = spin+1
    name    = (f"{metal}_{ligand}{n_lig}"
               f"_chg{charge}_spin{spin}")
    outdir  = os.path.expanduser(
        "~/activeml/data/generated300")
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"{name}.json")

    if os.path.exists(outfile):
        with open(outfile) as f:
            r = json.load(f)
        if (r.get('converged') and
                r.get('corr_energy',0) < -0.001):
            log.info(f"SKIP: {name}"); return True

    # Double-check parity
    if not parity_ok(metal, ligand, n_lig, charge, spin):
        n_el = n_electrons(metal, ligand, n_lig, charge)
        log.error(f"PARITY FAIL: {name} "
                  f"n_elec={n_el} spin={spin}")
        json.dump({'name':name,'status':'skipped',
                   'reason':'parity'},
                  open(outfile,'w'))
        return True

    log.info(f"Starting: {name}  mult={mult}")
    try:
        mol = gto.Mole()
        mol.atom    = build_geometry(
            metal, ligand, n_lig, dist)
        mol.basis   = 'def2-SVP'
        mol.charge  = charge
        mol.spin    = spin
        mol.verbose = 0
        mol.build()

        # Verify electron count matches expectation
        expected = n_electrons(
            metal, ligand, n_lig, charge)
        if abs(mol.nelectron - expected) > 2:
            log.warning(
                f"Electron count mismatch: "
                f"expected {expected}, got {mol.nelectron}")

        n_act = get_active_electrons(mol.nelectron)
        mf    = run_hf(mol)
        mc, converged = run_casscf(mf, mol, n_act)

        if mc is None:
            json.dump({'name':name,'status':'failed',
                       'ligand':ligand,'metal':metal},
                      open(outfile,'w'))
            return False

        ecorr = mc.e_tot - mf.e_tot
        if ecorr >= 0:
            json.dump({'name':name,'status':'unphysical',
                       'corr_energy':float(ecorr),
                       'ligand':ligand,'metal':metal},
                      open(outfile,'w'))
            return False

        casdm1 = mc.fcisolver.make_rdm1(
            mc.ci, mc.ncas, mc.nelecas)
        no_occ,_ = np.linalg.eigh(casdm1)
        no_occ   = np.sort(no_occ)[::-1]
        n_active = sum(1 for n in no_occ
                       if 0.02 < n < 1.98)

        with open(outfile,'w') as f:
            json.dump({
                "name"        : name,
                "metal"       : metal,
                "ligand"      : ligand,
                "n_ligands"   : n_lig,
                "charge"      : charge,
                "spin"        : spin,
                "mult"        : mult,
                "dist_ang"    : dist,
                "n_electrons" : mol.nelectron,
                "n_active_e"  : n_act,
                "E_HF"        : float(mf.e_tot),
                "E_CASSCF"    : float(mc.e_tot),
                "corr_energy" : float(ecorr),
                "converged"   : bool(converged),
                "n_active"    : n_active,
                "no_occ"      : [float(n) for n in no_occ],
                "status"      : "ok",
            }, f, indent=2)

        log.info(f"  Saved: active={n_active} "
                 f"Ecorr={ecorr:.4f}")
        return True

    except Exception as e:
        log.error(f"  Error: {e}")
        json.dump({'name':name,'status':'error',
                   'reason':str(e),'ligand':ligand,
                   'metal':metal},
                  open(outfile,'w'))
        return False

# ── BUILD SYSTEMS WITH PRE-VERIFICATION ───────────────────────
gen300   = os.path.expanduser('~/activeml/data/generated300')
existing = set(os.path.basename(f)
               for f in __import__('glob').glob(
                   f'{gen300}/*.json'))

def bls(eq):
    return [round(eq*f,3)
            for f in [0.95, 1.00, 1.05]]

ALL_JOBS = []
seen     = set()
rejected = []

for ligand in ['N','O','S','C']:
    for metal in ['Fe','Mn','Cr','Co','Ni','Cu']:
        eq = EQ_DIST[metal][ligand]
        for charge in [-1,-2,-3,-4,-5]:
            for n_lig in [4, 6]:
                spins = get_valid_spins(
                    metal, ligand, n_lig, charge)
                if not spins:
                    continue
                for frac in [0.95, 1.00, 1.05]:
                    dist = round(eq*frac, 3)
                    for spin in spins:
                        fname = (f"{metal}_{ligand}{n_lig}"
                                 f"_chg{charge}_spin{spin}"
                                 f".json")
                        if fname in existing:
                            continue
                        key = (metal,ligand,charge,
                               n_lig,dist,spin)
                        if key in seen:
                            continue
                        # Final verification
                        ok = parity_ok(
                            metal,ligand,n_lig,charge,spin)
                        if not ok:
                            rejected.append(fname)
                            continue
                        seen.add(key)
                        ALL_JOBS.append(
                            (metal,charge,n_lig,
                             ligand,dist,spin))

if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1]=='summary':
        from collections import defaultdict
        by_lig   = defaultdict(int)
        by_metal = defaultdict(int)
        by_ml    = defaultdict(int)
        for m,c,n,l,d,s in ALL_JOBS:
            by_lig[l]    += 1
            by_metal[m]  += 1
            by_ml[(m,l)] += 1
        print(f"Total valid jobs:  {len(ALL_JOBS)}")
        print(f"Rejected (parity): {len(rejected)}")
        print(f"Zero parity errors in submitted jobs ✓")
        print()
        print("By ligand:")
        for k,v in sorted(by_lig.items()):
            print(f"  {k}: {v}")
        print()
        print("By metal:")
        for k,v in sorted(by_metal.items()):
            print(f"  {k}: {v}")
        print()
        print("By metal x ligand:")
        for m in ['Fe','Mn','Cr','Co','Ni','Cu']:
            row = [f"{by_ml[(m,l)]:3d}"
                   for l in ['N','O','S','C']]
            print(f"  {m}: N={row[0]} O={row[1]} "
                  f"S={row[2]} C={row[3]}")
        sys.exit(0)

    job_idx = int(sys.argv[1]) if len(sys.argv)>1 else 0
    if job_idx >= len(ALL_JOBS): sys.exit(1)
    success = run_one(*ALL_JOBS[job_idx])
    sys.exit(0 if success else 1)
