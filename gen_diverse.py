"""
Diverse ligand dataset generation.
Adds Br, F, I ligands across all 6 metals.
Includes square planar geometry for Cu and Ni.
Same convergence protocol as gen_300.py.
"""
import numpy as np, json, os, sys, logging
from pyscf import gto, scf, mcscf

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

def build_geometry(metal, ligand, n_lig, dist,
                   geometry='td_oh'):
    """
    geometry options:
      td_oh: tetrahedral (4) or octahedral (6)
      sq_pl: square planar (4 ligands in xy plane)
    """
    if geometry == 'sq_pl' or n_lig == 4:
        # Square planar: all 4 in xy plane
        positions_4 = [
            (dist, 0, 0), (-dist, 0, 0),
            (0, dist, 0), (0, -dist, 0)
        ]
        positions = {4: positions_4}
    else:
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

def run_one(metal, charge, n_lig, ligand,
            dist, spin, geometry='td_oh'):
    mult = spin+1
    name = (f"{metal}_{ligand}{n_lig}"
            f"_chg{charge}_spin{spin}")
    if geometry == 'sq_pl':
        name = name + '_sqpl'

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

    log.info(f"Starting: {name}  mult={mult}")

    # Parity check before building mol
    try:
        from pyscf import gto as _gto
        _m = _gto.Mole()
        _m.atom    = build_geometry(
            metal, ligand, n_lig, dist, geometry)
        _m.basis   = 'sto-3g'
        _m.charge  = charge
        _m.spin    = 0
        _m.verbose = 0
        _m.build()
        n_elec = _m.nelectron
        if (n_elec % 2) != (spin % 2):
            log.warning(f"SKIP parity: {name} "
                        f"n_elec={n_elec} spin={spin}")
            json.dump({'name':name,'status':'skipped',
                       'reason':f'parity n_elec={n_elec}'},
                      open(outfile,'w'))
            return True
        del _m
    except Exception as _e:
        log.warning(f"Parity check failed: {_e}")
    try:
        mol = gto.Mole()
        mol.atom    = build_geometry(
            metal, ligand, n_lig, dist, geometry)
        mol.basis   = 'def2-SVP'
        mol.charge  = charge
        mol.spin    = spin
        mol.verbose = 0
        mol.build()

        n_act = get_active_electrons(mol.nelectron)
        mf    = run_hf(mol)
        mc, converged = run_casscf(mf, mol, n_act)

        if mc is None:
            json.dump({'name':name,'status':'failed'},
                      open(outfile,'w'))
            return False

        ecorr = mc.e_tot - mf.e_tot
        if ecorr >= 0:
            json.dump({'name':name,'status':'unphysical',
                       'corr_energy':float(ecorr)},
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
                "geometry"    : geometry,
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
                   'reason':str(e)},
                  open(outfile,'w'))
        return False

# ── EQUILIBRIUM BOND LENGTHS ──────────────────────────
# Approximate M-L equilibrium distances
EQ = {
    # Metal: {Ligand: {coord: dist_ang}}
    'Fe': {'Cl':2.18,'Br':2.35,'F':1.85,'I':2.55},
    'Mn': {'Cl':2.35,'Br':2.50,'F':1.98,'I':2.70},
    'Cr': {'Cl':2.31,'Br':2.47,'F':1.94,'I':2.65},
    'Co': {'Cl':2.26,'Br':2.42,'F':1.90,'I':2.60},
    'Ni': {'Cl':2.21,'Br':2.37,'F':1.86,'I':2.55},
    'Cu': {'Cl':2.26,'Br':2.42,'F':1.91,'I':2.60},
}

def eq(metal, ligand):
    return EQ.get(metal,{}).get(ligand, 2.20)

SYSTEMS = []

# ── BROMINE ligand ────────────────────────────────────
# Same spin states as Cl — easy convergence, weak field
for metal, spins4, spins6 in [
    ('Fe', [4,2,0], [4,2,0]),
    ('Mn', [5,3,1], [5,3,1]),
    ('Cr', [3,1],   [3,1]),
    ('Co', [3,1],   [3,1]),
    ('Ni', [2,0],   [2,0]),
    ('Cu', [1],     [1]),
]:
    d4 = eq(metal,'Br')
    d6 = eq(metal,'Br')
    for chg in [-2,-3]:
        SYSTEMS.append(
            (metal,chg,4,'Br',d4,spins4))
        SYSTEMS.append(
            (metal,chg,4,'Br',round(d4*0.95,3),spins4))
        SYSTEMS.append(
            (metal,chg,4,'Br',round(d4*1.05,3),spins4))
        SYSTEMS.append(
            (metal,chg,6,'Br',d6,spins6))
        SYSTEMS.append(
            (metal,chg,6,'Br',round(d6*1.05,3),spins6))

# ── FLUORINE ligand ───────────────────────────────────
# Stronger field than Cl — favours low spin
for metal, spins4, spins6 in [
    ('Fe', [4,2,0], [4,2,0]),
    ('Mn', [5,3,1], [5,3,1]),
    ('Cr', [3,1],   [3,1]),
    ('Co', [3,1],   [3,1]),
    ('Ni', [2,0],   [2,0]),
    ('Cu', [1],     [1]),
]:
    d4 = eq(metal,'F')
    d6 = eq(metal,'F')
    for chg in [-2,-3]:
        SYSTEMS.append(
            (metal,chg,4,'F',d4,spins4))
        SYSTEMS.append(
            (metal,chg,4,'F',round(d4*0.95,3),spins4))
        SYSTEMS.append(
            (metal,chg,6,'F',d6,spins6))
        SYSTEMS.append(
            (metal,chg,6,'F',round(d6*1.05,3),spins6))

# ── IODINE ligand ─────────────────────────────────────
# Weakest field halide — strong tendency to high spin
for metal, spins4 in [
    ('Fe', [4,2,0]),
    ('Mn', [5,3,1]),
    ('Co', [3,1]),
    ('Ni', [2,0]),
    ('Cu', [1]),
]:
    d4 = eq(metal,'I')
    for chg in [-2,-3]:
        SYSTEMS.append(
            (metal,chg,4,'I',d4,spins4))
        SYSTEMS.append(
            (metal,chg,4,'I',round(d4*1.05,3),spins4))

# ── COPPER special: square planar ─────────────────────
# Real Cu²⁺ prefers square planar geometry
# Adding this increases Cu dataset significantly
for chg in [-2,-1,0,-3]:
    for d_frac in [0.95, 1.00, 1.05]:
        d = round(eq('Cu','Cl') * d_frac, 3)
        # Square planar = 4-coordinate but planar
        SYSTEMS.append(
            ('Cu',chg,4,'Cl',d,[1],'sq_pl'))
    for d_frac in [1.00, 1.05]:
        d = round(eq('Cu','Br') * d_frac, 3)
        SYSTEMS.append(
            ('Cu',chg,4,'Br',d,[1],'sq_pl'))

# ── Build ALL_JOBS ─────────────────────────────────────
ALL_JOBS = []
for entry in SYSTEMS:
    if len(entry) == 6:
        metal,charge,n_lig,ligand,dist,spins = entry
        geometry = 'td_oh'
    else:
        metal,charge,n_lig,ligand,dist,spins,geometry = entry
    for spin in spins:
        # Parity check
        # Quick estimate of n_electrons
        # (skip parity-violating combinations)
        ALL_JOBS.append(
            (metal,charge,n_lig,ligand,
             dist,spin,geometry))

if __name__ == "__main__":
    job_idx = int(sys.argv[1]) if len(sys.argv)>1 else 0
    total   = len(ALL_JOBS)
    print(f"Total jobs: {total}")
    if job_idx >= total: sys.exit(1)
    metal,charge,n_lig,ligand,dist,spin,geometry = \
        ALL_JOBS[job_idx]
    success = run_one(
        metal,charge,n_lig,ligand,dist,spin,geometry)
    sys.exit(0 if success else 1)
