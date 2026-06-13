"""
gen_row5d_extra.py
==================
Hard-coded CSD-survey structures for Re, Os, W.
These metals are entirely absent from the training data.

Re: radiolabeling, CO2 reduction photocatalysis (d1-d5)
Os: Sharpless dihydroxylation, anticancer (d3-d5)
W:  nitrogen fixation analogue to Mo (d0-d4)

Same pipeline as gen_csd_4d5d.py — CASSCF(10,10), def2-SVP + ECP.
Output: ~/activeml/data/generated_row5d_extra/

Bond lengths from:
  CSD mean values for Re/Os/W halide complexes
  Alvarez S., Dalton Trans. 2013

Usage:
  python gen_row5d_extra.py summary
  python gen_row5d_extra.py <idx>
  sbatch --array=0-N job_row5d_extra.sh
"""
import numpy as np, json, os, sys, logging
from pyscf import gto, scf, mcscf

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

OUTPUT_DIR = os.path.expanduser('~/activeml/data/generated_row5d_extra')

METAL_CONSTANTS = {
    # Existing 5d (for reference consistency)
    'Ir': {'z_eff': 17.00, 'zeta_so_cm1': 3909, 'metal_row': '5d'},
    'Pt': {'z_eff': 17.33, 'zeta_so_cm1': 4146, 'metal_row': '5d'},
    # New 5d metals
    'Re': {'z_eff': 17.01, 'zeta_so_cm1': 2456, 'metal_row': '5d'},
    'Os': {'z_eff': 17.17, 'zeta_so_cm1': 3381, 'metal_row': '5d'},
    'W':  {'z_eff': 11.61, 'zeta_so_cm1': 2748, 'metal_row': '5d'},
}

ECP_METALS = {'Re', 'Os', 'W', 'Ir', 'Pt'}

METAL_Z = {'Re': 75, 'Os': 76, 'W': 74}
LIG_Z   = {'Cl': 17, 'Br': 35, 'F': 9, 'N': 7, 'O': 8}

# ══════════════════════════════════════════════════════════════
# CSD STRUCTURES
# Bond lengths from CSD survey means (Alvarez 2013)
# Re-Cl oct: 2.340-2.400 Å depending on oxidation state
# Os-Cl oct: 2.330-2.380 Å
# W-Cl oct:  2.420-2.480 Å (similar to Mo)
# ══════════════════════════════════════════════════════════════

CSD_STRUCTURES = [

    # ══════════════════════════════════════════════════════════
    # RHENIUM (Re) — d1 to d5, octahedral dominant
    # Re is 5d, strong SOC (ζ=2456 cm⁻¹)
    # Most common: Re(III) d4, Re(IV) d3, Re(V) d2
    # ══════════════════════════════════════════════════════════

    # Re(III) d4 — octahedral
    ("ReCl6_2m_oct",
     "Re", -2,
     """Re  0.000  0.000  0.000
        Cl  2.360  0.000  0.000
        Cl -2.360  0.000  0.000
        Cl  0.000  2.360  0.000
        Cl  0.000 -2.360  0.000
        Cl  0.000  0.000  2.360
        Cl  0.000  0.000 -2.360"""),

    # Re(II) d5 — octahedral (paramagnetic)
    ("ReCl6_3m_oct",
     "Re", -3,
     """Re  0.000  0.000  0.000
        Cl  2.400  0.000  0.000
        Cl -2.400  0.000  0.000
        Cl  0.000  2.400  0.000
        Cl  0.000 -2.400  0.000
        Cl  0.000  0.000  2.400
        Cl  0.000  0.000 -2.400"""),

    # Re(IV) d3 — octahedral
    ("ReCl6_1m_oct",
     "Re", -1,
     """Re  0.000  0.000  0.000
        Cl  2.330  0.000  0.000
        Cl -2.330  0.000  0.000
        Cl  0.000  2.330  0.000
        Cl  0.000 -2.330  0.000
        Cl  0.000  0.000  2.330
        Cl  0.000  0.000 -2.330"""),

    # Re(V) d2 — oxo complex (trans-dioxo common)
    ("ReCl4O2_trans",
     "Re", -1,
     """Re  0.000  0.000  0.000
        Cl  2.340  0.000  0.000
        Cl -2.340  0.000  0.000
        Cl  0.000  2.340  0.000
        Cl  0.000 -2.340  0.000
        O   0.000  0.000  1.760
        O   0.000  0.000 -1.760"""),

    # Re(III) distorted octahedral (real crystal packing)
    ("ReCl6_2m_dist",
     "Re", -2,
     """Re  0.000  0.000  0.000
        Cl  2.350  0.110 -0.065
        Cl -2.370  0.075  0.085
        Cl  0.085  2.360 -0.055
        Cl -0.065 -2.365  0.075
        Cl  0.045  0.035  2.375
        Cl -0.055 -0.045 -2.370"""),

    # Re bromide octahedral
    ("ReBr6_2m_oct",
     "Re", -2,
     """Re  0.000  0.000  0.000
        Br  2.490  0.000  0.000
        Br -2.490  0.000  0.000
        Br  0.000  2.490  0.000
        Br  0.000 -2.490  0.000
        Br  0.000  0.000  2.490
        Br  0.000  0.000 -2.490"""),

    # Re with N-donor (anticancer/imaging relevance)
    ("ReCl4N2_trans",
     "Re", -1,
     """Re  0.000  0.000  0.000
        Cl  2.360  0.000  0.000
        Cl -2.360  0.000  0.000
        Cl  0.000  2.360  0.000
        Cl  0.000 -2.360  0.000
        N   0.000  0.000  2.100
        N   0.000  0.000 -2.100"""),

    # ══════════════════════════════════════════════════════════
    # OSMIUM (Os) — d3 to d5, octahedral dominant
    # Os is 5d, very strong SOC (ζ=3381 cm⁻¹)
    # Used in Sharpless dihydroxylation (OsO4)
    # ══════════════════════════════════════════════════════════

    # Os(IV) d4 — octahedral (most stable)
    ("OsCl6_2m_oct",
     "Os", -2,
     """Os  0.000  0.000  0.000
        Cl  2.330  0.000  0.000
        Cl -2.330  0.000  0.000
        Cl  0.000  2.330  0.000
        Cl  0.000 -2.330  0.000
        Cl  0.000  0.000  2.330
        Cl  0.000  0.000 -2.330"""),

    # Os(III) d5 — octahedral (paramagnetic)
    ("OsCl6_3m_oct",
     "Os", -3,
     """Os  0.000  0.000  0.000
        Cl  2.360  0.000  0.000
        Cl -2.360  0.000  0.000
        Cl  0.000  2.360  0.000
        Cl  0.000 -2.360  0.000
        Cl  0.000  0.000  2.360
        Cl  0.000  0.000 -2.360"""),

    # OsO4 — tetrahedral Os(VIII) d0 (Sharpless catalyst)
    ("OsO4_tet",
     "Os", 0,
     """Os  0.000  0.000  0.000
        O   1.710  0.000  0.000
        O  -1.710  0.000  0.000
        O   0.000  1.710  0.000
        O   0.000 -1.710  0.000"""),

    # Os bromide octahedral
    ("OsBr6_2m_oct",
     "Os", -2,
     """Os  0.000  0.000  0.000
        Br  2.460  0.000  0.000
        Br -2.460  0.000  0.000
        Br  0.000  2.460  0.000
        Br  0.000 -2.460  0.000
        Br  0.000  0.000  2.460
        Br  0.000  0.000 -2.460"""),

    # Os(IV) distorted
    ("OsCl6_2m_dist",
     "Os", -2,
     """Os  0.000  0.000  0.000
        Cl  2.320  0.105 -0.060
        Cl -2.340  0.070  0.080
        Cl  0.080  2.330 -0.050
        Cl -0.060 -2.335  0.070
        Cl  0.040  0.030  2.345
        Cl -0.050 -0.040 -2.340"""),

    # ══════════════════════════════════════════════════════════
    # TUNGSTEN (W) — d0 to d4, octahedral dominant
    # W is 5d, strong SOC (ζ=2748 cm⁻¹)
    # Close analogue to Mo — nitrogen fixation relevant
    # ══════════════════════════════════════════════════════════

    # W(IV) d2 — octahedral (most common)
    ("WCl6_2m_oct",
     "W", -2,
     """W   0.000  0.000  0.000
        Cl  2.440  0.000  0.000
        Cl -2.440  0.000  0.000
        Cl  0.000  2.440  0.000
        Cl  0.000 -2.440  0.000
        Cl  0.000  0.000  2.440
        Cl  0.000  0.000 -2.440"""),

    # W(III) d3 — octahedral
    ("WCl6_3m_oct",
     "W", -3,
     """W   0.000  0.000  0.000
        Cl  2.470  0.000  0.000
        Cl -2.470  0.000  0.000
        Cl  0.000  2.470  0.000
        Cl  0.000 -2.470  0.000
        Cl  0.000  0.000  2.470
        Cl  0.000  0.000 -2.470"""),

    # W(VI) d0 — tungstate WO4^2- tetrahedral
    ("WO4_2m_tet",
     "W", -2,
     """W   0.000  0.000  0.000
        O   1.780  0.000  0.000
        O  -1.780  0.000  0.000
        O   0.000  1.780  0.000
        O   0.000 -1.780  0.000"""),

    # W(VI) mixed oxo-chloride (common in catalysis)
    ("WCl4O2_trans",
     "W", -2,
     """W   0.000  0.000  0.000
        Cl  2.420  0.000  0.000
        Cl -2.420  0.000  0.000
        Cl  0.000  2.420  0.000
        Cl  0.000 -2.420  0.000
        O   0.000  0.000  1.680
        O   0.000  0.000 -1.680"""),

    # W(IV) distorted octahedral
    ("WCl6_2m_dist",
     "W", -2,
     """W   0.000  0.000  0.000
        Cl  2.430  0.115 -0.070
        Cl -2.450  0.080  0.095
        Cl  0.095  2.440 -0.065
        Cl -0.070 -2.445  0.085
        Cl  0.055  0.040  2.455
        Cl -0.065 -0.055 -2.460"""),

    # W bromide
    ("WBr6_2m_oct",
     "W", -2,
     """W   0.000  0.000  0.000
        Br  2.560  0.000  0.000
        Br -2.560  0.000  0.000
        Br  0.000  2.560  0.000
        Br  0.000 -2.560  0.000
        Br  0.000  0.000  2.560
        Br  0.000  0.000 -2.560"""),
]

# ── SPIN STATES ───────────────────────────────────────────────
SPIN_STATES = {
    # Re: d1(Re VI)=1, d2(Re V)=2, d3(Re IV)=3, d4(Re III)=4, d5(Re II)=5
    'Re': [1, 3, 5],
    # Os: d3(Os V)=3, d4(Os IV)=4, d5(Os III)=5; low-spin preferred
    'Os': [0, 2, 4],
    # W: d0(W VI)=0, d1(W V)=1, d2(W IV)=2, d3(W III)=3; low-spin preferred
    'W':  [0, 1, 2, 3],
}


def count_electrons(atoms_str, charge):
    total = 0
    for line in atoms_str.strip().split('\n'):
        sym = line.strip().split()[0]
        total += METAL_Z.get(sym, LIG_Z.get(sym, 0))
    return total - charge


def get_nact(n_total):
    for n in [10, 9, 11, 8, 12, 7, 13, 6, 14]:
        if (n_total - n) >= 0 and (n_total - n) % 2 == 0:
            return n
    return 10


def run_uhf(mol):
    for settings in [
        dict(max_cycle=300, conv_tol=1e-10, damp=0.0, level_shift=0.0),
        dict(max_cycle=500, conv_tol=1e-9,  damp=0.3, level_shift=0.2),
        dict(max_cycle=800, conv_tol=1e-8,  damp=0.5, level_shift=0.5),
    ]:
        mf = scf.UHF(mol)
        for k, v in settings.items(): setattr(mf, k, v)
        mf.verbose = 0
        mf.run()
        if mf.converged: return mf
    return mf


def run_casscf(mf, mol, n_act):
    e_m = (mf.mo_energy[0] + mf.mo_energy[1]) / 2
    occ = mf.mo_occ[0] + mf.mo_occ[1]
    occ_idx  = np.where(occ > 0.5)[0]
    virt_idx = np.where(occ < 0.5)[0]
    if len(occ_idx) == 0 or len(virt_idx) == 0: return None, False
    homo_e = float(e_m[occ_idx[-1]])
    lumo_e = float(e_m[virt_idx[0]])
    gap    = (homo_e + lumo_e) / 2
    w14    = sorted(np.argsort(np.abs(e_m - gap))[:14],
                    key=lambda i: e_m[i])
    best_mc = None; best_e = 0.0
    for window in [w14[:10], w14[2:12], w14[1:11], w14[3:13]]:
        for shift in [1e-3, 1e-2, 5e-2, 1e-1]:
            try:
                mc = mcscf.CASSCF(mf, 10, n_act)
                mc.max_cycle_macro = 500
                mc.conv_tol = 1e-8
                mc.ah_level_shift = shift
                mc.verbose = 0
                mc.kernel(mc.sort_mo(window, base=0))
                ec = mc.e_tot - mf.e_tot
                if ec < 0 and ec < best_e:
                    best_mc = mc; best_e = ec
                if mc.converged and ec < -0.01:
                    return mc, True
            except: continue
    if best_mc: return best_mc, best_mc.converged
    return None, False


def run_one(struct_name, metal, charge, atoms_str, spin):
    name    = f"CSD_{struct_name}_spin{spin}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outfile = os.path.join(OUTPUT_DIR, f"{name}.json")

    if os.path.exists(outfile):
        r = json.load(open(outfile))
        if r.get('converged') and r.get('corr_energy', 0) < -0.001:
            log.info(f"SKIP: {name}"); return True

    n_e = count_electrons(atoms_str, charge)
    if (n_e % 2) != (spin % 2):
        json.dump({'name': name, 'status': 'skipped',
                   'reason': 'parity', 'geometry': 'csd_real'},
                  open(outfile, 'w')); return True

    consts = METAL_CONSTANTS[metal]
    log.info(f"Start: {name}  n_e={n_e}  row={consts['metal_row']}")

    try:
        mol = gto.Mole()
        mol.atom    = atoms_str
        mol.basis   = 'def2-SVP'
        mol.ecp     = 'def2-SVP'
        mol.charge  = charge
        mol.spin    = spin
        mol.verbose = 0
        mol.build()

        n_act = get_nact(mol.nelectron)
        mf    = run_uhf(mol)
        mc, conv = run_casscf(mf, mol, n_act)

        if mc is None:
            json.dump({'name': name, 'status': 'failed',
                       'geometry': 'csd_real', 'metal': metal},
                      open(outfile, 'w')); return False

        ec = mc.e_tot - mf.e_tot
        if ec >= 0:
            json.dump({'name': name, 'status': 'unphysical',
                       'corr_energy': float(ec),
                       'geometry': 'csd_real'},
                      open(outfile, 'w')); return False

        casdm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
        no, _  = np.linalg.eigh(casdm1)
        no     = np.sort(no)[::-1]
        n_active = sum(1 for n in no if 0.02 < n < 1.98)

        lig = 'Cl'
        for l in ['Br', 'F', 'N', 'O']:
            if l in atoms_str: lig = l; break

        e_m = (mf.mo_energy[0] + mf.mo_energy[1]) / 2
        occ = mf.mo_occ[0] + mf.mo_occ[1]
        oi  = np.where(occ > 0.5)[0]
        vi  = np.where(occ < 0.5)[0]
        homo_e = float(e_m[oi[-1]])  if len(oi) > 0 else 0.0
        lumo_e = float(e_m[vi[0]])   if len(vi) > 0 else 0.0
        oa = np.where(mf.mo_occ[0] > 0.5)[0]
        ob = np.where(mf.mo_occ[1] > 0.5)[0]
        homo_a = float(mf.mo_energy[0][oa[-1]]) if len(oa) > 0 else 0.0
        homo_b = float(mf.mo_energy[1][ob[-1]]) if len(ob) > 0 else 0.0
        S = spin / 2.0
        spin_contam = float(mf.spin_square()[0] - S*(S+1))

        result = {
            'name':              name,
            'metal':             metal,
            'ligand':            lig,
            'n_ligands':         atoms_str.count(lig),
            'charge':            charge,
            'spin':              spin,
            'mult':              spin + 1,
            'geometry':          'csd_real',
            'struct_name':       struct_name,
            'n_electrons':       mol.nelectron,
            'n_active_e':        n_act,
            'E_HF':              float(mf.e_tot),
            'E_CASSCF':          float(mc.e_tot),
            'corr_energy':       float(ec),
            'converged':         bool(conv),
            'n_active':          n_active,
            'no_occ':            [float(x) for x in no],
            'status':            'ok',
            'z_eff':             consts['z_eff'],
            'zeta_so_cm1':       consts['zeta_so_cm1'],
            'metal_row':         consts['metal_row'],
            'spin_contamination': spin_contam,
            'homo_lumo_gap':     float(lumo_e - homo_e),
            'homo_energy':       homo_e,
            'lumo_energy':       lumo_e,
            'homo_ab_gap':       float(abs(homo_a - homo_b)),
        }

        with open(outfile, 'w') as f:
            json.dump(result, f, indent=2)
        log.info(f"  OK: n_active={n_active} Ec={ec:.4f} converged={conv}")
        return True

    except Exception as e:
        log.error(f"  Error {name}: {e}")
        json.dump({'name': name, 'status': 'error', 'reason': str(e),
                   'geometry': 'csd_real'}, open(outfile, 'w'))
        return False


# ── BUILD JOB LIST ─────────────────────────────────────────────
ALL_JOBS = []
seen     = set()
os.makedirs(OUTPUT_DIR, exist_ok=True)
existing = set(os.path.basename(f)
               for f in __import__('glob').glob(f'{OUTPUT_DIR}/*.json'))

for struct_name, metal, charge, atoms_str in CSD_STRUCTURES:
    spins = SPIN_STATES.get(metal, [0, 2])
    n_e   = count_electrons(atoms_str, charge)
    for spin in spins:
        if (n_e % 2) != (spin % 2): continue
        fname = f"CSD_{struct_name}_spin{spin}.json"
        if fname in existing: continue
        key = (struct_name, spin)
        if key in seen: continue
        seen.add(key)
        ALL_JOBS.append((struct_name, metal, charge, atoms_str, spin))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        from collections import defaultdict
        by_metal = defaultdict(int)
        for s, m, c, a, sp in ALL_JOBS:
            by_metal[m] += 1
        print(f"\nTotal jobs:  {len(ALL_JOBS)}")
        print(f"Structures:  {len(CSD_STRUCTURES)}")
        print(f"\nJobs by metal:")
        for m in ['Re', 'Os', 'W']:
            print(f"  {m}: {by_metal.get(m, 0)}")
        print(f"\nEst. time: ~{len(ALL_JOBS)*25/60:.0f} hrs serial, "
              f"~{len(ALL_JOBS)*25/60/32:.1f} hrs on 32 cores")
        sys.exit(0)

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if idx >= len(ALL_JOBS): sys.exit(1)
    sys.exit(0 if run_one(*ALL_JOBS[idx]) else 1)
