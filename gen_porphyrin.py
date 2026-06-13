"""
gen_porphyrin.py
================
Simplified porphyrin/heme model structures.
Fe-porphyrin is THE canonical multireference benchmark.
Every reviewer of a TM active space paper tests on heme.

Simplified model: represent porphyrin ring as 4 N donors
at correct M-N bond distance in square planar arrangement.
Axial ligands added for 5- and 6-coordinate variants.
This captures the essential d-orbital splitting without
the full organic porphyrin scaffold.

Why this is physically valid:
  The active space in Fe-porphyrin is determined by:
  1. Fe d-orbitals (5 orbitals)
  2. Fe-N sigma bonding combinations (4 orbitals)
  3. Fe-N pi backbonding (2-4 orbitals)
  The porphyrin carbon framework does NOT enter the active space
  for ground-state electronic structure — only the N donor
  positions matter for CASSCF orbital ordering.

Uses CASSCF(14,14) — porphyrin systems have n_active > 10
due to mixing of Fe d with porphyrin a1u/a2u pi orbitals.

Metals: Fe (heme), Mn (Mn-porphyrin), Co (cobalt porphyrin),
        Ni, Cu, Zn (metalloporphyrins), Ru, Ir (photocatalysis)

Output: ~/activeml/data/generated_porphyrin/

Bond lengths (CSD mean values):
  Fe-N porphyrin: 2.010 Ang (high spin), 1.990 Ang (low spin)
  Mn-N porphyrin: 2.010 Ang
  Co-N porphyrin: 1.980 Ang
  Ni-N porphyrin: 1.960 Ang
  Cu-N porphyrin: 2.000 Ang
  Zn-N porphyrin: 2.040 Ang
  Ru-N porphyrin: 2.050 Ang
  Ir-N porphyrin: 2.020 Ang

Axial ligands (for 5- and 6-coordinate):
  M-Cl axial: 2.250-2.400 Ang
  M-O  axial: 1.600-1.800 Ang (oxo, very short)
  M-N  axial: 2.100-2.200 Ang (histidine model)

Usage:
  python gen_porphyrin.py summary
  python gen_porphyrin.py <idx>
  sbatch --array=0-N job_porphyrin.sh
"""
import numpy as np, json, os, sys, logging, math
from pyscf import gto, scf, mcscf

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

OUTPUT_DIR = os.path.expanduser('~/activeml/data/generated_porphyrin')

METAL_CONSTANTS = {
    'Fe': {'z_eff': 11.18, 'zeta_so_cm1':  460, 'metal_row': '3d'},
    'Mn': {'z_eff': 10.53, 'zeta_so_cm1':  355, 'metal_row': '3d'},
    'Co': {'z_eff': 12.00, 'zeta_so_cm1':  533, 'metal_row': '3d'},
    'Ni': {'z_eff': 12.78, 'zeta_so_cm1':  669, 'metal_row': '3d'},
    'Cu': {'z_eff': 13.20, 'zeta_so_cm1':  831, 'metal_row': '3d'},
    'Zn': {'z_eff': 13.57, 'zeta_so_cm1': 1042, 'metal_row': '3d'},
    'Ru': {'z_eff': 12.33, 'zeta_so_cm1':  880, 'metal_row': '4d'},
    'Ir': {'z_eff': 17.00, 'zeta_so_cm1': 3909, 'metal_row': '5d'},
}

ECP_METALS = {'Ru', 'Ir'}

METAL_Z = {
    'Fe':26,'Mn':25,'Co':27,'Ni':28,'Cu':29,'Zn':30,
    'Ru':44,'Ir':77,
}
LIG_Z = {'N':7,'Cl':17,'O':8,'Br':35}


def make_porphyrin_4coord(metal, mn_dist):
    """4-coordinate square planar — base-free metalloporphyrin."""
    return f"""{metal}  0.000  0.000  0.000
N  {mn_dist:.3f}  0.000  0.000
N -{mn_dist:.3f}  0.000  0.000
N  0.000  {mn_dist:.3f}  0.000
N  0.000 -{mn_dist:.3f}  0.000"""


def make_porphyrin_5coord(metal, mn_dist, axial_sym, axial_dist):
    """5-coordinate square pyramidal — one axial ligand."""
    return f"""{metal}  0.000  0.000  0.000
N  {mn_dist:.3f}  0.000  0.000
N -{mn_dist:.3f}  0.000  0.000
N  0.000  {mn_dist:.3f}  0.000
N  0.000 -{mn_dist:.3f}  0.000
{axial_sym}  0.000  0.000  {axial_dist:.3f}"""


def make_porphyrin_6coord(metal, mn_dist, ax1_sym, ax1_dist,
                          ax2_sym, ax2_dist):
    """6-coordinate octahedral — two axial ligands."""
    return f"""{metal}  0.000  0.000  0.000
N  {mn_dist:.3f}  0.000  0.000
N -{mn_dist:.3f}  0.000  0.000
N  0.000  {mn_dist:.3f}  0.000
N  0.000 -{mn_dist:.3f}  0.000
{ax1_sym}  0.000  0.000  {ax1_dist:.3f}
{ax2_sym}  0.000  0.000 -{ax2_dist:.3f}"""


# Build structure list
def build_structures():
    structs = []

    # ── Fe porphyrin — most important system ─────────────────
    # Fe(II) d6 4-coord (free base heme model)
    structs.append(("Fe_porph_4coord", "Fe", 0,
        make_porphyrin_4coord("Fe", 2.010), "N4"))

    # Fe(III) d5 4-coord (ferric heme)
    structs.append(("Fe_porph_4coord_3plus", "Fe", 1,
        make_porphyrin_4coord("Fe", 2.010), "N4"))

    # Fe(II) d6 5-coord with Cl (deoxy-heme model)
    structs.append(("Fe_porph_5coord_Cl", "Fe", 0,
        make_porphyrin_5coord("Fe", 2.010, "Cl", 2.280), "N4Cl"))

    # Fe(III) d5 5-coord with Cl (met-heme model)
    structs.append(("Fe_porph_5coord_Cl_3plus", "Fe", 1,
        make_porphyrin_5coord("Fe", 2.010, "Cl", 2.228), "N4Cl"))

    # Fe(IV)=O Compound I model (P450 active species, key benchmark)
    structs.append(("Fe_porph_5coord_oxo", "Fe", 1,
        make_porphyrin_5coord("Fe", 2.010, "O", 1.650), "N4O"))

    # Fe(II) d6 6-coord with two Cl (ferric-bis-halide)
    structs.append(("Fe_porph_6coord_2Cl", "Fe", 0,
        make_porphyrin_6coord("Fe", 1.990, "Cl", 2.290, "Cl", 2.290), "N4Cl2"))

    # Fe(II) 6-coord with Cl and N (histidine model — myoglobin)
    structs.append(("Fe_porph_6coord_ClN", "Fe", 0,
        make_porphyrin_6coord("Fe", 1.990, "Cl", 2.290, "N", 2.120), "N5Cl"))

    # Fe(II) 6-coord bis-N (hemoglobin CO model)
    structs.append(("Fe_porph_6coord_2N", "Fe", 0,
        make_porphyrin_6coord("Fe", 1.990, "N", 2.120, "N", 2.120), "N6"))

    # ── Mn porphyrin ─────────────────────────────────────────
    # Mn(II) d5 4-coord
    structs.append(("Mn_porph_4coord", "Mn", 0,
        make_porphyrin_4coord("Mn", 2.010), "N4"))

    # Mn(III) d4 5-coord with Cl (water oxidation catalyst model)
    structs.append(("Mn_porph_5coord_Cl", "Mn", 1,
        make_porphyrin_5coord("Mn", 2.010, "Cl", 2.350), "N4Cl"))

    # Mn(IV)=O (oxidized catalyst)
    structs.append(("Mn_porph_5coord_oxo", "Mn", 1,
        make_porphyrin_5coord("Mn", 2.010, "O", 1.700), "N4O"))

    # Mn(III) 6-coord bis-Cl
    structs.append(("Mn_porph_6coord_2Cl", "Mn", 1,
        make_porphyrin_6coord("Mn", 2.010, "Cl", 2.350, "Cl", 2.350), "N4Cl2"))

    # ── Co porphyrin ─────────────────────────────────────────
    # Co(II) d7 4-coord (vitamin B12 model)
    structs.append(("Co_porph_4coord", "Co", 0,
        make_porphyrin_4coord("Co", 1.980), "N4"))

    # Co(III) d6 5-coord with Cl
    structs.append(("Co_porph_5coord_Cl", "Co", 1,
        make_porphyrin_5coord("Co", 1.980, "Cl", 2.270), "N4Cl"))

    # Co(II) 6-coord bis-N (B12 model)
    structs.append(("Co_porph_6coord_2N", "Co", 0,
        make_porphyrin_6coord("Co", 1.980, "N", 2.100, "N", 2.100), "N6"))

    # ── Ni porphyrin ─────────────────────────────────────────
    # Ni(II) d8 4-coord (square planar, closed shell)
    structs.append(("Ni_porph_4coord", "Ni", 0,
        make_porphyrin_4coord("Ni", 1.960), "N4"))

    # Ni(II) 6-coord bis-Cl
    structs.append(("Ni_porph_6coord_2Cl", "Ni", 0,
        make_porphyrin_6coord("Ni", 1.960, "Cl", 2.420, "Cl", 2.420), "N4Cl2"))

    # ── Cu porphyrin ─────────────────────────────────────────
    # Cu(II) d9 4-coord (EPR benchmark)
    structs.append(("Cu_porph_4coord", "Cu", 0,
        make_porphyrin_4coord("Cu", 2.000), "N4"))

    # ── Zn porphyrin ─────────────────────────────────────────
    # Zn(II) d10 4-coord (fluorescent dye model)
    structs.append(("Zn_porph_4coord", "Zn", 0,
        make_porphyrin_4coord("Zn", 2.040), "N4"))

    # Zn(II) 5-coord with Cl
    structs.append(("Zn_porph_5coord_Cl", "Zn", 0,
        make_porphyrin_5coord("Zn", 2.040, "Cl", 2.300), "N4Cl"))

    # ── Ru porphyrin (photocatalysis) ─────────────────────────
    # Ru(II) d6 6-coord bis-Cl
    structs.append(("Ru_porph_6coord_2Cl", "Ru", 0,
        make_porphyrin_6coord("Ru", 2.050, "Cl", 2.380, "Cl", 2.380), "N4Cl2"))

    # Ru(II) 5-coord with Cl
    structs.append(("Ru_porph_5coord_Cl", "Ru", 0,
        make_porphyrin_5coord("Ru", 2.050, "Cl", 2.380), "N4Cl"))

    # Ru(IV)=O (oxidized photocatalyst)
    structs.append(("Ru_porph_5coord_oxo", "Ru", 0,
        make_porphyrin_5coord("Ru", 2.050, "O", 1.720), "N4O"))

    # ── Ir porphyrin (C-H activation) ─────────────────────────
    # Ir(III) d6 5-coord with Cl
    structs.append(("Ir_porph_5coord_Cl", "Ir", 0,
        make_porphyrin_5coord("Ir", 2.020, "Cl", 2.360), "N4Cl"))

    # Ir(III) d6 6-coord bis-Cl
    structs.append(("Ir_porph_6coord_2Cl", "Ir", 0,
        make_porphyrin_6coord("Ir", 2.020, "Cl", 2.360, "Cl", 2.360), "N4Cl2"))

    return structs


CSD_STRUCTURES = build_structures()

# ── SPIN STATES ───────────────────────────────────────────────
SPIN_STATES = {
    'Fe': [0, 2, 4],   # d6: S=0,1,2; d5: S=1/2,3/2,5/2
    'Mn': [1, 3, 5],   # d4/d5: S=0...5/2
    'Co': [1, 3],      # d6/d7: S=0,1/2,3/2
    'Ni': [0, 2],      # d8: S=0,1
    'Cu': [1],         # d9: S=1/2
    'Zn': [0],         # d10: S=0
    'Ru': [0, 2],      # d6: S=0,1
    'Ir': [0, 2],      # d6: S=0,1
}


def count_electrons(atoms_str, charge):
    total = 0
    for line in atoms_str.strip().split('\n'):
        parts = line.strip().split()
        if parts:
            sym = parts[0]
            total += METAL_Z.get(sym, LIG_Z.get(sym, 0))
    return total - charge


def get_nact_14(n_total):
    """Active electrons for CASSCF(14,14)."""
    for n in [14, 13, 15, 12, 16, 11, 10, 9]:
        if (n_total - n) >= 0 and (n_total - n) % 2 == 0:
            return n
    return 14


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


def run_casscf_14(mf, mol, n_act):
    """CASSCF(14,14) with 18-orbital window."""
    e_m = (mf.mo_energy[0] + mf.mo_energy[1]) / 2
    occ = mf.mo_occ[0] + mf.mo_occ[1]
    occ_idx  = np.where(occ > 0.5)[0]
    virt_idx = np.where(occ < 0.5)[0]
    if len(occ_idx) == 0 or len(virt_idx) == 0: return None, False

    homo_e = float(e_m[occ_idx[-1]])
    lumo_e = float(e_m[virt_idx[0]])
    gap    = (homo_e + lumo_e) / 2

    # 18-orbital window for (14,14): 9 occ + 9 virt
    w18 = sorted(np.argsort(np.abs(e_m - gap))[:18],
                 key=lambda i: e_m[i])

    best_mc = None; best_e = 0.0
    for window in [w18[:14], w18[2:16], w18[1:15], w18[4:18]]:
        for shift in [1e-3, 1e-2, 5e-2, 1e-1, 2e-1]:
            try:
                mc = mcscf.CASSCF(mf, 14, n_act)
                mc.max_cycle_macro = 600
                mc.conv_tol        = 1e-8
                mc.ah_level_shift  = shift
                mc.verbose         = 0
                mc.kernel(mc.sort_mo(window, base=0))
                ec = mc.e_tot - mf.e_tot
                if ec < 0 and ec < best_e:
                    best_mc = mc; best_e = ec
                if mc.converged and ec < -0.01:
                    return mc, True
            except: continue

    if best_mc: return best_mc, best_mc.converged
    return None, False


def run_one(struct_name, metal, charge, atoms_str, spin, ligand_label):
    name    = f"PORPH_{struct_name}_spin{spin}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outfile = os.path.join(OUTPUT_DIR, f"{name}.json")

    if os.path.exists(outfile):
        r = json.load(open(outfile))
        if r.get('converged') and r.get('corr_energy', 0) < -0.001:
            log.info(f"SKIP: {name}"); return True

    n_e = count_electrons(atoms_str, charge)
    if (n_e % 2) != (spin % 2):
        json.dump({'name': name, 'status': 'skipped',
                   'reason': 'parity', 'geometry': 'porphyrin'},
                  open(outfile, 'w')); return True

    consts = METAL_CONSTANTS[metal]
    log.info(f"Start: {name}  n_e={n_e}  row={consts['metal_row']}")

    try:
        mol = gto.Mole()
        mol.atom    = atoms_str
        mol.basis   = 'def2-SVP'
        mol.charge  = charge
        mol.spin    = spin
        mol.verbose = 0
        if metal in ECP_METALS:
            mol.ecp = 'def2-SVP'
        mol.build()

        n_act = get_nact_14(mol.nelectron)
        mf    = run_uhf(mol)
        mc, conv = run_casscf_14(mf, mol, n_act)

        if mc is None:
            json.dump({'name': name, 'status': 'failed',
                       'geometry': 'porphyrin', 'metal': metal},
                      open(outfile, 'w')); return False

        ec = mc.e_tot - mf.e_tot
        if ec >= 0:
            json.dump({'name': name, 'status': 'unphysical',
                       'corr_energy': float(ec),
                       'geometry': 'porphyrin'},
                      open(outfile, 'w')); return False

        casdm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
        no, _  = np.linalg.eigh(casdm1)
        no     = np.sort(no)[::-1]
        n_active = sum(1 for n in no if 0.02 < n < 1.98)

        e_m = (mf.mo_energy[0] + mf.mo_energy[1]) / 2
        occ = mf.mo_occ[0] + mf.mo_occ[1]
        oi  = np.where(occ > 0.5)[0]; vi = np.where(occ < 0.5)[0]
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
            'ligand':            'N',           # porphyrin N donors
            'ligand_type':       'porphyrin',   # explicit label
            'n_ligands':         atoms_str.count('\nN') + atoms_str.count('\nN '),
            'charge':            charge,
            'spin':              spin,
            'mult':              spin + 1,
            'geometry':          'porphyrin',   # new geometry type
            'struct_name':       struct_name,
            'coordination':      ligand_label,  # e.g. N4, N4Cl, N4Cl2
            'casscf_size':       14,            # CASSCF(14,14)
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
        log.info(f"  OK: n_active={n_active} Ec={ec:.4f} "
                 f"converged={conv} coord={ligand_label}")
        return True

    except Exception as e:
        log.error(f"  Error {name}: {e}")
        json.dump({'name': name, 'status': 'error', 'reason': str(e),
                   'geometry': 'porphyrin'}, open(outfile, 'w'))
        return False


# ── BUILD JOB LIST ─────────────────────────────────────────────
ALL_JOBS = []
seen     = set()
os.makedirs(OUTPUT_DIR, exist_ok=True)
existing = set(os.path.basename(f)
               for f in __import__('glob').glob(f'{OUTPUT_DIR}/*.json'))

for struct_name, metal, charge, atoms_str, ligand_label in CSD_STRUCTURES:
    spins = SPIN_STATES.get(metal, [0, 2])
    n_e   = count_electrons(atoms_str, charge)
    for spin in spins:
        if (n_e % 2) != (spin % 2): continue
        fname = f"PORPH_{struct_name}_spin{spin}.json"
        if fname in existing: continue
        key = (struct_name, spin)
        if key in seen: continue
        seen.add(key)
        ALL_JOBS.append((struct_name, metal, charge,
                         atoms_str, spin, ligand_label))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        from collections import defaultdict
        by_metal = defaultdict(int)
        by_coord = defaultdict(int)
        for s, m, c, a, sp, lig in ALL_JOBS:
            by_metal[m] += 1
            cn = lig.count('N') + lig.count('Cl') + lig.count('O')
            by_coord[cn] += 1
        print(f"\nTotal jobs:  {len(ALL_JOBS)}")
        print(f"Structures:  {len(CSD_STRUCTURES)}")
        print(f"\nJobs by metal:")
        for m in ['Fe','Mn','Co','Ni','Cu','Zn','Ru','Ir']:
            if by_metal.get(m, 0) > 0:
                print(f"  {m}: {by_metal[m]}")
        print(f"\nNote: all use CASSCF(14,14) — ~3x slower than (10,10)")
        print(f"Est. time: ~{len(ALL_JOBS)*60/60:.0f} hrs serial, "
              f"~{len(ALL_JOBS)*60/60/32:.1f} hrs on 32 cores")
        sys.exit(0)

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if idx >= len(ALL_JOBS): sys.exit(1)
    sys.exit(0 if run_one(*ALL_JOBS[idx]) else 1)
