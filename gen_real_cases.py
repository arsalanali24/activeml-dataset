"""
gen_real_cases.py
=================
Generates ML1 feature JSON files for 10 real benchmark systems.
Output format identical to generated300/ training data.
Used to validate the trained model against published active spaces.

Systems chosen because:
  1. Published CASSCF active space exists in literature
  2. Cover range of metals, ligands, geometries
  3. Include the Microsoft AutoCAS comparison target

Usage:
  python gen_real_cases.py summary
  python gen_real_cases.py <idx>
  sbatch --array=0-N job_real_cases.sh

After running: download JSON files and test in local notebook.
"""
import numpy as np, json, os, sys, logging
from pyscf import gto, scf, mp

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

OUTPUT_DIR = os.path.expanduser('~/activeml/data/real_cases')
os.makedirs(OUTPUT_DIR, exist_ok=True)

METAL_CONSTANTS = {
    'Fe': {'z_eff':11.18,'zeta_so_cm1': 460,'metal_row':'3d'},
    'Mn': {'z_eff':10.53,'zeta_so_cm1': 355,'metal_row':'3d'},
    'Co': {'z_eff':12.00,'zeta_so_cm1': 533,'metal_row':'3d'},
    'Ni': {'z_eff':12.78,'zeta_so_cm1': 669,'metal_row':'3d'},
    'Ru': {'z_eff':12.33,'zeta_so_cm1': 880,'metal_row':'4d'},
    'Rh': {'z_eff':12.67,'zeta_so_cm1':1097,'metal_row':'4d'},
    'Pd': {'z_eff':13.00,'zeta_so_cm1':1334,'metal_row':'4d'},
    'Ir': {'z_eff':17.00,'zeta_so_cm1':3909,'metal_row':'5d'},
    'Os': {'z_eff':17.17,'zeta_so_cm1':3381,'metal_row':'5d'},
}
ECP_METALS = {'Ru','Rh','Pd','Ir','Os'}

# ══════════════════════════════════════════════════════════════
# BENCHMARK SYSTEMS
# Format: (name, metal, charge, spin, atom_str, ligand,
#          n_ligands, dist_ang, geometry,
#          published_n_active, published_reference)
# ══════════════════════════════════════════════════════════════

REAL_CASES = [

    # ── Case 1: [PdCl4]^2- square planar ──────────────────────
    # Published: n_active=2-4, CASSCF(4,4)
    # Reference: Roos et al. JPCB 2003
    # This is a confidence check — Pd d8 sp square planar
    ("PdCl4_2m_sqpl_real",
     "Pd", -2, 0,
     """Pd  0.000  0.000  0.000
        Cl  2.295  0.000  0.000
        Cl -2.295  0.000  0.000
        Cl  0.000  2.295  0.000
        Cl  0.000 -2.295  0.000""",
     "Cl", 4, 2.295, "sq_pl",
     4, "Roos et al. JPCB 2003 — CASSCF(4,4)"),

    # ── Case 2: RhCl(PH3)3 — Wilkinson's catalyst ─────────────
    # Published: n_active=8, CASSCF(8,8) for Rh(I) d8
    # Reference: Siegbahn et al. JACS 1993
    # Key test of phosphine training data
    ("RhClP3_Wilkinson",
     "Rh", 0, 0,
     """Rh  0.000  0.000  0.000
        Cl  2.383  0.000  0.000
        P  -2.318  0.000  0.000
        P   0.000  2.318  0.000
        P   0.000 -2.318  0.000""",
     "P", 3, 2.318, "sq_pl",
     8, "Siegbahn JACS 1993 — CASSCF(8,8)"),

    # ── Case 3: Fe-porphyrin 5-coord (deoxy-heme model) ───────
    # Published: n_active=8-10, CASSCF(10,10)
    # Reference: Li & Gagliardi JCTC 2018
    # Canonical multireference benchmark
    ("FePorph_5coord_deoxyheme",
     "Fe", 0, 4,
     """Fe  0.000  0.000  0.000
        N   2.010  0.000  0.000
        N  -2.010  0.000  0.000
        N   0.000  2.010  0.000
        N   0.000 -2.010  0.000
        Cl  0.000  0.000  2.280""",
     "N", 4, 2.010, "sqpyr",
     10, "Li & Gagliardi JCTC 2018 — CASSCF(10,10)"),

    # ── Case 4: [Fe(bipy)3]^2+ spin crossover ─────────────────
    # Published: n_active=10, CASSCF(10,10) LS state
    # Reference: Pierloot et al. JCTC 2017
    # Tests Fe bidentate generalisation
    ("Fe_bipy3_SCO",
     "Fe", 2, 0,
     """Fe  0.000  0.000  0.000
        N   2.080  0.000  0.000
        N  -2.080  0.000  0.000
        N   0.000  2.080  0.000
        N   0.000 -2.080  0.000
        N   0.000  0.000  2.080
        N   0.000  0.000 -2.080""",
     "N", 6, 2.080, "oct",
     10, "Pierloot JCTC 2017 — CASSCF(10,10)"),

    # ── Case 5: Fe-PNP pincer (Microsoft complex) ─────────────
    # Published: AutoCAS suggests 59, expert = 12-14
    # Reference: Elfving et al. PhysChemChemPhys 2021 (Microsoft)
    # THE AutoCAS comparison target
    ("Fe_PNP_pincer_Microsoft",
     "Fe", 0, 2,
     """Fe  0.000  0.000  0.000
        N   2.050  0.000  0.000
        P   0.000  2.250  0.000
        P   0.000 -2.250  0.000
        Cl -2.400  0.000  0.000
        Cl  0.000  0.000  2.400
        Cl  0.000  0.000 -2.400""",
     "P", 2, 2.250, "oct",
     12, "Elfving/Microsoft 2021 — expert (12-14) vs AutoCAS (59)"),

    # ── Case 6: IrCl(CO)(PH3)2 — Vaska's compound ─────────────
    # Published: n_active=2, CASSCF(2,2) for Ir(I) d8 closed shell
    # Reference: Budzelaar et al. 2010
    # Tests Ir phosphine — should be very low active space
    ("IrClCO_Vaska",
     "Ir", 0, 0,
     """Ir  0.000  0.000  0.000
        Cl  2.380  0.000  0.000
        C  -1.840  0.000  0.000
        P   0.000  2.340  0.000
        P   0.000 -2.340  0.000""",
     "P", 2, 2.340, "sq_pl",
     2, "Budzelaar 2010 — CASSCF(2,2)"),

    # ── Case 7: OsO4 — osmium tetroxide ───────────────────────
    # Published: n_active=0, Os(VIII) d0 closed shell
    # Reference: Standard inorganic — d0 has zero active space
    # Tests: does model correctly predict zero for fully oxidised?
    ("OsO4_Sharpless",
     "Os", 0, 0,
     """Os  0.000  0.000  0.000
        O   1.710  0.000  0.000
        O  -1.710  0.000  0.000
        O   0.000  1.710  0.000
        O   0.000 -1.710  0.000""",
     "O", 4, 1.710, "tet",
     0, "Standard — Os(VIII) d0, n_active=0"),

    # ── Case 8: [RuCl2(bipy)2] — Ru photocatalyst ─────────────
    # Published: n_active=6, CASSCF(6,8) for Ru(II) d6
    # Reference: Zobel et al. JCTC 2021
    # Tests Ru bidentate
    ("RuCl2bipy2_photocatalyst",
     "Ru", 0, 0,
     """Ru  0.000  0.000  0.000
        N   2.060  0.000  0.000
        N  -2.060  0.000  0.000
        N   0.000  2.060  0.000
        N   0.000 -2.060  0.000
        Cl  0.000  0.000  2.410
        Cl  0.000  0.000 -2.410""",
     "N", 4, 2.060, "oct",
     6, "Zobel JCTC 2021 — CASSCF(6,8)"),

    # ── Case 9: MnCl2(phen)2 — Mn porphyrin analogue ──────────
    # Published: n_active=5, CASSCF(5,5) for Mn(II) d5
    # Reference: Phung et al. JCTC 2016
    # Tests Mn with bidentate N-N ligand
    ("MnCl2phen2_oct",
     "Mn", 0, 5,
     """Mn  0.000  0.000  0.000
        N   2.010  0.000  0.000
        N  -2.010  0.000  0.000
        N   0.000  2.210  0.000
        N   0.000 -2.210  0.000
        Cl  0.000  0.000  2.580
        Cl  0.000  0.000 -2.580""",
     "N", 4, 2.010, "oct",
     5, "Phung JCTC 2016 — CASSCF(5,5)"),

    # ── Case 10: Pd(0) oxidative addition reactant ─────────────
    # Published: n_active=2, Pd(0) d10 low correlation
    # Reference: Your IRC dataset — reactant geometry
    # Tests IRC point 0 — model should predict small active space
    ("Pd_OA_reactant_IRC",
     "Pd", 0, 0,
     """Pd  0.000  0.000  0.000
        P   2.290  0.000  0.000
        P  -2.290  0.000  0.000""",
     "P", 2, 2.290, "linear",
     2, "IRC reactant Pd(0) d10 — n_active~2"),
]


def run_uhf(mol):
    for s in [
        dict(max_cycle=300, conv_tol=1e-10, damp=0.0, level_shift=0.0),
        dict(max_cycle=500, conv_tol=1e-9,  damp=0.3, level_shift=0.2),
        dict(max_cycle=800, conv_tol=1e-8,  damp=0.5, level_shift=0.5),
    ]:
        mf = scf.UHF(mol)
        for k, v in s.items(): setattr(mf, k, v)
        mf.verbose = 0; mf.run()
        if mf.converged: return mf
    return mf


def extract_features(mf, mol, metal, charge, spin,
                     ligand, n_ligands, dist_ang, geometry,
                     consts):
    """Extract all ML1 features from UHF wavefunction."""

    # ── HF orbital features ───────────────────────────────────
    e_a = mf.mo_energy[0]; e_b = mf.mo_energy[1]
    occ_a = mf.mo_occ[0];  occ_b = mf.mo_occ[1]

    occ_idx_a = np.where(occ_a > 0.5)[0]
    occ_idx_b = np.where(occ_b > 0.5)[0]
    virt_idx  = np.where((mf.mo_occ[0] + mf.mo_occ[1]) < 0.5)[0]

    homo_a = float(e_a[occ_idx_a[-1]]) if len(occ_idx_a) else 0.0
    homo_b = float(e_b[occ_idx_b[-1]]) if len(occ_idx_b) else 0.0
    homo_e = max(homo_a, homo_b)
    lumo_e = float(((mf.mo_energy[0] + mf.mo_energy[1])/2)[virt_idx[0]]) \
             if len(virt_idx) else 0.0

    # ── Spin contamination ────────────────────────────────────
    S = spin / 2.0
    spin_contam = float(mf.spin_square()[0] - S*(S+1))

    # ── Alpha-beta overlap ────────────────────────────────────
    ovlp = mol.intor('int1e_ovlp')
    ab_ovlp = float(np.trace(
        mf.mo_coeff[0][:, occ_idx_a].T @
        ovlp @
        mf.mo_coeff[1][:, occ_idx_b]
    )) if len(occ_idx_a) and len(occ_idx_b) else 0.0

    # ── HS-LS energy gap ──────────────────────────────────────
    # Run second UHF at different spin state
    delta_e_hs_ls = 0.0
    if spin > 0:
        try:
            mol2 = mol.copy()
            mol2.spin = 0 if spin > 0 else 2
            mol2.build()
            mf2 = scf.UHF(mol2); mf2.max_cycle=300
            mf2.conv_tol=1e-9; mf2.verbose=0; mf2.run()
            if mf2.converged:
                delta_e_hs_ls = float(mf.e_tot - mf2.e_tot)
        except: pass

    # ── UNO fractional occupations ───────────────────────────
    dm_a = mf.make_rdm1()[0]; dm_b = mf.make_rdm1()[1]
    dm_tot = dm_a + dm_b
    ovlp_mat = mol.intor('int1e_ovlp')
    ps = dm_tot @ ovlp_mat
    uno_occ, _ = np.linalg.eigh(ps)
    uno_occ = np.sort(uno_occ)[::-1]

    n_frac_001 = int(np.sum((uno_occ > 0.01) & (uno_occ < 1.99)))
    n_frac_005 = int(np.sum((uno_occ > 0.05) & (uno_occ < 1.95)))
    n_frac_010 = int(np.sum((uno_occ > 0.10) & (uno_occ < 1.90)))
    n_frac_020 = int(np.sum((uno_occ > 0.20) & (uno_occ < 1.80)))

    # ── MP2 correlation ───────────────────────────────────────
    mp2_corr   = 0.0
    largest_t2 = 0.0
    n_frac_mp2_002 = 0
    n_frac_mp2_005 = 0
    n_frac_mp2_010 = 0
    try:
        pt = mp.UMP2(mf).run()
        mp2_corr = float(pt.e_corr)
        # T2 amplitudes
        t2 = pt.t2
        if t2 is not None:
            t2_aa, t2_ab, t2_bb = t2
            all_t2 = np.concatenate([
                np.abs(t2_aa).flatten(),
                np.abs(t2_ab).flatten(),
                np.abs(t2_bb).flatten()
            ])
            largest_t2 = float(np.max(all_t2)) if len(all_t2) else 0.0
            n_frac_mp2_002 = int(np.sum(all_t2 > 0.002))
            n_frac_mp2_005 = int(np.sum(all_t2 > 0.005))
            n_frac_mp2_010 = int(np.sum(all_t2 > 0.010))
    except Exception as e:
        log.warning(f"  MP2 failed: {e}")

    # ── Mayer bond order (mean over ligands) ──────────────────
    mayer_mean = mayer_std = 0.0
    try:
        from pyscf.lo import Boys
        dm  = mf.make_rdm1()
        ovp = mol.intor('int1e_ovlp')
        ps_a = dm[0] @ ovp; ps_b = dm[1] @ ovp
        mbo_list = []
        metal_idx = 0  # first atom is always metal
        for j in range(1, mol.natm):
            ao_a = mol.aoslice_by_atom()[metal_idx]
            ao_j = mol.aoslice_by_atom()[j]
            ba  = ao_a[2]; ea = ao_a[3]
            bj  = ao_j[2]; ej = ao_j[3]
            mbo = 2*(
                np.sum(ps_a[ba:ea, bj:ej] * ps_a[bj:ej, ba:ea].T) +
                np.sum(ps_b[ba:ea, bj:ej] * ps_b[bj:ej, ba:ea].T)
            )
            mbo_list.append(float(mbo))
        if mbo_list:
            mayer_mean = float(np.mean(mbo_list))
            mayer_std  = float(np.std(mbo_list))
    except: pass

    return {
        'spin_contamination':   spin_contam,
        'homo_lumo_gap':        float(lumo_e - homo_e),
        'homo_lumo_gap_eV':     float((lumo_e - homo_e) * 27.2114),
        'homo_energy':          homo_e,
        'lumo_energy':          lumo_e,
        'homo_ab_gap':          float(abs(homo_a - homo_b)),
        'alpha_beta_overlap':   ab_ovlp,
        'delta_E_HS_LS':        delta_e_hs_ls,
        'mp2_corr':             mp2_corr,
        'corr_energy':          mp2_corr,
        'largest_t2':           largest_t2,
        'n_frac_uno_001':       n_frac_001,
        'n_frac_uno_005':       n_frac_005,
        'n_frac_uno_010':       n_frac_010,
        'n_frac_uno_020':       n_frac_020,
        'n_frac_mp2_002':       n_frac_mp2_002,
        'n_frac_mp2_005':       n_frac_mp2_005,
        'n_frac_mp2_010':       n_frac_mp2_010,
        'mayer_bond_order_mean': mayer_mean,
        'mayer_bond_order_std':  mayer_std,
        'mulliken_metal_charge': 0.0,  # placeholder
        'loewdin_metal_charge':  0.0,
        'n_electrons':          mol.nelectron,
        'z_eff':                consts['z_eff'],
        'zeta_so_cm1':          consts['zeta_so_cm1'],
        'metal_row':            consts['metal_row'],
        'charge':               charge,
        'spin':                 spin,
        'mult':                 spin + 1,
        'ligand':               ligand,
        'n_ligands':            n_ligands,
        'dist_ang':             dist_ang,
        'geometry':             geometry,
        'E_HF':                 float(mf.e_tot),
    }


def run_one(idx):
    case = REAL_CASES[idx]
    (name, metal, charge, spin, atom_str, ligand,
     n_ligands, dist_ang, geometry,
     published_n_active, reference) = case

    outfile = os.path.join(OUTPUT_DIR, f"{name}.json")
    if os.path.exists(outfile):
        r = json.load(open(outfile))
        if r.get('status') == 'ok':
            log.info(f"SKIP: {name}"); return True

    consts = METAL_CONSTANTS[metal]
    log.info(f"\n{'='*55}")
    log.info(f"Case {idx}: {name}")
    log.info(f"  Metal={metal} charge={charge} spin={spin}")
    log.info(f"  Published n_active={published_n_active}")
    log.info(f"  Reference: {reference}")

    try:
        mol = gto.Mole()
        mol.atom    = atom_str
        mol.basis   = 'def2-SVP'
        mol.charge  = charge
        mol.spin    = spin
        mol.verbose = 0
        if metal in ECP_METALS:
            mol.ecp = 'def2-SVP'
        mol.build()

        log.info(f"  n_electrons = {mol.nelectron}")

        # UHF
        mf = run_uhf(mol)
        log.info(f"  UHF: E={mf.e_tot:.6f}  converged={mf.converged}")

        # Extract all features
        feats = extract_features(
            mf, mol, metal, charge, spin,
            ligand, n_ligands, dist_ang, geometry, consts)

        # Build result — same schema as training data
        result = {
            'name':               name,
            'metal':              metal,
            'ligand':             ligand,
            'n_ligands':          n_ligands,
            'charge':             charge,
            'spin':               spin,
            'mult':               spin + 1,
            'dist_ang':           dist_ang,
            'geometry':           geometry,
            'status':             'ok',
            'converged':          bool(mf.converged),
            'published_n_active': published_n_active,
            'reference':          reference,
            # All ML1 features
            **feats,
            # Placeholder for model prediction (filled later)
            'predicted_n_active': None,
            'prediction_correct': None,
        }

        with open(outfile, 'w') as f:
            json.dump(result, f, indent=2)

        log.info(f"  OK: spin_contam={feats['spin_contamination']:.4f}")
        log.info(f"  homo_lumo_gap={feats['homo_lumo_gap']:.4f}")
        log.info(f"  mp2_corr={feats['mp2_corr']:.4f}")
        return True

    except Exception as e:
        log.error(f"  FAILED: {e}")
        json.dump({'name': name, 'status': 'error',
                   'reason': str(e)}, open(outfile, 'w'))
        return False


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        print(f"\nReal case benchmarks: {len(REAL_CASES)}")
        print(f"{'Idx':>3}  {'Name':<35} {'Metal':<4} "
              f"{'Spin':>4}  {'Pub n_active':>12}  Reference")
        print("-" * 100)
        for i, c in enumerate(REAL_CASES):
            name, metal, charge, spin, _, _, _, _, geom, pub, ref = c
            print(f"  {i:2d}  {name:<35} {metal:<4} "
                  f"{spin:>4}  {pub:>12}  {ref}")
        print(f"\nOutput: {OUTPUT_DIR}")
        sys.exit(0)

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if idx >= len(REAL_CASES):
        sys.exit(0)
    sys.exit(0 if run_one(idx) else 1)
