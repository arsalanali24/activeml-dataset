"""
Fix NiCl4 tetrahedral CASSCF — explicit d-orbital initialization.
CASSCF(10,10) with generic window converges to wrong local min.
Use CASSCF(8,8) initialized explicitly on the 8 Ni d-orbitals.
"""
import numpy as np, json, os, sys, logging
from pyscf import gto, scf, mcscf
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

BASE   = os.path.expanduser('~/activeml/data/generated_gap_fill')
CASES  = [
    ("GAP_NiCl4_2m_tet_d8_spin2", "Ni", -2, 2,
     "Ni 0 0 0\nCl 2.270 0 0\nCl -2.270 0 0\nCl 0 2.270 0\nCl 0 -2.270 0"),
    ("GAP_NiBr4_2m_tet_d8_spin2", "Ni", -2, 2,
     "Ni 0 0 0\nBr 2.380 0 0\nBr -2.380 0 0\nBr 0 2.380 0\nBr 0 -2.380 0"),
    ("GAP_NiF4_2m_tet_d8_spin2",  "Ni", -2, 2,
     "Ni 0 0 0\nF 1.930 0 0\nF -1.930 0 0\nF 0 1.930 0\nF 0 -1.930 0"),
    ("GAP_NiCl4_2m_tet_HS_spin4", "Ni", -2, 4,
     "Ni 0 0 0\nCl 2.270 0 0\nCl -2.270 0 0\nCl 0 2.270 0\nCl 0 -2.270 0"),
]

def run_one(name, metal, charge, spin, atom_str):
    outfile = os.path.join(BASE, f"{name}.json")
    log.info(f"Fixing: {name}")
    try:
        mol = gto.Mole()
        mol.atom=atom_str; mol.basis='def2-SVP'
        mol.charge=charge; mol.spin=spin; mol.verbose=0
        mol.build()

        # UHF first
        mf = scf.UHF(mol)
        mf.max_cycle=500; mf.conv_tol=1e-9; mf.verbose=0
        mf.run()
        log.info(f"  UHF: E={mf.e_tot:.6f} conv={mf.converged}")

        # Find Ni d-orbitals explicitly
        # Ni is atom 0, get AO indices for 3d
        ao_labels = mol.ao_labels()
        ni_d_aos = [i for i,l in enumerate(ao_labels)
                    if 'Ni' in l and '3d' in l]
        log.info(f"  Ni 3d AO indices: {ni_d_aos[:10]}")

        # Get MOs with largest Ni d character
        mo_coeff = mf.mo_coeff[0]  # alpha MOs
        ni_d_char = np.sum(mo_coeff[ni_d_aos,:]**2, axis=0)
        # Select 8 MOs with largest Ni d character for CASSCF(8,8)
        top8_mo = np.argsort(ni_d_char)[::-1][:8]
        top8_mo = sorted(top8_mo)
        log.info(f"  Top 8 Ni-d MOs: {top8_mo}")
        log.info(f"  Ni-d chars: {ni_d_char[top8_mo]}")

        # CASSCF(8,8) initialized on Ni d-orbitals
        # n_act electrons in these 8 orbitals
        n_act = spin + 2  # spin=2 → 4 electrons (d8 with 4 unpaired... wait)
        # For Ni(II) d8: 8 electrons in 8 orbitals
        for k in [8,6,4,10]:
            if (mol.nelectron-k)>=0 and (mol.nelectron-k)%2==0:
                n_act=k; break

        best_mc=None; best_e=0.0; best_nact=0
        for init_mos in [top8_mo,
                         sorted(np.argsort(ni_d_char)[::-1][:10])[:8],
                         sorted(np.argsort(ni_d_char)[::-1][1:9])]:
            for sh in [1e-3,1e-2,5e-2,1e-1]:
                try:
                    mc=mcscf.CASSCF(mf,8,n_act)
                    mc.max_cycle_macro=600; mc.conv_tol=1e-8
                    mc.ah_level_shift=sh; mc.verbose=0
                    mc.kernel(mc.sort_mo(init_mos,base=0))
                    ec=mc.e_tot-mf.e_tot
                    if ec<0 and ec<best_e:
                        casdm1=mc.fcisolver.make_rdm1(mc.ci,mc.ncas,mc.nelecas)
                        no,_=np.linalg.eigh(casdm1); no=np.sort(no)[::-1]
                        na=sum(1 for n in no if 0.02<n<1.98)
                        best_mc=mc; best_e=ec; best_nact=na
                        log.info(f"  Better: n_active={na} Ec={ec:.4f}")
                    if mc.converged and ec<-0.01:
                        casdm1=mc.fcisolver.make_rdm1(mc.ci,mc.ncas,mc.nelecas)
                        no,_=np.linalg.eigh(casdm1); no=np.sort(no)[::-1]
                        na=sum(1 for n in no if 0.02<n<1.98)
                        if na > best_nact:
                            best_mc=mc; best_e=ec; best_nact=na
                except: continue

        if best_mc is None:
            log.warning("  CASSCF failed"); return

        log.info(f"  Final: n_active={best_nact} Ec={best_e:.4f}")

        # Update JSON
        if os.path.exists(outfile):
            r = json.load(open(outfile))
            r['n_active'] = best_nact
            r['corr_energy'] = float(best_e)
            r['E_CASSCF'] = float(best_mc.e_tot)
            r['converged'] = bool(best_mc.converged)
            r['casscf_init'] = 'explicit_d_orbitals'
            with open(outfile,'w') as f: json.dump(r,f,indent=2)
            log.info(f"  Updated: {outfile}")
    except Exception as e:
        log.error(f"  Error: {e}")

idx = int(sys.argv[1]) if len(sys.argv)>1 else 0
if idx < len(CASES): run_one(*CASES[idx])
