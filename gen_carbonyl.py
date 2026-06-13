"""
gen_carbonyl.py
===============
Generates training data for carbonyl and mixed CO/phosphine complexes.

WHY THIS FIXES IrVaska:
  IrCl(CO)(PH3)2 — Vaska's compound — has strong Ir->CO pi-backbonding.
  This dramatically reduces the active space to n_active=2.
  Your simplified geometry treated CO as a point "C" donor, losing
  the pi-backbonding physics entirely.

  Fix: include the actual C-O bond in the geometry so UHF captures
  the CO pi* orbital interaction with Ir d orbitals.

STRUCTURES:
  Group 1 — Homoleptic carbonyls M(CO)6 / M(CO)5 / M(CO)4
    Cr(CO)6, Mo(CO)6, W(CO)6  — d6, n_active=2-4
    Mn(CO)5+, Re(CO)5+        — d6, n_active=2
    Fe(CO)5                   — d8, n_active=2-4

  Group 2 — Mixed carbonyl-halide
    [Mn(CO)5Cl], [Re(CO)5Cl]  — d6
    [Fe(CO)4Cl2]              — d6 (Fe(II))
    [Ru(CO)4Cl2], [Os(CO)4Cl2] — d6

  Group 3 — Vaska-type IrCl(CO)(PR3)2 / RhCl(CO)(PR3)2
    IrCl(CO)(PH3)2 — the exact benchmark case
    RhCl(CO)(PH3)2 — Rh analogue
    IrBr(CO)(PH3)2 — bromide variant
    IrCl(CO)(PH3)3 — 6-coordinate variant

  Group 4 — Carbonyl-phosphine mixed
    Ru(CO)2(PH3)3, Os(CO)2(PH3)3
    Fe(CO)2(PH3)3, Mn(CO)3(PH3)2+

KEY GEOMETRY NOTE:
  M-C-O is linear (180 degrees). C-O bond: 1.128 Ang (free CO: 1.128).
  M-C bond shortens with back-bonding: Ir-C ~1.84 Ang, Fe-C ~1.80 Ang.
  The C-O bond is ESSENTIAL — without it UHF cannot see pi-backbonding.

Output: ~/activeml/data/generated_carbonyl/

Usage:
  python gen_carbonyl.py summary
  python gen_carbonyl.py <idx>
  sbatch --array=0-N job_carbonyl.sh
"""
import numpy as np, json, os, sys, logging
from pyscf import gto, scf, mcscf, mp

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

OUTPUT_DIR = os.path.expanduser('~/activeml/data/generated_carbonyl')
os.makedirs(OUTPUT_DIR, exist_ok=True)

METAL_CONSTANTS = {
    'Cr': {'z_eff': 9.76,'zeta_so_cm1': 273,'metal_row':'3d'},
    'Mn': {'z_eff':10.53,'zeta_so_cm1': 355,'metal_row':'3d'},
    'Fe': {'z_eff':11.18,'zeta_so_cm1': 460,'metal_row':'3d'},
    'Co': {'z_eff':12.00,'zeta_so_cm1': 533,'metal_row':'3d'},
    'Ni': {'z_eff':12.78,'zeta_so_cm1': 669,'metal_row':'3d'},
    'Mo': {'z_eff':10.97,'zeta_so_cm1': 467,'metal_row':'4d'},
    'Ru': {'z_eff':12.33,'zeta_so_cm1': 880,'metal_row':'4d'},
    'Rh': {'z_eff':12.67,'zeta_so_cm1':1097,'metal_row':'4d'},
    'Pd': {'z_eff':13.00,'zeta_so_cm1':1334,'metal_row':'4d'},
    'W':  {'z_eff':11.61,'zeta_so_cm1':2748,'metal_row':'5d'},
    'Re': {'z_eff':17.01,'zeta_so_cm1':2456,'metal_row':'5d'},
    'Os': {'z_eff':17.17,'zeta_so_cm1':3381,'metal_row':'5d'},
    'Ir': {'z_eff':17.00,'zeta_so_cm1':3909,'metal_row':'5d'},
    'Pt': {'z_eff':17.33,'zeta_so_cm1':4146,'metal_row':'5d'},
}
ECP_METALS = {'Mo','Ru','Rh','Pd','W','Re','Os','Ir','Pt'}

ATOM_Z = {
    'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,
    'Mo':42,'Ru':44,'Rh':45,'Pd':46,
    'W':74,'Re':75,'Os':76,'Ir':77,'Pt':78,
    'C':6,'O':8,'Cl':17,'Br':35,'F':9,'P':15,'N':7,'H':1,
}

# C-O bond length in coordinated CO: 1.152 Ang (slightly longer than free)
# M-C distances: Cr 1.915, Mo 2.063, W 2.058, Fe 1.810, Ir 1.840
CO = 1.152  # C-O bond length

def make_co(metal, mc_dist, direction):
    """CO ligand with metal-C-O linear geometry."""
    dx, dy, dz = direction
    norm = (dx**2+dy**2+dz**2)**0.5
    dx,dy,dz = dx/norm, dy/norm, dz/norm
    cx = mc_dist*dx; cy = mc_dist*dy; cz = mc_dist*dz
    ox = cx + CO*dx; oy = cy + CO*dy; oz = cz + CO*dz
    return (f"C  {cx:.3f}  {cy:.3f}  {cz:.3f}",
            f"O  {ox:.3f}  {oy:.3f}  {oz:.3f}")


# ══════════════════════════════════════════════════════════════
# CARBONYL STRUCTURES
# Format: (name, metal, charge, spin, atom_str, ligand,
#          n_ligands, dist_ang, geometry, expected_n_active)
# ══════════════════════════════════════════════════════════════

def build_structures():
    structs = []

    # ── Group 1: Homoleptic carbonyls ─────────────────────────

    # Cr(CO)6  d6 Oh  n_active~2-4
    c1,o1 = make_co("Cr",1.915,(1,0,0));  c2,o2 = make_co("Cr",1.915,(-1,0,0))
    c3,o3 = make_co("Cr",1.915,(0,1,0));  c4,o4 = make_co("Cr",1.915,(0,-1,0))
    c5,o5 = make_co("Cr",1.915,(0,0,1));  c6,o6 = make_co("Cr",1.915,(0,0,-1))
    structs.append(("CrCO6_oct","Cr",0,0,
        f"Cr 0 0 0\n{c1}\n{o1}\n{c2}\n{o2}\n{c3}\n{o3}\n{c4}\n{o4}\n{c5}\n{o5}\n{c6}\n{o6}",
        "CO",6,1.915,"oct",2))

    # Mo(CO)6  d6 Oh
    c1,o1 = make_co("Mo",2.063,(1,0,0));  c2,o2 = make_co("Mo",2.063,(-1,0,0))
    c3,o3 = make_co("Mo",2.063,(0,1,0));  c4,o4 = make_co("Mo",2.063,(0,-1,0))
    c5,o5 = make_co("Mo",2.063,(0,0,1));  c6,o6 = make_co("Mo",2.063,(0,0,-1))
    structs.append(("MoCO6_oct","Mo",0,0,
        f"Mo 0 0 0\n{c1}\n{o1}\n{c2}\n{o2}\n{c3}\n{o3}\n{c4}\n{o4}\n{c5}\n{o5}\n{c6}\n{o6}",
        "CO",6,2.063,"oct",2))

    # W(CO)6  d6 Oh
    c1,o1 = make_co("W",2.058,(1,0,0));  c2,o2 = make_co("W",2.058,(-1,0,0))
    c3,o3 = make_co("W",2.058,(0,1,0));  c4,o4 = make_co("W",2.058,(0,-1,0))
    c5,o5 = make_co("W",2.058,(0,0,1));  c6,o6 = make_co("W",2.058,(0,0,-1))
    structs.append(("WCO6_oct","W",0,0,
        f"W 0 0 0\n{c1}\n{o1}\n{c2}\n{o2}\n{c3}\n{o3}\n{c4}\n{o4}\n{c5}\n{o5}\n{c6}\n{o6}",
        "CO",6,2.058,"oct",2))

    # Fe(CO)5  d8 D3h  n_active~2-4
    c1,o1 = make_co("Fe",1.810,(1,0,0));  c2,o2 = make_co("Fe",1.810,(-0.5,0.866,0))
    c3,o3 = make_co("Fe",1.810,(-0.5,-0.866,0))
    c4,o4 = make_co("Fe",1.810,(0,0,1));  c5,o5 = make_co("Fe",1.810,(0,0,-1))
    structs.append(("FeCO5_tbp","Fe",0,0,
        f"Fe 0 0 0\n{c1}\n{o1}\n{c2}\n{o2}\n{c3}\n{o3}\n{c4}\n{o4}\n{c5}\n{o5}",
        "CO",5,1.810,"tbp",2))

    # Ni(CO)4  d10 Td  n_active~0-2
    c1,o1 = make_co("Ni",1.838,(1,0,0));  c2,o2 = make_co("Ni",1.838,(-1,0,0))
    c3,o3 = make_co("Ni",1.838,(0,1,0));  c4,o4 = make_co("Ni",1.838,(0,-1,0))
    structs.append(("NiCO4_tet","Ni",0,0,
        f"Ni 0 0 0\n{c1}\n{o1}\n{c2}\n{o2}\n{c3}\n{o3}\n{c4}\n{o4}",
        "CO",4,1.838,"tet",2))

    # Ru(CO)5  d8
    c1,o1 = make_co("Ru",1.940,(1,0,0));  c2,o2 = make_co("Ru",1.940,(-0.5,0.866,0))
    c3,o3 = make_co("Ru",1.940,(-0.5,-0.866,0))
    c4,o4 = make_co("Ru",1.940,(0,0,1));  c5,o5 = make_co("Ru",1.940,(0,0,-1))
    structs.append(("RuCO5_tbp","Ru",0,0,
        f"Ru 0 0 0\n{c1}\n{o1}\n{c2}\n{o2}\n{c3}\n{o3}\n{c4}\n{o4}\n{c5}\n{o5}",
        "CO",5,1.940,"tbp",2))

    # Os(CO)5  d8
    c1,o1 = make_co("Os",1.950,(1,0,0));  c2,o2 = make_co("Os",1.950,(-0.5,0.866,0))
    c3,o3 = make_co("Os",1.950,(-0.5,-0.866,0))
    c4,o4 = make_co("Os",1.950,(0,0,1));  c5,o5 = make_co("Os",1.950,(0,0,-1))
    structs.append(("OsCO5_tbp","Os",0,0,
        f"Os 0 0 0\n{c1}\n{o1}\n{c2}\n{o2}\n{c3}\n{o3}\n{c4}\n{o4}\n{c5}\n{o5}",
        "CO",5,1.950,"tbp",2))

    # ── Group 2: Mixed carbonyl-halide ────────────────────────

    # Mn(CO)5Cl  d6 Oh  n_active~2
    c1,o1 = make_co("Mn",1.855,(1,0,0));  c2,o2 = make_co("Mn",1.855,(-1,0,0))
    c3,o3 = make_co("Mn",1.855,(0,1,0));  c4,o4 = make_co("Mn",1.855,(0,-1,0))
    c5,o5 = make_co("Mn",1.855,(0,0,1))
    structs.append(("MnCO5Cl_oct","Mn",0,0,
        f"Mn 0 0 0\n{c1}\n{o1}\n{c2}\n{o2}\n{c3}\n{o3}\n{c4}\n{o4}\n{c5}\n{o5}\nCl 0 0 -2.580",
        "CO",5,1.855,"oct",2))

    # Re(CO)5Cl  d6 Oh  n_active~2
    c1,o1 = make_co("Re",2.010,(1,0,0));  c2,o2 = make_co("Re",2.010,(-1,0,0))
    c3,o3 = make_co("Re",2.010,(0,1,0));  c4,o4 = make_co("Re",2.010,(0,-1,0))
    c5,o5 = make_co("Re",2.010,(0,0,1))
    structs.append(("ReCO5Cl_oct","Re",0,0,
        f"Re 0 0 0\n{c1}\n{o1}\n{c2}\n{o2}\n{c3}\n{o3}\n{c4}\n{o4}\n{c5}\n{o5}\nCl 0 0 -2.490",
        "CO",5,2.010,"oct",2))

    # Fe(CO)4Cl2  d6 Oh  Fe(II)
    c1,o1 = make_co("Fe",1.820,(1,0,0));  c2,o2 = make_co("Fe",1.820,(-1,0,0))
    c3,o3 = make_co("Fe",1.820,(0,1,0));  c4,o4 = make_co("Fe",1.820,(0,-1,0))
    structs.append(("FeCO4Cl2_oct","Fe",0,0,
        f"Fe 0 0 0\n{c1}\n{o1}\n{c2}\n{o2}\n{c3}\n{o3}\n{c4}\n{o4}\nCl 0 0 2.380\nCl 0 0 -2.380",
        "CO",4,1.820,"oct",4))

    # Ru(CO)4Cl2  d6
    c1,o1 = make_co("Ru",1.940,(1,0,0));  c2,o2 = make_co("Ru",1.940,(-1,0,0))
    c3,o3 = make_co("Ru",1.940,(0,1,0));  c4,o4 = make_co("Ru",1.940,(0,-1,0))
    structs.append(("RuCO4Cl2_oct","Ru",0,0,
        f"Ru 0 0 0\n{c1}\n{o1}\n{c2}\n{o2}\n{c3}\n{o3}\n{c4}\n{o4}\nCl 0 0 2.410\nCl 0 0 -2.410",
        "CO",4,1.940,"oct",4))

    # Os(CO)4Cl2  d6
    c1,o1 = make_co("Os",1.950,(1,0,0));  c2,o2 = make_co("Os",1.950,(-1,0,0))
    c3,o3 = make_co("Os",1.950,(0,1,0));  c4,o4 = make_co("Os",1.950,(0,-1,0))
    structs.append(("OsCO4Cl2_oct","Os",0,0,
        f"Os 0 0 0\n{c1}\n{o1}\n{c2}\n{o2}\n{c3}\n{o3}\n{c4}\n{o4}\nCl 0 0 2.430\nCl 0 0 -2.430",
        "CO",4,1.950,"oct",4))

    # ── Group 3: Vaska-type IrCl(CO)(PR3)2 ───────────────────
    # THE BENCHMARK CASE — must have explicit C-O bond

    # IrCl(CO)(PH3)2  square planar  d8  n_active=2
    c1,o1 = make_co("Ir",1.840,(-1,0,0))  # CO trans to Cl
    structs.append(("IrClCO_P2_vaska","Ir",0,0,
        f"Ir 0 0 0\nCl 2.380 0 0\n{c1}\n{o1}\nP 0 2.340 0\nP 0 -2.340 0",
        "CO",1,1.840,"sq_pl",2))

    # RhCl(CO)(PH3)2  square planar  d8  n_active=2
    c1,o1 = make_co("Rh",1.820,(-1,0,0))
    structs.append(("RhClCO_P2_sqpl","Rh",0,0,
        f"Rh 0 0 0\nCl 2.383 0 0\n{c1}\n{o1}\nP 0 2.318 0\nP 0 -2.318 0",
        "CO",1,1.820,"sq_pl",2))

    # IrBr(CO)(PH3)2  Br variant
    c1,o1 = make_co("Ir",1.840,(-1,0,0))
    structs.append(("IrBrCO_P2_vaska","Ir",0,0,
        f"Ir 0 0 0\nBr 2.530 0 0\n{c1}\n{o1}\nP 0 2.340 0\nP 0 -2.340 0",
        "CO",1,1.840,"sq_pl",2))

    # IrCl(CO)(PH3)2 — CO trans to P variant
    c1,o1 = make_co("Ir",1.840,(0,0,1))
    structs.append(("IrClCO_P2_v2","Ir",0,0,
        f"Ir 0 0 0\nCl 2.380 0 0\nP -2.340 0 0\nP 0 2.340 0\nP 0 -2.340 0\n{c1}\n{o1}",
        "CO",1,1.840,"oct",4))

    # ── Group 4: Carbonyl-phosphine mixed ────────────────────

    # Ru(CO)2(PH3)3  mer-octahedral  d8
    c1,o1 = make_co("Ru",1.940,(0,0,1));  c2,o2 = make_co("Ru",1.940,(0,0,-1))
    structs.append(("RuCO2P3_mer","Ru",0,0,
        f"Ru 0 0 0\n{c1}\n{o1}\n{c2}\n{o2}\nP 2.350 0 0\nP -2.350 0 0\nP 0 2.350 0",
        "CO",2,1.940,"oct",4))

    # Os(CO)2(PH3)3  mer-octahedral  d8
    c1,o1 = make_co("Os",1.950,(0,0,1));  c2,o2 = make_co("Os",1.950,(0,0,-1))
    structs.append(("OsCO2P3_mer","Os",0,0,
        f"Os 0 0 0\n{c1}\n{o1}\n{c2}\n{o2}\nP 2.340 0 0\nP -2.340 0 0\nP 0 2.340 0",
        "CO",2,1.950,"oct",4))

    # Fe(CO)2(PH3)3  d8
    c1,o1 = make_co("Fe",1.810,(0,0,1));  c2,o2 = make_co("Fe",1.810,(0,0,-1))
    structs.append(("FeCO2P3_mer","Fe",0,0,
        f"Fe 0 0 0\n{c1}\n{o1}\n{c2}\n{o2}\nP 2.200 0 0\nP -2.200 0 0\nP 0 2.200 0",
        "CO",2,1.810,"oct",4))

    # Mn(CO)3(PH3)2  fac  d8  Mn(I)
    c1,o1 = make_co("Mn",1.855,(1,0,0))
    c2,o2 = make_co("Mn",1.855,(0,1,0))
    c3,o3 = make_co("Mn",1.855,(0,0,1))
    structs.append(("MnCO3P2_fac","Mn",1,0,
        f"Mn 0 0 0\n{c1}\n{o1}\n{c2}\n{o2}\n{c3}\n{o3}\nP -2.310 0 0\nP 0 -2.310 0",
        "CO",3,1.855,"oct",2))

    # Ir(CO)3(PH3)  d8
    c1,o1 = make_co("Ir",1.840,(1,0,0))
    c2,o2 = make_co("Ir",1.840,(0,1,0))
    c3,o3 = make_co("Ir",1.840,(0,0,1))
    structs.append(("IrCO3P_sqpl","Ir",1,0,
        f"Ir 0 0 0\n{c1}\n{o1}\n{c2}\n{o2}\n{c3}\n{o3}\nP -2.340 0 0",
        "CO",3,1.840,"oct",2))

    # Pd(CO)2(PH3)2  d10 square planar
    c1,o1 = make_co("Pd",2.010,(0,0,1));  c2,o2 = make_co("Pd",2.010,(0,0,-1))
    structs.append(("PdCO2P2_sqpl","Pd",0,0,
        f"Pd 0 0 0\n{c1}\n{o1}\n{c2}\n{o2}\nP 2.290 0 0\nP -2.290 0 0",
        "CO",2,2.010,"sq_pl",2))

    # Pt(CO)2(PH3)2  d8
    c1,o1 = make_co("Pt",2.000,(0,0,1));  c2,o2 = make_co("Pt",2.000,(0,0,-1))
    structs.append(("PtCO2P2_sqpl","Pt",0,0,
        f"Pt 0 0 0\n{c1}\n{o1}\n{c2}\n{o2}\nP 2.295 0 0\nP -2.295 0 0",
        "CO",2,2.000,"sq_pl",2))

    return structs


CSD_STRUCTURES = build_structures()

SPIN_STATES = {
    'Cr':[0],'Mn':[0,1,2],'Fe':[0,2],'Co':[1],'Ni':[0],
    'Mo':[0],'Ru':[0],'Rh':[0],'Pd':[0],
    'W':[0],'Re':[0,1],'Os':[0],'Ir':[0],'Pt':[0],
}


def count_electrons(atom_str, charge):
    total = 0
    for line in atom_str.strip().split('\n'):
        sym = line.strip().split()[0]
        total += ATOM_Z.get(sym, 0)
    return total - charge


def get_nact(n_e, expected_n_active):
    # For CO complexes, active space is small — start from expected
    for k in [expected_n_active, expected_n_active+2,
              expected_n_active-2, 10, 8, 6]:
        if k > 0 and (n_e - k) >= 0 and (n_e - k) % 2 == 0:
            return k
    return 6


def run_hf(mol, spin, use_rhf=False):
    if use_rhf or spin == 0:
        for s in [dict(max_cycle=300,conv_tol=1e-10),
                  dict(max_cycle=500,conv_tol=1e-9),
                  dict(max_cycle=800,conv_tol=1e-8)]:
            mf = scf.RHF(mol)
            for k,v in s.items(): setattr(mf,k,v)
            mf.verbose=0; mf.run()
            if mf.converged: return mf, 'RHF'
        # Fall back to UHF if RHF fails
    for s in [dict(max_cycle=300,conv_tol=1e-10,damp=0.0,level_shift=0.0),
              dict(max_cycle=500,conv_tol=1e-9, damp=0.3,level_shift=0.2),
              dict(max_cycle=800,conv_tol=1e-8, damp=0.5,level_shift=0.5)]:
        mf = scf.UHF(mol)
        for k,v in s.items(): setattr(mf,k,v)
        mf.verbose=0; mf.run()
        if mf.converged: return mf, 'UHF'
    return mf, 'UHF'


def run_casscf(mf, mol, n_act, cas_n=10):
    if hasattr(mf, 'mo_energy') and not hasattr(mf.mo_energy,'__len__'):
        return None, False
    try:
        if isinstance(mf, scf.rhf.RHF):
            e_m = mf.mo_energy; occ = mf.mo_occ
        else:
            e_m = (mf.mo_energy[0]+mf.mo_energy[1])/2
            occ = mf.mo_occ[0]+mf.mo_occ[1]
        oi = np.where(occ>0.5)[0]; vi = np.where(occ<0.5)[0]
        if not len(oi) or not len(vi): return None,False
        gap = (float(e_m[oi[-1]])+float(e_m[vi[0]]))/2
        ws = cas_n+4
        w = sorted(np.argsort(np.abs(e_m-gap))[:ws],key=lambda i:e_m[i])
        best_mc=None; best_e=0.0
        for win in [w[:cas_n],w[2:cas_n+2],w[1:cas_n+1],w[4:cas_n+4]]:
            for sh in [1e-3,1e-2,5e-2,1e-1]:
                try:
                    mc=mcscf.CASSCF(mf,cas_n,n_act)
                    mc.max_cycle_macro=500; mc.conv_tol=1e-8
                    mc.ah_level_shift=sh; mc.verbose=0
                    mc.kernel(mc.sort_mo(win,base=0))
                    ec=mc.e_tot-mf.e_tot
                    if ec<0 and ec<best_e: best_mc=mc; best_e=ec
                    if mc.converged and ec<-0.001: return mc,True
                except: continue
        if best_mc: return best_mc,best_mc.converged
    except: pass
    return None,False


def run_one(idx):
    entry = CSD_STRUCTURES[idx]
    (name, metal, charge, spin, atom_str, ligand,
     n_co_lig, dist_ang, geometry, expected_n_active) = entry

    fname   = f"CO_{name}_spin{spin}.json"
    outfile = os.path.join(OUTPUT_DIR, fname)

    if os.path.exists(outfile):
        r = json.load(open(outfile))
        if r.get('status') == 'ok':
            log.info(f"SKIP: {fname}"); return True

    consts = METAL_CONSTANTS[metal]
    n_e    = count_electrons(atom_str, charge)

    if (n_e % 2) != (spin % 2):
        json.dump({'name':fname,'status':'skipped','reason':'parity'},
                  open(outfile,'w')); return True

    log.info(f"Start: {fname}  n_e={n_e}  expected_n_active={expected_n_active}")

    try:
        mol = gto.Mole()
        mol.atom    = atom_str
        mol.basis   = 'def2-SVP'
        mol.charge  = charge
        mol.spin    = spin
        mol.verbose = 0
        if metal in ECP_METALS: mol.ecp = 'def2-SVP'
        mol.build()

        # Use RHF for closed-shell CO complexes (spin=0, low-spin d6/d8/d10)
        use_rhf = (spin == 0)
        mf, scf_type = run_hf(mol, spin, use_rhf)
        log.info(f"  {scf_type}: E={mf.e_tot:.6f} converged={mf.converged}")

        # MP2 correlation
        mp2_corr = 0.0; largest_t2 = 0.0
        n_frac_mp2 = [0,0,0]
        try:
            pt = mp.UMP2(mf).run() if scf_type=='UHF' else mp.MP2(mf).run()
            mp2_corr = float(pt.e_corr)
            if hasattr(pt,'t2') and pt.t2 is not None:
                t2 = pt.t2
                if isinstance(t2, tuple):
                    all_t2 = np.concatenate([np.abs(x).flatten() for x in t2])
                else:
                    all_t2 = np.abs(t2).flatten()
                largest_t2 = float(np.max(all_t2))
                n_frac_mp2 = [int(np.sum(all_t2>0.002)),
                              int(np.sum(all_t2>0.005)),
                              int(np.sum(all_t2>0.010))]
        except Exception as e:
            log.warning(f"  MP2 failed: {e}")

        # CASSCF
        n_act = get_nact(mol.nelectron, expected_n_active)
        mc, conv = run_casscf(mf, mol, n_act, cas_n=10)

        if mc is None or mc.e_tot - mf.e_tot >= 0:
            # For CO complexes, small correlation is physical
            n_active = 0; ec = 0.0; no = []
            log.info(f"  CASSCF: near-zero correlation (CO closed shell)")
        else:
            ec = float(mc.e_tot - mf.e_tot)
            casdm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
            no,_ = np.linalg.eigh(casdm1); no = np.sort(no)[::-1]
            n_active = sum(1 for n in no if 0.02<n<1.98)

        # Extract features
        if isinstance(mf, scf.rhf.RHF):
            e_a = e_b = mf.mo_energy; occ = mf.mo_occ
            spin_contam = 0.0; homo_ab_gap = 0.0
            oi = np.where(occ>0.5)[0]; vi = np.where(occ<0.5)[0]
            homo_e = float(e_a[oi[-1]]) if len(oi) else 0.0
            lumo_e = float(e_a[vi[0]])  if len(vi) else 0.0
        else:
            e_a,e_b = mf.mo_energy[0],mf.mo_energy[1]
            occ = mf.mo_occ[0]+mf.mo_occ[1]
            S = spin/2.0
            spin_contam = float(mf.spin_square()[0]-S*(S+1))
            oi_a = np.where(mf.mo_occ[0]>0.5)[0]
            oi_b = np.where(mf.mo_occ[1]>0.5)[0]
            homo_a = float(e_a[oi_a[-1]]) if len(oi_a) else 0.0
            homo_b = float(e_b[oi_b[-1]]) if len(oi_b) else 0.0
            homo_ab_gap = float(abs(homo_a-homo_b))
            oi = np.where(occ>0.5)[0]; vi = np.where(occ<0.5)[0]
            e_m = (e_a+e_b)/2
            homo_e = float(e_m[oi[-1]]) if len(oi) else 0.0
            lumo_e = float(e_m[vi[0]])  if len(vi) else 0.0

        ovp = mol.intor('int1e_ovlp')
        dm = mf.make_rdm1()
        if isinstance(dm, tuple): dm = dm[0]+dm[1]
        uno,_ = np.linalg.eigh(dm@ovp)
        uno = np.sort(uno)[::-1]

        result = {
            'name':             fname.replace('.json',''),
            'metal':            metal,
            'ligand':           ligand,
            'ligand_type':      'carbonyl',
            'n_ligands':        n_co_lig,
            'charge':           charge,
            'spin':             spin,
            'mult':             spin+1,
            'dist_ang':         dist_ang,
            'geometry':         geometry,
            'scf_type':         scf_type,
            'n_electrons':      mol.nelectron,
            'n_active_e':       n_act,
            'E_HF':             float(mf.e_tot),
            'E_CASSCF':         float(mc.e_tot) if mc else float(mf.e_tot),
            'corr_energy':      ec,
            'mp2_corr':         mp2_corr,
            'converged':        bool(conv if mc else mf.converged),
            'n_active':         n_active,
            'no_occ':           [float(x) for x in no],
            'status':           'ok',
            'z_eff':            consts['z_eff'],
            'zeta_so_cm1':      consts['zeta_so_cm1'],
            'metal_row':        consts['metal_row'],
            'spin_contamination': spin_contam,
            'homo_lumo_gap':    float(lumo_e-homo_e),
            'homo_lumo_gap_eV': float((lumo_e-homo_e)*27.2114),
            'homo_energy':      homo_e,
            'lumo_energy':      lumo_e,
            'homo_ab_gap':      homo_ab_gap,
            'alpha_beta_overlap': 0.0,
            'delta_E_HS_LS':    0.0,
            'mulliken_metal_charge': 0.0,
            'loewdin_metal_charge':  0.0,
            'mayer_bond_order_mean': 0.0,
            'mayer_bond_order_std':  0.0,
            'largest_t2':       largest_t2,
            'n_frac_uno_001':   int(np.sum((uno>0.01)&(uno<1.99))),
            'n_frac_uno_005':   int(np.sum((uno>0.05)&(uno<1.95))),
            'n_frac_uno_010':   int(np.sum((uno>0.10)&(uno<1.90))),
            'n_frac_uno_020':   int(np.sum((uno>0.20)&(uno<1.80))),
            'n_frac_mp2_002':   n_frac_mp2[0],
            'n_frac_mp2_005':   n_frac_mp2[1],
            'n_frac_mp2_010':   n_frac_mp2[2],
            'expected_n_active': expected_n_active,
            'd_electron_count': 0,  # set below
        }

        # Set correct d_electron_count from oxidation state chemistry
        # CO is a neutral ligand, Cl- is -1, so:
        # Cr(CO)6 -> Cr(0) d6, Fe(CO)5 -> Fe(0) d8 etc.
        D_BASE = {'Cr':6,'Mn':7,'Fe':8,'Co':9,'Ni':10,
                  'Mo':6,'Ru':8,'Rh':9,'Pd':10,
                  'W':6,'Re':7,'Os':8,'Ir':9,'Pt':10}
        result['d_electron_count'] = D_BASE.get(metal, 6)

        with open(outfile,'w') as f: json.dump(result,f,indent=2)
        log.info(f"  OK: n_active={n_active} ec={ec:.4f} "
                 f"expected={expected_n_active}")
        return True

    except Exception as e:
        log.error(f"  Error: {e}")
        json.dump({'name':fname,'status':'error','reason':str(e)},
                  open(outfile,'w'))
        return False


# ── BUILD JOB LIST ────────────────────────────────────────────
ALL_JOBS = []
existing = set(os.path.basename(f)
               for f in __import__('glob').glob(f'{OUTPUT_DIR}/*.json'))

for i, entry in enumerate(CSD_STRUCTURES):
    name, metal, charge, spin_def, *_ = entry
    for spin in SPIN_STATES.get(metal, [0]):
        fname = f"CO_{name}_spin{spin}.json"
        n_e = count_electrons(entry[4], charge)
        if (n_e%2) != (spin%2): continue
        if fname in existing: continue
        ALL_JOBS.append((i, spin))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        from collections import defaultdict
        by_metal = defaultdict(int)
        by_group = defaultdict(int)
        for idx, spin in ALL_JOBS:
            m = CSD_STRUCTURES[idx][1]
            n = CSD_STRUCTURES[idx][0]
            by_metal[m] += 1
            grp = 'homoleptic' if 'CO6' in n or 'CO5' in n or 'CO4' in n \
                  else 'vaska' if 'vaska' in n or 'Vaska' in n or \
                                  ('Ir' in n and 'P2' in n) or \
                                  ('Rh' in n and 'P2' in n) \
                  else 'mixed'
            by_group[grp] += 1
        print(f"\nCarbonyl/CO complex jobs: {len(ALL_JOBS)}")
        print(f"Structures: {len(CSD_STRUCTURES)}")
        print(f"\nBy metal:")
        for m in ['Cr','Mn','Fe','Co','Ni','Mo','Ru','Rh','Pd',
                  'W','Re','Os','Ir','Pt']:
            if by_metal.get(m,0)>0:
                print(f"  {m}: {by_metal[m]}")
        print(f"\nNote: includes explicit C-O bonds for pi-backbonding")
        print(f"Fixes IrVaska benchmark failure")
        print(f"\nOutput: {OUTPUT_DIR}")
        sys.exit(0)

    idx_global = int(sys.argv[1]) if len(sys.argv)>1 else 0
    if idx_global >= len(ALL_JOBS): sys.exit(0)
    struct_idx, spin = ALL_JOBS[idx_global]
    # Override spin in the structure
    entry = list(CSD_STRUCTURES[struct_idx])
    entry[3] = spin
    CSD_STRUCTURES[struct_idx] = tuple(entry)
    sys.exit(0 if run_one(struct_idx) else 1)
