#!/usr/bin/env python3
"""
asf_all_cases.py
----------------
Runs ASF benchmark for all remaining test cases sequentially.
Uses the proven ASF approach from asf_benchmark.py (worked for CrCl6 and RhCl6).
Pin this job to n2cn0203 (has system MKL, proven to work).

Submit: sbatch submit_asf_all.sh
Author: Muhammad Ali Akbar | Dobrautz Group | June 2026
"""
import os, json, glob, csv
import numpy as np
from pyscf import gto, scf
# MUST be set before any pyscf.dmrgscf import (including via asf)
from pyscf import __config__
__config__.dmrgscf_BLOCKEXE = "/pc2/users/h/hpcmual/activeml/scripts/block2main_wrapper.sh"
import asf
# Also set at runtime for safety
from pyscf.dmrgscf import dmrgci
dmrgci.settings.BLOCKEXE = "/pc2/users/h/hpcmual/activeml/scripts/block2main_wrapper.sh"
from pyscf.dmrgscf import dmrgci

SCRATCH_BASE = "/scratch/hpc-prf-qehpc/hpcmual/dmrg_scratch"
DATA_V2      = "/pc2/users/h/hpcmual/activeml/data/new_test_cases_v2"
DATA_REAL    = "/pc2/users/h/hpcmual/activeml/data/real_structure_test"
DATA_CASSCF  = "/pc2/users/h/hpcmual/activeml/data/new_test_cases_casscf"
ECP_METALS   = {"Ru","Rh","Pd","Mo","Os","Ir","Pt","Re","W","Zr","Hf","Tc"}

# ── All test cases ─────────────────────────────────────────────────────
# Format: (label, pub, ml, our_cas, charge, spin, metal, ligand, geom, bond_len, use_ecp)
CASES = [
    ("PtCl4_2minus",  2, 4, 2,  -2, 0, "Pt", "Cl", "sqpl", 2.31, True),
    ("RuCl6_3minus",  5, 4, 1,  -3, 1, "Ru", "Cl", "oct",  2.35, True),
    ("MoCl6_3minus",  3, 4, 3,  -3, 3, "Mo", "Cl", "oct",  2.42, True),
    ("ReCl6_2minus",  3, 4, 3,  -2, 1, "Re", "Cl", "oct",  2.35, True),
    ("CoNH3_6_3plus", 6, 7, 6,  +3, 0, "Co", "N",  "oct",  1.97, False),
    ("REAL_CrCl6",    3, 3, 3,  -3, 3, "Cr", "Cl", "oct",  2.34, False),
    ("REAL_FeCl4",    5, 4, 5,  -1, 5, "Fe", "Cl", "tet",  2.26, False),
    ("REAL_MnCl4",    5, 4, 5,  -2, 5, "Mn", "Cl", "tet",  2.35, False),
    ("REAL_OsCl6",    4, 3, 4,  -2, 2, "Os", "Cl", "oct",  2.38, True),
    ("REAL_FeN6",     6, 7, 6,  +2, 4, "Fe", "N",  "oct",  2.04, False),
]

def oct_atom(m, l, d):
    return (f"{m} 0 0 0\n{l} {d} 0 0\n{l} -{d} 0 0\n"
            f"{l} 0 {d} 0\n{l} 0 -{d} 0\n{l} 0 0 {d}\n{l} 0 0 -{d}")

def tet_atom(m, l, d):
    a = d/np.sqrt(3)
    return (f"{m} 0 0 0\n{l} {a} {a} {a}\n{l} {a} -{a} -{a}\n"
            f"{l} -{a} {a} -{a}\n{l} -{a} -{a} {a}")

def sqpl_atom(m, l, d):
    return f"{m} 0 0 0\n{l} {d} 0 0\n{l} -{d} 0 0\n{l} 0 {d} 0\n{l} 0 -{d} 0"

def nh3_oct(m, d=1.97, dNH=1.012, ang=111.5):
    r = np.radians(ang)
    dirs = [np.array(v, float) for v in
            [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]]
    def perp(nd):
        nd = nd/np.linalg.norm(nd)
        v = np.cross(nd,[1,0,0])
        if np.linalg.norm(v)<1e-6: v = np.cross(nd,[0,1,0])
        v /= np.linalg.norm(v); w = np.cross(nd,v); w /= np.linalg.norm(w)
        return v, w
    lines = [f"{m} 0.000 0.000 0.000"]
    for nd in dirs:
        N = nd*d; lines.append(f"N {N[0]:.4f} {N[1]:.4f} {N[2]:.4f}")
        v,w = perp(nd)
        for phi in [0, 2*np.pi/3, 4*np.pi/3]:
            H = N + dNH*(np.cos(np.pi-r)*(-nd) +
                         np.sin(np.pi-r)*(np.cos(phi)*v+np.sin(phi)*w))
            lines.append(f"H {H[0]:.4f} {H[1]:.4f} {H[2]:.4f}")
    return "\n".join(lines)

def make_atom(metal, ligand, geom, d):
    L = ligand.upper()
    if L in ("N","NH3") or "nh3" in ligand.lower():
        return nh3_oct(metal, d)
    if "sqpl" in geom: return sqpl_atom(metal, L, d)
    if "tet"  in geom: return tet_atom(metal, L, d)
    return oct_atom(metal, L, d)

# ── Main loop ──────────────────────────────────────────────────────────
results = []

for (label, pub, ml, our_cas, charge, spin, metal, ligand, geom, d, use_ecp) in CASES:
    print(f"\n{'='*60}")
    print(f"  {label}  pub={pub}  ml={ml}  cas={our_cas}")
    print(f"{'='*60}")

    scratch = f"{SCRATCH_BASE}/asf_{label}"
    os.makedirs(scratch, exist_ok=True)
    os.environ["PYSCF_TMPDIR"] = scratch
    dmrgci.DMRGCI.runtimeDir = scratch

    try:
        ecp = {metal: "def2-svp"} if use_ecp else {}
        mol = gto.Mole(
            atom    = make_atom(metal, ligand, geom, d),
            charge  = charge,
            spin    = spin,
            basis   = "def2-svp",
            ecp     = ecp,
            verbose = 4,
            output  = f"{scratch}/pyscf.log",
        )
        mol.build()

        mf = scf.UHF(mol)
        mf.max_cycle = 300
        mf.conv_tol  = 1e-10
        mf.kernel()
        print(f"  UHF E={mf.e_tot:.6f}  <S2>={mf.spin_square()[0]:.4f}  conv={mf.converged}")

        # ASF: proven to give correct DMRG entropy-based active space
        result = asf.find_from_scf(
            mf,
            switch_dmrg = 12,
            dmrg_kwargs = {"maxM": 500},
        )

        n_asf   = len(result.to_active_indices())
        e_asf   = result.to_dict().get("energy", 0.0)
        print(f"  ASF n_active = {n_asf}")
        print(f"  ASF energy   = {e_asf:.8f}")
        print(f"  Active MOs:  {result.to_active_indices()}")

        if   n_asf == pub == ml: verdict = f"ALL AGREE → {n_asf}"
        elif n_asf == ml:        verdict = f"CONFIRMS ML ({ml}), pub ({pub}) differs"
        elif n_asf == pub:       verdict = f"CONFIRMS published ({pub}), ML ({ml}) differs"
        else:                    verdict = f"THIRD VALUE ({n_asf}); pub={pub} ML={ml}"

        print(f"  VERDICT: {verdict}")

        results.append(dict(system=label, asf=n_asf, pub=pub, ml=ml,
                            cas=our_cas, verdict=verdict, status="OK"))

        with open(f"{scratch}/result.txt","w") as f:
            f.write(f"system={label}\nasf={n_asf}\npub={pub}\nml={ml}\n"
                    f"cas={our_cas}\nverdict={verdict}\n")

    except Exception as e:
        import traceback; traceback.print_exc()
        results.append(dict(system=label, asf="ERR", pub=pub, ml=ml,
                            cas=our_cas, verdict=str(e)[:80], status="FAILED"))

# ── Summary ────────────────────────────────────────────────────────────
print("\n\n" + "="*65)
print("  COMPLETE ASF BENCHMARK SUMMARY")
print("="*65)
print(f"  {'System':<22} {'Pub':>4} {'ML':>4} {'CAS':>5} {'ASF':>5}  Verdict")
print("-"*65)
for r in results:
    print(f"  {r['system']:<22} {str(r['pub']):>4} {str(r['ml']):>4} "
          f"{str(r['cas']):>5} {str(r['asf']):>5}  {r['verdict']}")
print("="*65)

csv_path = f"{SCRATCH_BASE}/asf_all_summary.csv"
with open(csv_path,"w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader(); w.writerows(results)
print(f"\n  Summary → {csv_path}")
