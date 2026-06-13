"""
Complete HF feature extraction v3 - final version.
All features for Paper 1, Paper 2, and future extensions.
Run: python pyscf_features_v2.py metal ligand n_lig charge spin [dist]
"""
import sys, os, json
import numpy as np
import scipy.linalg as la
from pyscf import gto, scf, mp

SPEC_STRENGTH = {'I':1,'Br':2,'Cl':3,'S':4,'F':5,
                 'O':6,'N':7,'H':8,'P':8,'C':9}
LIG_CHARGE    = {'Cl':-1,'F':-1,'Br':-1,'I':-1,
                 'N':-3,'O':-2,'S':-2,'C':-4,'H':-1}
D_ELECTRONS   = {
    'Fe':{1:7,2:6,3:5,4:4,5:3,6:2},
    'Mn':{1:6,2:5,3:4,4:3,5:2,6:1,7:0},
    'Cr':{1:5,2:4,3:3,4:2,5:1,6:0},
    'Co':{1:8,2:7,3:6,4:5,5:4},
    'Ni':{1:9,2:8,3:7,4:6,5:5},
    'Cu':{1:10,2:9,3:8,4:7},
    'Zn':{2:10},
}

def get_ox_state(metal, charge, n_lig, ligand):
    return charge - n_lig * LIG_CHARGE.get(ligand, 0)

def build_geometry(metal, ligand, n_lig, dist):
    pos = {
        4:[(dist,0,0),(-dist,0,0),(0,dist,0),(0,-dist,0)],
        6:[(dist,0,0),(-dist,0,0),(0,dist,0),
           (0,-dist,0),(0,0,dist),(0,0,-dist)]
    }
    s = f"{metal}  0.000  0.000  0.000\n"
    for p in pos[n_lig]:
        s += f"{ligand}  {p[0]:.3f}  {p[1]:.3f}  {p[2]:.3f}\n"
    return s

# ── ARGUMENTS ─────────────────────────────────────────────────
metal  = sys.argv[1]
ligand = sys.argv[2]
n_lig  = int(sys.argv[3])
charge = int(sys.argv[4])
spin   = int(sys.argv[5])
dist   = float(sys.argv[6]) if len(sys.argv)>6 else 2.18

name    = f"{metal}_{ligand}{n_lig}_chg{charge}_spin{spin}"
outdir  = os.path.expanduser("~/activeml/data/generated")
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, f"{name}.json")

# ── BUILD MOLECULE ────────────────────────────────────────────
mol = gto.Mole()
mol.atom    = build_geometry(metal, ligand, n_lig, dist)
mol.basis   = 'def2-SVP'
mol.charge  = charge
mol.spin    = spin
mol.verbose = 0
mol.build()

ox_state  = get_ox_state(metal, charge, n_lig, ligand)
d_elec    = D_ELECTRONS.get(metal,{}).get(ox_state,-1)
spec_str  = SPEC_STRENGTH.get(ligand, 5)

# ── UHF ───────────────────────────────────────────────────────
mf = scf.UHF(mol)
mf.max_cycle = 500
mf.conv_tol  = 1e-10
mf.verbose   = 0
mf.run()

s_ideal     = spin / 2.0
spin_contam = float(abs(mf.spin_square()[0] -
                        s_ideal*(s_ideal+1)))

# ── ORBITAL ENERGIES ──────────────────────────────────────────
e_a   = mf.mo_energy[0]; e_b   = mf.mo_energy[1]
occ_a = mf.mo_occ[0];    occ_b = mf.mo_occ[1]
mo_a  = mf.mo_coeff[0];  mo_b  = mf.mo_coeff[1]
e_mean    = (e_a + e_b) / 2
occ_total = occ_a + occ_b
n_mo      = len(e_mean)

# ── OVERLAP MATRIX ────────────────────────────────────────────
S_ovlp = mol.intor('int1e_ovlp')
X_orth = la.inv(la.sqrtm(S_ovlp))

# ── DENSITY MATRICES ──────────────────────────────────────────
dm_a, dm_b = mf.make_rdm1()

# ── UNO OCCUPATIONS (molecule-level) ─────────────────────────
dm_tot_orth = X_orth @ (dm_a+dm_b) @ X_orth.T
uno_occ     = np.sort(np.linalg.eigvalsh(dm_tot_orth))[::-1]
n_frac_uno_001 = int(np.sum((uno_occ>0.01)&(uno_occ<1.99)))
n_frac_uno_005 = int(np.sum((uno_occ>0.05)&(uno_occ<1.95)))
n_frac_uno_010 = int(np.sum((uno_occ>0.10)&(uno_occ<1.90)))
n_frac_uno_020 = int(np.sum((uno_occ>0.20)&(uno_occ<1.80)))

# MI entropy from UNO
eps    = 1e-12
n_clip = np.clip(uno_occ/2, eps, 1-eps)
mi_entropy_sum = float(
    -(n_clip*np.log(n_clip) +
      (1-n_clip)*np.log(1-n_clip)).sum())

# ── S_ALPHABETA (Bao-Truhlar) ─────────────────────────────────
n_occ_a = int(occ_a.sum()); n_occ_b = int(occ_b.sum())
n_broken = 0; n_broken_05 = 0
min_overlap = 1.0; mean_overlap = 1.0
if n_occ_a > 0 and n_occ_b > 0:
    try:
        S_ab = mo_a[:,:n_occ_a].T @ S_ovlp @ mo_b[:,:n_occ_b]
        sv   = np.linalg.svd(S_ab, compute_uv=False)
        n_broken    = int(np.sum(sv < 0.9))
        n_broken_05 = int(np.sum(sv < 0.5))
        min_overlap  = float(sv.min())
        mean_overlap = float(sv.mean())
    except: pass

# ── DELTA E HS-LS ─────────────────────────────────────────────
delta_E_HS_LS = 0.0
hs_spin = min(mol.nelectron, 8)
if (hs_spin%2) != (mol.nelectron%2): hs_spin -= 1
if hs_spin > spin:
    try:
        mol_hs = gto.Mole()
        mol_hs.atom    = build_geometry(metal,ligand,n_lig,dist)
        mol_hs.basis   = 'def2-SVP'
        mol_hs.charge  = charge
        mol_hs.spin    = hs_spin
        mol_hs.verbose = 0
        mol_hs.build()
        mf_hs = scf.UHF(mol_hs)
        mf_hs.max_cycle=300; mf_hs.conv_tol=1e-8
        mf_hs.verbose=0; mf_hs.run()
        if mf_hs.converged:
            delta_E_HS_LS = float(mf.e_tot - mf_hs.e_tot)
    except: pass

# ── ALPHA-BETA HOMO GAP ───────────────────────────────────────
homo_a = float(e_a[occ_a>0.5].max()) if (occ_a>0.5).any() else 0.0
homo_b = float(e_b[occ_b>0.5].max()) if (occ_b>0.5).any() else 0.0
homo_alpha_beta_gap = float(abs(homo_a - homo_b))

# ── BROKEN-SYMMETRY FOR SINGLETS (Paper 2 key feature) ────────
n_broken_bs    = 0
min_overlap_bs = 1.0
if spin == 0 and mol.nelectron % 2 == 0:
    try:
        mol_bs = gto.Mole()
        mol_bs.atom    = build_geometry(metal,ligand,n_lig,dist)
        mol_bs.basis   = 'def2-SVP'
        mol_bs.charge  = charge
        mol_bs.spin    = 2
        mol_bs.verbose = 0
        mol_bs.build()
        mf_bs = scf.UHF(mol_bs)
        mf_bs.max_cycle=300; mf_bs.conv_tol=1e-8
        mf_bs.verbose=0; mf_bs.run()
        if mf_bs.converged:
            occ_a_bs = mf_bs.mo_occ[0]
            occ_b_bs = mf_bs.mo_occ[1]
            mo_a_bs  = mf_bs.mo_coeff[0]
            mo_b_bs  = mf_bs.mo_coeff[1]
            n_oa = int(occ_a_bs.sum())
            n_ob = int(occ_b_bs.sum())
            if n_oa>0 and n_ob>0:
                S_bs = (mo_a_bs[:,:n_oa].T @
                        S_ovlp @
                        mo_b_bs[:,:n_ob])
                sv_bs = np.linalg.svd(S_bs, compute_uv=False)
                n_broken_bs    = int(np.sum(sv_bs < 0.9))
                min_overlap_bs = float(sv_bs.min())
    except: pass

# ── MP2 ───────────────────────────────────────────────────────
n_frac_mp2_002 = 0; n_frac_mp2_005 = 0
n_frac_mp2_010 = 0; n_frac_mp2_020 = 0
mp2_corr = 0.0; largest_t2 = 0.0
t2_stored = None
mi_proxy_pairs_01 = 0; mi_proxy_pairs_05 = 0
mi_max_pair = 0.0
# Full MI matrix top pairs (for ADAPT-VQE + DMRG ordering)
top_mi_pairs   = []
mi_matrix_sum  = 0.0
# Per-orbital MP2 occupations (for orbital identification)
per_orb_mp2_occ = []

try:
    mp2_calc = mp.MP2(mf)
    mp2_calc.verbose = 0
    mp2_calc.run()
    mp2_corr = float(mp2_calc.e_corr)

    dm_mp2_raw  = mp2_calc.make_rdm1()
    # UHF MP2 returns (alpha_dm, beta_dm) tuple
    if isinstance(dm_mp2_raw, tuple):
        dm_mp2 = dm_mp2_raw[0] + dm_mp2_raw[1]
    else:
        dm_mp2 = dm_mp2_raw
    dm_mp2_orth = X_orth @ dm_mp2 @ X_orth.T
    mp2_occ     = np.sort(
        np.linalg.eigvalsh(dm_mp2_orth))[::-1]
    n_frac_mp2_002 = int(np.sum((mp2_occ>0.02)&(mp2_occ<1.98)))
    n_frac_mp2_005 = int(np.sum((mp2_occ>0.05)&(mp2_occ<1.95)))
    n_frac_mp2_010 = int(np.sum((mp2_occ>0.10)&(mp2_occ<1.90)))
    n_frac_mp2_020 = int(np.sum((mp2_occ>0.20)&(mp2_occ<1.80)))

    # Per-orbital MP2 occupations for window orbitals
    occ_idx_w  = np.where(occ_total>0.5)[0]
    virt_idx_w = np.where(occ_total<0.5)[0]
    if len(occ_idx_w)>=7 and len(virt_idx_w)>=7:
        window14 = list(occ_idx_w[-7:]) + list(virt_idx_w[:7])
        per_orb_mp2_occ = [
            float(mp2_occ[min(i, len(mp2_occ)-1)])
            for i in range(14)]

    t2_stored = mp2_calc.t2
    if isinstance(t2_stored, tuple):
        largest_t2 = float(max(
            np.abs(t).max() for t in t2_stored
            if t is not None))
    elif t2_stored is not None:
        largest_t2 = float(np.abs(t2_stored).max())

    # Full T2 MI matrix — top 10 correlated pairs
    # Key for ADAPT-VQE operator pool and DMRG ordering
    if isinstance(t2_stored, tuple) and t2_stored[1] is not None:
        t2_ab    = t2_stored[1]
        mi_mat   = np.einsum('ijab,ijab->ij', t2_ab, t2_ab)
        mi_matrix_sum = float(mi_mat.sum())
        n_oa2, n_ob2 = mi_mat.shape
        pairs = [(i,j,float(mi_mat[i,j]))
                 for i in range(n_oa2)
                 for j in range(n_ob2)]
        pairs.sort(key=lambda x:-x[2])
        top_mi_pairs = [[p[0],p[1],p[2]] for p in pairs[:10]]
        mi_proxy_pairs_01 = int(np.sum(mi_mat > 0.01))
        mi_proxy_pairs_05 = int(np.sum(mi_mat > 0.05))
        mi_max_pair = float(mi_mat.max())
except: pass

# ── HOMO-LUMO AND ORBITAL FEATURES ───────────────────────────
occ_mask  = occ_total > 0.5
virt_mask = occ_total < 0.5
homo_e    = float(e_mean[occ_mask].max()) if occ_mask.any() else 0.0
lumo_e    = float(e_mean[virt_mask].min()) if virt_mask.any() else 0.0
gap_cen   = (homo_e + lumo_e) / 2
n_orbs    = len(e_mean)

d_region    = [float(e_mean[i]) for i in range(n_orbs)
               if abs(e_mean[i]-gap_cen) < 2.0]
d_bandwidth = float(max(d_region)-min(d_region)) \
              if len(d_region)>1 else 0.0

# All molecule-level features in one dict
mol_feats = {
    "d_electron_count"    : d_elec,
    "spin_contamination"  : spin_contam,
    "spec_strength"       : spec_str,
    "coord_number"        : n_lig,
    "d_bandwidth"         : d_bandwidth,
    "multiplicity"        : spin+1,
    "n_frac_uno_001"      : n_frac_uno_001,
    "n_frac_uno_005"      : n_frac_uno_005,
    "n_frac_uno_010"      : n_frac_uno_010,
    "n_frac_uno_020"      : n_frac_uno_020,
    "n_broken_symm_09"    : n_broken,
    "n_broken_symm_05"    : n_broken_05,
    "min_sab_overlap"     : min_overlap,
    "mean_sab_overlap"    : mean_overlap,
    "delta_E_HS_LS"       : delta_E_HS_LS,
    "homo_ab_gap"         : homo_alpha_beta_gap,
    "n_frac_mp2_002"      : n_frac_mp2_002,
    "n_frac_mp2_005"      : n_frac_mp2_005,
    "n_frac_mp2_010"      : n_frac_mp2_010,
    "n_frac_mp2_020"      : n_frac_mp2_020,
    "mp2_corr"            : mp2_corr,
    "largest_t2"          : largest_t2,
    "mi_entropy_sum"      : mi_entropy_sum,
    "mi_proxy_pairs_01"   : mi_proxy_pairs_01,
    "mi_proxy_pairs_05"   : mi_proxy_pairs_05,
    "mi_max_pair"         : mi_max_pair,
    # NEW v3 features
    "mi_matrix_sum"       : mi_matrix_sum,
    "top_mi_pairs"        : top_mi_pairs,
    "per_orb_mp2_occ"     : per_orb_mp2_occ,
    "n_broken_bs"         : n_broken_bs,
    "min_overlap_bs"      : min_overlap_bs,
}

# ── BUILD ORBITAL LIST ────────────────────────────────────────
orbitals = []
for i in range(n_orbs):
    e_i   = float(e_mean[i])
    occ_i = float(occ_total[i])
    gaps  = [abs(e_i-float(e_mean[j]))
             for j in range(n_orbs) if j!=i]
    near_gap = float(min(gaps))
    if occ_i > 0.5:
        ct = [abs(e_i-float(e_mean[j]))
              for j in range(n_orbs)
              if j!=i and occ_total[j]<0.5]
    else:
        ct = [abs(e_i-float(e_mean[j]))
              for j in range(n_orbs)
              if j!=i and occ_total[j]>0.5]
    ct_gap   = float(min(ct)) if ct else 999.0
    occ_dev  = float(min(occ_i, 2.0-occ_i))
    n_near   = sum(1 for j in range(n_orbs)
                   if j!=i and abs(e_i-float(e_mean[j]))<0.3)
    orb = {
        "index"           : i,
        "energy"          : e_i,
        "occupation"      : occ_i,
        "uno_occ"         : occ_i,
        "uno_frac_dev"    : occ_dev,
        "dist_homo"       : float(abs(e_i-homo_e)),
        "dist_lumo"       : float(abs(e_i-lumo_e)),
        "dist_gap_center" : float(abs(e_i-gap_cen)),
        "nearest_gap"     : near_gap,
        "ct_gap"          : ct_gap,
        "n_near_orbs"     : n_near,
    }
    orb.update(mol_feats)
    orbitals.append(orb)

# ── SAVE EVERYTHING ───────────────────────────────────────────
record = {
    "name"              : name,
    "metal"             : metal,
    "ligand"            : ligand,
    "n_ligands"         : n_lig,
    "charge"            : charge,
    "spin"              : spin,
    "mult"              : spin+1,
    "dist_ang"          : dist,
    "n_electrons"       : mol.nelectron,
    "E_HF"              : float(mf.e_tot),
    "homo_energy"       : homo_e,
    "lumo_energy"       : lumo_e,
    "homo_lumo_gap"     : lumo_e - homo_e,
    "orbitals"          : orbitals,
}
record.update(mol_feats)

with open(outfile, "w") as f:
    json.dump(record, f, indent=2)

print(f"Saved: {outfile}")
print(f"Features: uno={n_frac_uno_001} broken={n_broken} "
      f"dE={delta_E_HS_LS:.3f} mp2={n_frac_mp2_010} "
      f"mi={mi_entropy_sum:.2f} t2={largest_t2:.4f} "
      f"mi_pairs={len(top_mi_pairs)} "
      f"mp2_orb={len(per_orb_mp2_occ)} "
      f"bs={n_broken_bs}")
