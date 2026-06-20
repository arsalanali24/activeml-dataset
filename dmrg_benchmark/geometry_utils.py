
import math, logging, re
import numpy as np
log = logging.getLogger("geometry_utils")

LIGAND_SYMBOL = {
    "Cl":"Cl","Br":"Br","F":"F","I":"I","H":"H",
    "NH3":"N","H2O":"O","OH":"O","SH":"S","SCN":"S","NCS":"N",
    "CO":"C","CN":"C","NO":"N","PH3":"P","PPh3":"P",
    "N":"N","O":"O","S":"S","P":"P","C":"C",
    "en":"N","bipy":"N","acac":"O","ox":"O","py":"N","phen":"N",
}
DEFAULT_DIST = {
    ("Fe","Cl"):2.29,("Fe","Br"):2.45,("Fe","F"):1.99,("Fe","I"):2.65,
    ("Fe","H"):1.60,("Fe","C"):1.82,("Fe","N"):2.20,("Fe","O"):2.10,
    ("Fe","S"):2.35,("Fe","P"):2.22,
    ("Co","Cl"):2.27,("Co","Br"):2.43,("Co","F"):1.95,("Co","I"):2.61,
    ("Co","H"):1.55,("Co","C"):1.82,("Co","N"):1.96,("Co","O"):2.08,
    ("Co","S"):2.30,("Co","P"):2.20,
    ("Mn","Cl"):2.34,("Mn","Br"):2.50,("Mn","F"):2.02,("Mn","I"):2.70,
    ("Mn","H"):1.68,("Mn","C"):1.85,("Mn","N"):2.27,("Mn","O"):2.20,
    ("Mn","S"):2.40,("Mn","P"):2.28,
    ("Cr","Cl"):2.35,("Cr","Br"):2.50,("Cr","F"):2.00,("Cr","I"):2.68,
    ("Cr","H"):1.73,("Cr","C"):1.86,("Cr","N"):2.11,("Cr","O"):1.99,
    ("Cr","S"):2.38,("Cr","P"):2.25,
    ("Ni","Cl"):2.38,("Ni","Br"):2.53,("Ni","F"):2.03,("Ni","I"):2.68,
    ("Ni","H"):1.56,("Ni","C"):1.84,("Ni","N"):2.15,("Ni","O"):2.05,
    ("Ni","S"):2.32,("Ni","P"):2.18,
    ("Cu","Cl"):2.26,("Cu","Br"):2.40,("Cu","F"):1.93,("Cu","I"):2.60,
    ("Cu","H"):1.57,("Cu","C"):1.85,("Cu","N"):2.03,("Cu","O"):1.97,
    ("Cu","S"):2.27,("Cu","P"):2.15,
    ("Ru","Cl"):2.38,("Rh","Cl"):2.35,("Pd","Cl"):2.30,
    ("Mo","Cl"):2.49,("Tc","Cl"):2.42,
    ("Re","Cl"):2.37,("Os","Cl"):2.38,("Ir","Cl"):2.37,
    ("Pt","Cl"):2.33,("W","Cl"):2.45,
}
FALLBACK_DIST = 2.20

def _ligand_sym(s):
    if s in LIGAND_SYMBOL: return LIGAND_SYMBOL[s]
    import re
    m = re.match(r"([A-Z][a-z]?)", s)
    return m.group(1) if m else "N"

def _bond_dist(metal, lig_sym, dist_ang):
    if dist_ang is not None:
        try: return float(dist_ang)
        except: pass
    return DEFAULT_DIST.get((metal, lig_sym), FALLBACK_DIST)

def _perp_pair(u):
    ref = np.array([1,0,0.]) if abs(u[0])<0.9 else np.array([0,1,0.])
    v1 = np.cross(u,ref); v1 /= np.linalg.norm(v1)
    v2 = np.cross(u,v1);  v2 /= np.linalg.norm(v2)
    return v1, v2

def _nh3_h_positions(n_pos, u_away):
    r=1.012; ax=r*math.cos(math.radians(67.8)); pr=r*math.sin(math.radians(67.8))
    v1,v2=_perp_pair(u_away)
    return [n_pos+ax*u_away+pr*(math.cos(2*math.pi*k/3)*v1+math.sin(2*math.pi*k/3)*v2)
            for k in range(3)]

def _h2o_h_positions(o_pos, u_away):
    r=0.957; ax=r*0.5; pr=r*math.sqrt(3)/2
    v1,_=_perp_pair(u_away)
    return [o_pos+ax*u_away+s*pr*v1 for s in [+1,-1]]

def _oct_uvecs():
    return [np.array(v,dtype=float) for v in
            [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]]

def build_atom_string(metal, ligand_name, n_lig, geom_type, dist):
    g=str(geom_type).lower()
    if   n_lig==2: uvecs=[np.array([1,0,0.]),np.array([-1,0,0.])]
    elif n_lig==4: uvecs=[np.array([1,0,0.]),np.array([-1,0,0.]),
                          np.array([0,1,0.]),np.array([0,-1,0.])]
    elif n_lig==5:
        if "tbp" in g or "trigonal" in g:
            uvecs=[np.array([1,0,0.]),np.array([-0.5,math.sqrt(3)/2,0.]),
                   np.array([-0.5,-math.sqrt(3)/2,0.]),
                   np.array([0,0,1.]),np.array([0,0,-1.])]
        else:
            uvecs=[np.array([1,0,0.]),np.array([-1,0,0.]),
                   np.array([0,1,0.]),np.array([0,-1,0.]),np.array([0,0,1.])]
    else: uvecs=_oct_uvecs()
    coord_sym=_ligand_sym(ligand_name)
    lines=[f"{metal}  0.0000  0.0000  0.0000"]
    for u in uvecs:
        p=u*dist
        lines.append(f"{coord_sym}  {p[0]:.4f}  {p[1]:.4f}  {p[2]:.4f}")
        if ligand_name=="NH3":
            for h in _nh3_h_positions(p,u):
                lines.append(f"H  {h[0]:.4f}  {h[1]:.4f}  {h[2]:.4f}")
        elif ligand_name=="H2O":
            for h in _h2o_h_positions(p,u):
                lines.append(f"H  {h[0]:.4f}  {h[1]:.4f}  {h[2]:.4f}")
        elif ligand_name=="CO":
            o=p+u*1.128
            lines.append(f"O  {o[0]:.4f}  {o[1]:.4f}  {o[2]:.4f}")
    return "\n".join(lines)

def rebuild_mol_from_params(system: dict):
    from pyscf import gto
    atom_str=system.get("atom","")
    if atom_str and "\n" in str(atom_str):
        mol=gto.Mole()
        mol.atom=atom_str; mol.charge=int(system["charge"])
        mol.spin=int(system["spin"]); mol.basis=system.get("basis","def2-svp")
        mol.ecp=mol.basis; mol.verbose=3; mol.build()
        log.info("Geometry: from JSON atom field (%d atoms)",mol.natm)
        return mol
    metal=system["metal"]; ligand=system.get("ligand","Cl")
    n_lig=int(system.get("n_ligands",6)); charge=int(system["charge"])
    spin=int(system["spin"]); dist_ang=system.get("dist_ang")
    geom=system.get("geometry","oct"); basis=system.get("basis","def2-svp")
    lig_sym=_ligand_sym(ligand); dist=_bond_dist(metal,lig_sym,dist_ang)
    atom_str=build_atom_string(metal,ligand,n_lig,geom,dist)
    log.info("Geometry rebuilt: %s[%s]%d_%s d=%.3f",metal,ligand,n_lig,geom,dist)
    mol=gto.Mole()
    mol.atom=atom_str; mol.charge=charge; mol.spin=spin
    mol.basis=basis; mol.ecp=mol.basis; mol.verbose=3; mol.build()
    return mol

def get_mp2_natural_orbitals(mol, mf):
    from pyscf import mp
    log.info("Running MP2 for natural orbitals ...")
    mp2=mp.UMP2(mf); mp2.verbose=0; mp2.kernel()
    dm1_a,dm1_b=mp2.make_rdm1()
    dm1=dm1_a+dm1_b
    occ,U=np.linalg.eigh(dm1)
    idx=np.argsort(-occ); occ=occ[idx]; U=U[:,idx]
    no_coeff=mf.mo_coeff[0]@U
    frac=int(np.sum((occ>0.02)&(occ<1.98)))
    log.info("MP2 NOs: frac=%d, occ range [%.4f,%.4f]",frac,occ[-1],occ[0])
    return no_coeff, occ

def select_window_mp2no(no_occ, n_window=20):
    score=np.abs(no_occ-1.0)
    ranked=np.argsort(score)
    win_idx=sorted(ranked[:n_window].tolist())
    log.info("MP2 NO window: %d orbitals",len(win_idx))
    return win_idx

def get_integrals_mp2no(mol, mf, no_coeff, win_idx, no_occ=None):
    """MP2 NO window integrals. no_occ used for correct electron counting."""
    from pyscf import ao2mo
    n_orb   = len(win_idx)
    nao     = mol.nao
    nmo     = no_coeff.shape[1]
    win_set = set(win_idx)
    mo_win  = no_coeff[:, win_idx]

    # ALWAYS compute alpha/beta DMs — needed for core Fock regardless of no_occ
    dm1_a_ao = (mf.mo_coeff[0] * mf.mo_occ[0]) @ mf.mo_coeff[0].T
    dm1_b_ao = (mf.mo_coeff[1] * mf.mo_occ[1]) @ mf.mo_coeff[1].T

    # Electron counting in window
    if no_occ is not None:
        # Use MP2 NO occupations: sum of NO occ in window / 2 ± spin/2
        n_tot   = float(np.real(sum(float(no_occ[i]) for i in win_idx)))
        n_alpha = max(0, round((n_tot + mol.spin) / 2))
        n_beta  = max(0, round((n_tot - mol.spin) / 2))
    else:
        _pa = mo_win.T @ dm1_a_ao @ mo_win
        _pb = mo_win.T @ dm1_b_ao @ mo_win
        n_alpha = max(0, round(float(np.real(np.einsum("ii->", _pa)))))
        n_beta  = max(0, round(float(np.real(np.einsum("ii->", _pb)))))

    # Core contribution to Fock
    core_idx = [i for i in range(nmo) if i not in win_set]
    mo_core  = no_coeff[:, core_idx]
    hcore    = mf.get_hcore()
    if core_idx:
        dm_ca = mo_core @ (mo_core.T @ dm1_a_ao @ mo_core) @ mo_core.T
        dm_cb = mo_core @ (mo_core.T @ dm1_b_ao @ mo_core) @ mo_core.T
        vhf   = mf.get_veff(mol, np.array([dm_ca, dm_cb]))
        vhf_sp = 0.5 * (np.array(vhf[0]) + np.array(vhf[1]))
        ecore  = float(mol.energy_nuc())
        ecore += float(np.einsum("ij,ij->", hcore, dm_ca + dm_cb))
        ecore += 0.5 * float(np.einsum("ij,ij->", vhf[0], dm_ca)
                              + np.einsum("ij,ij->", vhf[1], dm_cb))
    else:
        vhf_sp = np.zeros((nao, nao))
        ecore  = float(mol.energy_nuc())

    h1e = mo_win.T @ (hcore + vhf_sp) @ mo_win
    g2e = ao2mo.kernel(mol, mo_win, compact=False).reshape(n_orb, n_orb, n_orb, n_orb)
    log.info("MP2NO integrals: na=%d nb=%d ecore=%.4f", n_alpha, n_beta, ecore)
    return h1e, g2e, ecore, n_alpha, n_beta

def get_integrals_uhf(mol, mf, win_idx: list) -> tuple:
    from pyscf import ao2mo
    n_orb=len(win_idx); nao=mol.nao; nmo=len(mf.mo_occ[0])
    win_set=set(win_idx); mo_occ_a=mf.mo_occ[0]; mo_occ_b=mf.mo_occ[1]
    n_alpha=int(sum(1 for i in win_idx if mo_occ_a[i]>0.5))
    n_beta =int(sum(1 for i in win_idx if mo_occ_b[i]>0.5))
    mo_win=mf.mo_coeff[0][:,win_idx]
    core_a=[i for i in range(nmo) if mo_occ_a[i]>0.5 and i not in win_set]
    core_b=[i for i in range(nmo) if mo_occ_b[i]>0.5 and i not in win_set]
    z=np.zeros((nao,nao))
    dm_ca=(mf.mo_coeff[0][:,core_a]@mf.mo_coeff[0][:,core_a].T) if core_a else z.copy()
    dm_cb=(mf.mo_coeff[1][:,core_b]@mf.mo_coeff[1][:,core_b].T) if core_b else z.copy()
    hcore=mf.get_hcore()
    if core_a or core_b:
        vhf=mf.get_veff(mol,np.array([dm_ca,dm_cb]))
        vhf_sp=0.5*(np.array(vhf[0])+np.array(vhf[1]))
        ecore=float(mol.energy_nuc())
        ecore+=float(np.einsum("ij,ij->",hcore,dm_ca+dm_cb))
        ecore+=0.5*float(np.einsum("ij,ij->",vhf[0],dm_ca)+np.einsum("ij,ij->",vhf[1],dm_cb))
    else:
        vhf_sp=z.copy(); ecore=float(mol.energy_nuc())
    h1e=mo_win.T@(hcore+vhf_sp)@mo_win
    g2e=ao2mo.kernel(mol,mo_win,compact=False).reshape(n_orb,n_orb,n_orb,n_orb)
    log.info("UHF integrals: na=%d nb=%d ecore=%.4f",n_alpha,n_beta,ecore)
    return h1e,g2e,ecore,n_alpha,n_beta
