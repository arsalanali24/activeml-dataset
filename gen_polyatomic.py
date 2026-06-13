"""
gen_polyatomic.py — polyatomic ligand CASSCF dataset generator
Ligands: NH3, H2O, CN, PH3
Usage: python gen_polyatomic.py <job_index>
       python gen_polyatomic.py --count
"""
import numpy as np
import json, os, sys, logging, math
from pyscf import gto, scf, mcscf
from pyscf.mcscf import addons as mcaddons

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

OUTDIR = os.path.expanduser('~/activeml/data/generated_polyatomic')
os.makedirs(OUTDIR, exist_ok=True)
BASIS = 'def2-svp'
ECP_METALS = {'Pd', 'Ru', 'Rh', 'Mo', 'Ir', 'Pt'}

METAL_CONSTANTS = {
    'Ti': (8.14,121,'3d'),'V':(8.98,208,'3d'),'Cr':(9.76,273,'3d'),
    'Mn': (10.53,355,'3d'),'Fe':(11.18,460,'3d'),'Co':(12.00,533,'3d'),
    'Ni': (12.78,669,'3d'),'Cu':(13.20,831,'3d'),'Zn':(13.57,1042,'3d'),
    'Pd': (13.00,1334,'4d'),'Ru':(12.33,880,'4d'),'Rh':(12.67,1097,'4d'),
    'Mo': (10.97,467,'4d'),'Ir':(17.00,3909,'5d'),'Pt':(17.33,4146,'5d'),
}

EQ_DIST = {
    ('Fe','NH3'):2.20,('Mn','NH3'):2.25,('Cr','NH3'):2.21,
    ('Co','NH3'):2.17,('Ni','NH3'):2.13,('Cu','NH3'):2.05,
    ('Ti','NH3'):2.28,('V','NH3'):2.22,('Zn','NH3'):2.10,
    ('Pd','NH3'):2.08,('Ru','NH3'):2.15,('Rh','NH3'):2.12,
    ('Mo','NH3'):2.28,('Ir','NH3'):2.10,('Pt','NH3'):2.05,
    ('Fe','H2O'):2.12,('Mn','H2O'):2.18,('Cr','H2O'):1.97,
    ('Co','H2O'):2.09,('Ni','H2O'):2.05,('Cu','H2O'):1.98,
    ('Ti','H2O'):2.07,('V','H2O'):2.00,('Zn','H2O'):2.08,
    ('Pd','H2O'):2.01,('Ru','H2O'):2.12,('Rh','H2O'):2.08,
    ('Mo','H2O'):2.20,('Ir','H2O'):2.05,('Pt','H2O'):2.01,
    ('Fe','CN'):1.92,('Mn','CN'):1.95,('Cr','CN'):2.08,
    ('Co','CN'):1.89,('Ni','CN'):1.86,('Cu','CN'):1.90,
    ('Ti','CN'):2.12,('V','CN'):2.05,('Zn','CN'):1.95,
    ('Pd','CN'):1.98,('Ru','CN'):2.05,('Rh','CN'):2.01,
    ('Mo','CN'):2.15,('Ir','CN'):2.00,('Pt','CN'):1.98,
    ('Fe','PH3'):2.25,('Mn','PH3'):2.32,('Cr','PH3'):2.28,
    ('Co','PH3'):2.22,('Ni','PH3'):2.18,('Cu','PH3'):2.20,
    ('Ti','PH3'):2.48,('V','PH3'):2.38,('Zn','PH3'):2.30,
    ('Pd','PH3'):2.28,('Ru','PH3'):2.35,('Rh','PH3'):2.32,
    ('Mo','PH3'):2.45,('Ir','PH3'):2.30,('Pt','PH3'):2.28,
}

def nh3_atoms(bind_pos, metal_pos, bl=1.012, angle=106.7):
    v = np.array(bind_pos) - np.array(metal_pos)
    v = v / np.linalg.norm(v)
    cone = math.radians(180 - angle)
    perp = np.cross(v, [1,0,0]) if abs(v[0])<0.9 else np.cross(v,[0,1,0])
    perp = perp / np.linalg.norm(perp)
    perp2 = np.cross(v, perp)
    n_pos = np.array(bind_pos)
    atoms = []
    for i in range(3):
        a = 2*math.pi*i/3
        d = math.cos(cone)*v + math.sin(cone)*(math.cos(a)*perp+math.sin(a)*perp2)
        atoms.append(('H', *(n_pos + bl*d)))
    return atoms

def h2o_atoms(bind_pos, metal_pos, bl=0.957, angle=104.5):
    v = np.array(bind_pos) - np.array(metal_pos)
    v = v / np.linalg.norm(v)
    ha = math.radians(angle/2)
    perp = np.cross(v,[1,0,0]) if abs(v[0])<0.9 else np.cross(v,[0,1,0])
    perp = perp / np.linalg.norm(perp)
    o_pos = np.array(bind_pos)
    atoms = []
    for s in [+1,-1]:
        d = math.cos(math.pi-ha)*v + s*math.sin(math.pi-ha)*perp
        d = d / np.linalg.norm(d)
        atoms.append(('H', *(o_pos + bl*d)))
    return atoms

def cn_atoms(bind_pos, metal_pos, cn_bond=1.154):
    v = np.array(bind_pos) - np.array(metal_pos)
    v = v / np.linalg.norm(v)
    return [('N', *(np.array(bind_pos) + cn_bond*v))]

def ph3_atoms(bind_pos, metal_pos, bl=1.415, angle=93.3):
    v = np.array(bind_pos) - np.array(metal_pos)
    v = v / np.linalg.norm(v)
    cone = math.radians(180 - angle)
    perp = np.cross(v,[1,0,0]) if abs(v[0])<0.9 else np.cross(v,[0,1,0])
    perp = perp / np.linalg.norm(perp)
    perp2 = np.cross(v, perp)
    p_pos = np.array(bind_pos)
    atoms = []
    for i in range(3):
        a = 2*math.pi*i/3
        d = math.cos(cone)*v + math.sin(cone)*(math.cos(a)*perp+math.sin(a)*perp2)
        atoms.append(('H', *(p_pos + bl*d)))
    return atoms

LIGANDS = {
    'NH3': {'bind_atom':'N','fn':nh3_atoms},
    'H2O': {'bind_atom':'O','fn':h2o_atoms},
    'CN':  {'bind_atom':'C','fn':cn_atoms},
    'PH3': {'bind_atom':'P','fn':ph3_atoms},
}

def get_positions(n_lig, dist, geom):
    if n_lig==6 or geom=='oct':
        return [(dist,0,0),(-dist,0,0),(0,dist,0),(0,-dist,0),(0,0,dist),(0,0,-dist)]
    elif geom=='sq_pl':
        return [(dist,0,0),(-dist,0,0),(0,dist,0),(0,-dist,0)]
    else:
        s=dist/np.sqrt(3)
        return [(s,s,s),(s,-s,-s),(-s,s,-s),(-s,-s,s)]

def build_mol(metal, lig_name, n_lig, dist, charge, spin, geom):
    info = LIGANDS[lig_name]
    bind = info['bind_atom']
    fn   = info['fn']
    metal_pos = (0.,0.,0.)
    positions = get_positions(n_lig, dist, geom)[:n_lig]
    atoms = [(metal,0.,0.,0.)]
    for p in positions:
        atoms.append((bind,*p))
        atoms.extend(fn(p, metal_pos))
    atom_str = ''.join(f"{a[0]}  {a[1]:.4f}  {a[2]:.4f}  {a[3]:.4f}\n" for a in atoms)
    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = BASIS
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 0
    mol.max_memory = 28000
    if metal in ECP_METALS:
        mol.ecp = BASIS
    mol.build()
    return mol

def get_active_electrons(n):
    for na in [10,9,11,8,12,7,13,6]:
        nc = n - na
        if nc >= 0 and nc % 2 == 0:
            return na
    raise ValueError(f"No valid n_active for n={n}")

def make_systems():
    systems = []
    spin_map = {
        'd0':[0],'d1':[1],'d2':[2,0],'d3':[3,1],'d4':[4,2,0],
        'd5':[5,3,1],'d6':[4,2,0],'d7':[3,1],'d8':[2,0],
        'd9':[1],'d10':[0],
    }
    metal_configs = {
        'Fe':[(-4,'d6'),(-3,'d5'),(-2,'d6')],
        'Mn':[(-4,'d5'),(-3,'d4'),(-2,'d5')],
        'Cr':[(-4,'d4'),(-3,'d3'),(-2,'d4')],
        'Co':[(-4,'d7'),(-3,'d6'),(-2,'d7')],
        'Ni':[(-4,'d8'),(-3,'d7'),(-2,'d8')],
        'Cu':[(-4,'d9'),(-3,'d8'),(-2,'d9')],
        'Ti':[(-4,'d0'),(-3,'d1'),(-2,'d2')],
        'V': [(-4,'d1'),(-3,'d2'),(-2,'d3')],
        'Zn':[(-4,'d10'),(-2,'d10')],
        'Pd':[(-2,'d8'),(-2,'d6')],
        'Ru':[(-3,'d5'),(-2,'d6')],
        'Rh':[(-3,'d6'),(-2,'d7')],
        'Mo':[(-3,'d3'),(-2,'d4')],
        'Ir':[(-3,'d6'),(-2,'d7')],
        'Pt':[(-2,'d8'),(-2,'d6')],
    }
    for lig in ['NH3','H2O','CN','PH3']:
        for metal, ox_states in metal_configs.items():
            if (metal, lig) not in EQ_DIST:
                continue
            eq = EQ_DIST[(metal, lig)]
            for charge, dcfg in ox_states:
                spins = spin_map[dcfg]
                for frac in [0.95, 1.00, 1.05]:
                    dist = round(eq*frac, 3)
                    for spin in spins:
                        systems.append((metal,lig,charge,6,dist,spin,'oct'))
                    if dcfg in ('d7','d8','d9','d10'):
                        for spin in spins:
                            systems.append((metal,lig,charge,4,dist,spin,'tet'))
    return systems

def run_one(idx):
    systems = make_systems()
    if idx >= len(systems):
        return
    metal,lig_name,charge,n_lig,dist,spin,geom = systems[idx]
    dist_str = str(dist).replace('.','p')
    name = f"{metal}_{lig_name}{n_lig}_chg{charge}_spin{spin}_{geom}_d{dist_str}"
    outfile = os.path.join(OUTDIR, f"{name}.json")
    log.info('='*55)
    log.info(f"Starting: {name}")
    if os.path.exists(outfile):
        try:
            d = json.load(open(outfile))
            if d.get('status')=='ok' and d.get('corr_energy',0)<0:
                log.info(f"  SKIP: {name}")
                return
        except: pass
    try:
        mol = build_mol(metal,lig_name,n_lig,dist,charge,spin,geom)
    except Exception as e:
        log.error(f"Build failed: {e}")
        json.dump({'name':name,'status':'build_error','error':str(e)},open(outfile,'w'),indent=2)
        return
    if (mol.nelectron%2) != (spin%2):
        json.dump({'name':name,'status':'skipped','reason':f'parity {mol.nelectron} {spin}'},open(outfile,'w'),indent=2)
        return
    try:
        mf = scf.UHF(mol)
        mf.max_memory=28000; mf.max_cycle=300; mf.conv_tol=1e-8
        mf.kernel()
        if not mf.converged:
            mf.damp=0.3; mf.kernel()
        E_HF = float(mf.e_tot)
        S2,_ = mf.spin_square()
        spin_contam = float(S2 - (spin/2)*(spin/2+1))
        log.info(f"  HF: E={E_HF:.6f} converged={mf.converged}")
    except Exception as e:
        log.error(f"HF failed: {e}")
        json.dump({'name':name,'status':'hf_failed','error':str(e)},open(outfile,'w'),indent=2)
        return
    try:
        n_active_e = get_active_electrons(mol.nelectron)
        mc = mcscf.CASSCF(mf,10,n_active_e)
        mc.max_memory=28000; mc.max_cycle_macro=300
        mc.conv_tol=1e-7; mc.conv_tol_grad=1e-4
        mc.internal_rotation=True
        mc.kernel()
        E_CAS = float(mc.e_tot)
        corr  = E_CAS - E_HF
        noons_all,_ = mcaddons.make_natural_orbitals(mc)
        noons = [float(x) for x in noons_all]
        na = int(np.sum((np.array(noons)>0.02)&(np.array(noons)<1.98)))
        log.info(f"  CASSCF: E={E_CAS:.6f} corr={corr:.6f} n_active={na}")
    except Exception as e:
        log.error(f"CASSCF failed: {e}")
        json.dump({'name':name,'status':'failed','error':str(e)},open(outfile,'w'),indent=2)
        return
    if corr >= 0:
        json.dump({'name':name,'status':'failed','reason':'corr>=0'},open(outfile,'w'),indent=2)
        return
    z_eff,zeta_so,row = METAL_CONSTANTS[metal]
    result = {
        'name':name,'metal':metal,'ligand':lig_name,'ligand_type':'polyatomic',
        'bind_atom':LIGANDS[lig_name]['bind_atom'],'n_ligands':n_lig,
        'charge':charge,'spin':spin,'mult':spin+1,'geometry':geom,
        'dist_ang':dist,'n_electrons':mol.nelectron,'n_active_e':n_active_e,
        'E_HF':E_HF,'E_CASSCF':E_CAS,'corr_energy':corr,
        'converged':bool(mc.converged),'n_active':na,'no_occ':noons,
        'spin_contamination':spin_contam,'z_eff':z_eff,
        'zeta_so_cm1':zeta_so,'metal_row':row,'basis':BASIS,'status':'ok',
    }
    json.dump(result, open(outfile,'w'), indent=2)
    log.info(f"  Saved: {outfile}")

if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--count':
        systems = make_systems()
        print(f"Total jobs: {len(systems)}")
        from collections import Counter
        for k,v in sorted(Counter(s[1] for s in systems).items()):
            print(f"  {k}: {v}")
    else:
        run_one(int(sys.argv[1]) if len(sys.argv)>1 else 0)
