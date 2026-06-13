"""
fix_irc_snapshots.py — regenerates IRC snapshots with correct interpolation.
Product side: all atoms interpolate TS->product EXCEPT Cl which goes
directly to product position (avoids Cl passing through Pd).
"""
import numpy as np, json, os, sys, logging
from pyscf import gto, dft

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)
IRC_DIR = os.path.expanduser('~/activeml/data/irc_pd')

def parse(s):
    a=[]
    for l in s.strip().split('\n'):
        p=l.strip().split()
        if len(p)==4: a.append((p[0],float(p[1]),float(p[2]),float(p[3])))
    return a

def atom_str(syms,coords):
    return '\n'.join(f"{s}  {c[0]:.6f}  {c[1]:.6f}  {c[2]:.6f}"
                     for s,c in zip(syms,coords))

def dft_energy(syms,coords):
    mol=gto.Mole(); mol.atom=atom_str(syms,coords)
    mol.basis='def2-SVP'; mol.ecp={'Pd':'def2-SVP'}
    mol.charge=0; mol.spin=0; mol.verbose=0; mol.build()
    mf=dft.RKS(mol); mf.xc='PBE0'; mf.max_cycle=200
    mf.conv_tol=1e-8; mf.verbose=0; mf.kernel()
    return float(mf.e_tot) if mf.converged else None

def save_xyz(syms,coords,idx,label,E):
    fname=os.path.join(IRC_DIR,f'snapshot_{idx:02d}.xyz')
    with open(fname,'w') as f:
        f.write(f"{len(syms)}\n{label} E={E:.6f}\n")
        for s,c in zip(syms,coords):
            f.write(f"{s:2s}  {c[0]:10.6f}  {c[1]:10.6f}  {c[2]:10.6f}\n")
    return fname

log.info("Loading optimized geometries...")
syms = [a[0] for a in parse(json.load(open(f'{IRC_DIR}/reactant.json'))['atom_str'])]
r  = np.array([[a[1],a[2],a[3]] for a in parse(json.load(open(f'{IRC_DIR}/reactant.json'))['atom_str'])])
ts = np.array([[a[1],a[2],a[3]] for a in parse(json.load(open(f'{IRC_DIR}/ts_guess.json'))['atom_str'])])
p  = np.array([[a[1],a[2],a[3]] for a in parse(json.load(open(f'{IRC_DIR}/product.json'))['atom_str'])])

pd_i=syms.index('Pd')
c_i =next(j for j,s in enumerate(syms) if s=='C')
cl_i=next(j for j,s in enumerate(syms) if s=='Cl')

# Build 12 snapshots
snaps=[]
# Reactant->TS: 6 points (t=0.0 to 1.0), full linear interpolation
for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    c=r+t*(ts-r)
    phase='transition_state' if t==1.0 else 'reactant_side'
    snaps.append((c, f'reactant_to_TS_t{t:.1f}', phase))

# TS->product: 6 points, Cl fixed at product position
for t in [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]:
    c=ts.copy()
    for j in range(len(syms)):
        if j != cl_i:
            c[j]=ts[j]+t*(p[j]-ts[j])
    c[cl_i]=p[cl_i].copy()  # Cl at product position
    snaps.append((c, f'TS_to_product_t{t:.1f}', 'product_side'))

assert len(snaps)==12
log.info("Geometry validation:")
all_ok=True
for i,(coords,label,phase) in enumerate(snaps):
    pd_c =np.linalg.norm(coords[pd_i]-coords[c_i])
    pd_cl=np.linalg.norm(coords[pd_i]-coords[cl_i])
    c_cl =np.linalg.norm(coords[c_i] -coords[cl_i])
    ok=pd_c>1.8 and pd_cl>1.8 and c_cl>1.4
    flag="OK" if ok else "*** BAD"
    log.info(f"  {i:2d}: Pd-C={pd_c:.3f} Pd-Cl={pd_cl:.3f} C-Cl={c_cl:.3f} {flag}")
    if not ok: all_ok=False

if not all_ok:
    log.error("Unphysical geometries — aborting"); sys.exit(1)

log.info("\nComputing DFT energies for 12 snapshots...")
irc_data=[]
for i,(coords,label,phase) in enumerate(snaps):
    log.info(f"  Snapshot {i:02d}: {label}")
    E=dft_energy(syms,coords)
    save_xyz(syms,coords,i,label,E or 0.0)
    pd_c =float(np.linalg.norm(coords[pd_i]-coords[c_i]))
    pd_cl=float(np.linalg.norm(coords[pd_i]-coords[cl_i]))
    c_cl =float(np.linalg.norm(coords[c_i] -coords[cl_i]))
    irc_data.append({'irc_index':i,'label':label,'phase':phase,
                     'energy':E,'pd_c_dist':pd_c,'pd_cl_dist':pd_cl,
                     'c_cl_dist':c_cl,'xyz_file':f'snapshot_{i:02d}.xyz',
                     'atom_str':atom_str(syms,coords)})
    log.info(f"    E={E:.4f} Pd-C={pd_c:.3f} Pd-Cl={pd_cl:.3f} C-Cl={c_cl:.3f}")

summary=json.load(open(f'{IRC_DIR}/irc_summary.json'))
summary['snapshots']=irc_data
summary['n_snapshots']=12
summary['note']='Fixed interpolation: Cl placed at product position for product side'
with open(f'{IRC_DIR}/irc_summary.json','w') as f: json.dump(summary,f,indent=2)
log.info(f"\nDone! 12 valid snapshots saved to {IRC_DIR}")
log.info("Next: run gen_irc_casscf.py")
