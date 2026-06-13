"""
Fill missing charge states for Br, F, I ligands.
Br needs: -6, 1, 2, 3
F  needs: -6, 1, 2, 3  
I  needs: -6, 1, 2, 3
All have parity checking built in.
"""
import numpy as np, json, os, sys, logging
from pyscf import gto, scf, mcscf

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

exec(open(os.path.expanduser(
    '~/activeml/scripts/gen_balanced.py'
    )).read().split('# ── BUILD SYSTEMS')[0].split(
    'if __name__')[0])

LIG_EQ = {
    'Fe':{'Br':2.35,'F':1.85,'I':2.55},
    'Mn':{'Br':2.50,'F':1.98,'I':2.70},
    'Cr':{'Br':2.47,'F':1.94,'I':2.65},
    'Co':{'Br':2.42,'F':1.90,'I':2.60},
    'Ni':{'Br':2.37,'F':1.86,'I':2.55},
    'Cu':{'Br':2.42,'F':1.91,'I':2.60},
}

# Missing charges per ligand
MISSING = {
    'Br': [-6, 1, 2, 3],
    'F':  [-6, 1, 2, 3],
    'I':  [-6, 1, 2, 3],
}

RAW_SYSTEMS = []

for ligand, charges in MISSING.items():
    for metal in ['Fe','Mn','Cr','Co','Ni','Cu']:
        eq = LIG_EQ[metal][ligand]
        for charge in charges:
            for n_lig in [4, 6]:
                spins = get_valid_spins(
                    metal, charge, n_lig, ligand)
                if not spins:
                    continue
                for frac in [0.95, 1.00, 1.05]:
                    dist = round(eq*frac, 3)
                    RAW_SYSTEMS.append(
                        (metal,charge,n_lig,
                         ligand,dist,spins))

# Build ALL_JOBS — only genuinely new files
gen300   = os.path.expanduser(
    '~/activeml/data/generated300')
existing = set(os.path.basename(f)
               for f in __import__('glob').glob(
                   f'{gen300}/*.json'))

ALL_JOBS = []
seen     = set()
for metal,charge,n_lig,ligand,dist,spins \
        in RAW_SYSTEMS:
    for spin in spins:
        fname = (f'{metal}_{ligand}{n_lig}'
                 f'_chg{charge}_spin{spin}.json')
        if fname in existing:
            continue
        key = (metal,ligand,charge,n_lig,dist,spin)
        if key in seen:
            continue
        seen.add(key)
        ALL_JOBS.append(
            (metal,charge,n_lig,ligand,dist,spin))

if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1]=='summary':
        from collections import defaultdict
        by_lig   = defaultdict(int)
        by_metal = defaultdict(int)
        by_lc    = defaultdict(set)
        for m,c,n,l,d,s in ALL_JOBS:
            by_lig[l]   += 1
            by_metal[m] += 1
            by_lc[l].add(c)
        print(f"Genuinely new jobs: {len(ALL_JOBS)}")
        print("\nBy ligand:")
        for k,v in sorted(by_lig.items()):
            print(f"  {k}: {v}  "
                  f"charges={sorted(by_lc[k])}")
        print("\nBy metal:")
        for k,v in sorted(by_metal.items()):
            print(f"  {k}: {v}")
        sys.exit(0)

    job_idx = int(sys.argv[1]) \
              if len(sys.argv)>1 else 0
    if job_idx >= len(ALL_JOBS):
        sys.exit(1)
    success = run_one(*ALL_JOBS[job_idx])
    sys.exit(0 if success else 1)
