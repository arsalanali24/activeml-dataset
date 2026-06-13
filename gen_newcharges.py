"""
New charge states never previously generated.
Previous scripts used charges -1 to -6.
This script uses:
  - Positive charges (+1, +2, +3)  
  - Very high negative charges (-7, -8)
  - Also retries failed Br/F/I with robust convergence
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
    'Fe':{'Cl':2.18,'Br':2.35,'F':1.85,'I':2.55},
    'Mn':{'Cl':2.35,'Br':2.50,'F':1.98,'I':2.70},
    'Cr':{'Cl':2.31,'Br':2.47,'F':1.94,'I':2.65},
    'Co':{'Cl':2.26,'Br':2.42,'F':1.90,'I':2.60},
    'Ni':{'Cl':2.21,'Br':2.37,'F':1.86,'I':2.55},
    'Cu':{'Cl':2.26,'Br':2.42,'F':1.91,'I':2.60},
}

RAW_SYSTEMS = []

for ligand in ['Cl','Br','F','I']:
    for metal in ['Fe','Mn','Cr','Co','Ni','Cu']:
        eq = LIG_EQ[metal][ligand]

        # NEW: positive charges (+1, +2, +3)
        # These were NEVER tried before
        for charge in [1, 2, 3]:
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

        # NEW: very high negative charges (-7, -8)
        # These were NEVER tried before
        for charge in [-7, -8]:
            for n_lig in [6]:  # only octahedral
                spins = get_valid_spins(
                    metal, charge, n_lig, ligand)
                if not spins:
                    continue
                for frac in [1.00, 1.05]:
                    dist = round(eq*frac, 3)
                    RAW_SYSTEMS.append(
                        (metal,charge,n_lig,
                         ligand,dist,spins))

        # Cu specifically: add 5-coordinate
        # trigonal bipyramidal approximated as
        # 4+1 coordination
        if metal == 'Cu':
            for charge in [-1,-2,-3,0,1]:
                spins = get_valid_spins(
                    metal, charge, 4, ligand)
                if not spins:
                    continue
                # Slightly longer bond = 5-coord proxy
                for frac in [1.10, 1.15, 1.20]:
                    dist = round(eq*frac, 3)
                    RAW_SYSTEMS.append(
                        (metal,charge,4,
                         ligand,dist,spins))

# Build ALL_JOBS with deduplication vs existing
ALL_JOBS = []
seen     = set()

gen300   = os.path.expanduser(
    '~/activeml/data/generated300')
existing = set(os.path.basename(f)
               for f in __import__('glob').glob(
                   f'{gen300}/*.json'))

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
        for m,c,n,l,d,s in ALL_JOBS:
            by_lig[l]   += 1
            by_metal[m] += 1
        print(f"Genuinely new jobs: {len(ALL_JOBS)}")
        print("\nBy ligand:")
        for k,v in sorted(by_lig.items()):
            print(f"  {k}: {v}")
        print("By metal:")
        for k,v in sorted(by_metal.items()):
            print(f"  {k}: {v}")
        sys.exit(0)

    job_idx = int(sys.argv[1]) if len(sys.argv)>1 else 0
    if job_idx >= len(ALL_JOBS): sys.exit(1)
    success = run_one(*ALL_JOBS[job_idx])
    sys.exit(0 if success else 1)
