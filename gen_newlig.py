"""
New ligand expansion — ONLY systems not yet generated.
Focuses on Br, F, I with new charge states and
bond lengths outside the previously tried range.
"""
import numpy as np, json, os, sys, logging
from pyscf import gto, scf, mcscf

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# Copy helper functions from gen_balanced.py
exec(open(os.path.expanduser(
    '~/activeml/scripts/gen_balanced.py'
    )).read().split('# ── BUILD SYSTEMS')[0].split(
    'if __name__')[0])

# ── NEW SYSTEMS: expand bond length range ─────────────────────
# Previous scripts used 0.90-1.10 × eq
# Now use 0.85-1.15 × eq AND new charge states
# This guarantees new filenames

def bls_new(eq):
    """Bond lengths NOT in previous batches."""
    return [round(eq*f, 3) for f in
            [0.85, 0.88, 0.92, 1.08, 1.12, 1.15]]

LIG_EQ = {
    'Fe':{'Br':2.35,'F':1.85,'I':2.55},
    'Mn':{'Br':2.50,'F':1.98,'I':2.70},
    'Cr':{'Br':2.47,'F':1.94,'I':2.65},
    'Co':{'Br':2.42,'F':1.90,'I':2.60},
    'Ni':{'Br':2.37,'F':1.86,'I':2.55},
    'Cu':{'Br':2.42,'F':1.91,'I':2.60},
}

RAW_SYSTEMS = []

for metal in ['Fe','Mn','Cr','Co','Ni','Cu']:
    for ligand in ['Br','F','I']:
        eq = LIG_EQ[metal][ligand]
        # New bond lengths only
        for dist in bls_new(eq):
            for charge in [-1,-2,-3,-4,-5]:
                spins = get_valid_spins(
                    metal, charge, 4, ligand)
                if spins:
                    RAW_SYSTEMS.append(
                        (metal,charge,4,ligand,
                         dist,spins))
                spins6 = get_valid_spins(
                    metal, charge, 6, ligand)
                if spins6:
                    RAW_SYSTEMS.append(
                        (metal,charge,6,ligand,
                         dist,spins6))
        # Cu needs more charge states
        if metal == 'Cu':
            for dist in bls_new(eq):
                for charge in [-6,-7,0,1,2]:
                    spins = get_valid_spins(
                        metal, charge, 4, ligand)
                    if spins:
                        RAW_SYSTEMS.append(
                            (metal,charge,4,ligand,
                             dist,spins))

# Build ALL_JOBS with deduplication
ALL_JOBS = []
seen     = set()
for metal,charge,n_lig,ligand,dist,spins \
        in RAW_SYSTEMS:
    for spin in spins:
        key = (metal,ligand,charge,n_lig,dist,spin)
        if key in seen:
            continue
        seen.add(key)
        ALL_JOBS.append(
            (metal,charge,n_lig,ligand,dist,spin))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        from collections import defaultdict
        by_lig   = defaultdict(int)
        by_metal = defaultdict(int)

        # Check against existing files
        gen300 = os.path.expanduser(
            '~/activeml/data/generated300')
        existing = set(os.path.basename(f) for f in
            __import__('glob').glob(f'{gen300}/*.json'))

        new = 0
        for m,c,n,l,d,s in ALL_JOBS:
            fname = f'{m}_{l}{n}_chg{c}_spin{s}.json'
            if fname not in existing:
                by_lig[l]   += 1
                by_metal[m] += 1
                new += 1

        print(f"Total jobs:    {len(ALL_JOBS)}")
        print(f"Genuinely new: {new}")
        print("\nNew by ligand:")
        for k,v in sorted(by_lig.items()):
            print(f"  {k}: {v}")
        print("New by metal:")
        for k,v in sorted(by_metal.items()):
            print(f"  {k}: {v}")
        sys.exit(0)

    job_idx = int(sys.argv[1]) if len(sys.argv)>1 else 0
    total   = len(ALL_JOBS)
    print(f"Total jobs: {total}")
    if job_idx >= total:
        sys.exit(1)
    success = run_one(*ALL_JOBS[job_idx])
    sys.exit(0 if success else 1)
