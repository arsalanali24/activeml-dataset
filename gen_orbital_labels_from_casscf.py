"""
gen_orbital_labels_from_casscf.py
Generates correct is_active labels for orbital identification
directly from existing CASSCF no_occ data.
No new HPC computation needed.

Output: ~/activeml/data/orbital_labels/ — one JSON per system
Format compatible with orbital_features_v2 but with correct labels.
"""
import json, glob, os, sys
import numpy as np

OUTDIR = os.path.expanduser('~/activeml/data/orbital_labels')
os.makedirs(OUTDIR, exist_ok=True)

def compute_labels(filepath):
    try:
        d = json.load(open(filepath))
    except:
        return False

    if d.get('status') != 'ok': return True
    no_occ   = d.get('no_occ', [])
    n_active = d.get('n_active', 0)
    if not no_occ or n_active == 0: return True

    name    = d['name']
    outfile = os.path.join(OUTDIR, f'{name}.json')

    if os.path.exists(outfile):
        return True

    # Compute noon_frac for each orbital
    noon_fracs = [float(min(n, 2.0 - n)) for n in no_occ]

    # Select top n_active orbitals by noon_frac
    ranked = sorted(range(len(noon_fracs)),
                    key=lambda i: noon_fracs[i], reverse=True)
    active_set = set(ranked[:n_active])

    # Build per-orbital records
    orbitals = []
    for i, (noon, frac) in enumerate(zip(no_occ, noon_fracs)):
        orbitals.append({
            'window_pos':  i,
            'noon':        float(noon),
            'noon_frac':   float(frac),
            's_i':         float(-min(noon/2,1-1e-10)*np.log(min(noon/2,1-1e-10))
                                 -(1-min(noon/2,1-1e-10))*np.log(1-min(noon/2,1-1e-10))),
            'is_active':   1 if i in active_set else 0,
        })

    result = {
        'name':       name,
        'status':     'done',
        'metal':      d.get('metal', ''),
        'ligand':     d.get('ligand', ''),
        'charge':     d.get('charge', 0),
        'spin':       d.get('spin', 0),
        'n_active':   n_active,
        'n_active_e': d.get('n_active_e', 0),
        'n_window':   len(no_occ),
        'source':     'casscf_no_occ',
        'orbitals':   orbitals,
    }
    json.dump(result, open(outfile, 'w'), indent=2)
    return True

# Process all datasets
total = done = 0
for folder in ['~/activeml/data/generated300',
               '~/activeml/data/generated_4d5d',
               '~/activeml/data/generated_polyatomic']:
    folder = os.path.expanduser(folder)
    files  = sorted(glob.glob(f'{folder}/*.json'))
    for f in files:
        total += 1
        if compute_labels(f):
            done += 1

print(f'Processed: {done}/{total}')
print(f'Output: {OUTDIR}')
print(f'Files: {len(glob.glob(OUTDIR+"/*.json"))}')
