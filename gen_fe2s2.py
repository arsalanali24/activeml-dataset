"""
Generate Fe2S2 artificial dataset.
~1000 systems: [Fe2S2(L)4]^n-
Varying: terminal ligand, charge, spin,
         Fe-Fe distance, bridge angle.
Outputs CASSCF-style JSON for each system
(geometry + metadata, NO quantum chem yet)
Run locally to generate input list,
then submit UHF+DMRG as array jobs.
"""
import json, os, itertools, math
import numpy as np

outdir = os.path.expanduser(
    '~/activeml/data/fe2s2_inputs')
os.makedirs(outdir, exist_ok=True)

# ── PARAMETERS ────────────────────────────────────────────────

TERMINALS = {
    'Cl' : 2.28,   # Fe-Cl distance (Ang)
    'SH' : 2.32,   # Fe-S(terminal)
    'OH' : 2.10,   # Fe-O
    'F'  : 1.95,   # Fe-F
    'CN' : 2.00,   # Fe-C(CN)
}

# Fe-Fe distances to sample
FE_FE_DISTS = [2.55, 2.69, 2.83, 2.97]

# Fe-S(bridge) distance (fixed ratio to Fe-Fe)
# Fe-S(bridge) ≈ Fe-Fe * 0.82
BRIDGE_RATIO = 0.82

# Charge → spin states mapping
CHARGE_SPINS = {
    -1: [0, 2, 4, 6, 8, 10],   # 2S values (spin kwarg)
    -2: [0, 2, 4, 6, 8, 10],
    -3: [0, 2, 4, 6, 8],       # mixed valence
    -4: [0, 2, 4, 6, 8],
}

# ── GEOMETRY BUILDER ──────────────────────────────────────────
def build_fe2s2_geometry(terminal, fe_fe, charge,
                          distort=0.0):
    """
    Build Fe2S2(L)4 geometry.
    Fe2S2 rhombus in xz plane.
    Terminal ligands along y axis.

    Fe1 at (0, +fe_fe/2, 0)
    Fe2 at (0, -fe_fe/2, 0)
    S1  at (+fe_s, 0, 0)
    S2  at (-fe_s, 0, 0)
    L on Fe1: (0, +fe_fe/2+d_L, 0) and
              (0, +fe_fe/2,     d_L)
    L on Fe2: (0, -fe_fe/2-d_L, 0) and
              (0, -fe_fe/2,    -d_L)
    """
    fe_s   = fe_fe * BRIDGE_RATIO
    d_L    = TERMINALS[terminal]

    # Add small distortion for variety
    rng    = np.random.default_rng(
        int(fe_fe*100) + abs(charge)*10)
    noise  = rng.uniform(-distort, distort, (12,3))

    Fe1 = np.array([0,  fe_fe/2, 0])
    Fe2 = np.array([0, -fe_fe/2, 0])
    S1  = np.array([ fe_s, 0, 0])
    S2  = np.array([-fe_s, 0, 0])

    # Terminal ligands — 2 per Fe
    if terminal in ['Cl', 'F', 'OH', 'SH']:
        atom_sym = terminal[0] if terminal != 'OH' \
                   else 'O'
        if terminal == 'SH':
            atom_sym = 'S'

        # Fe1 terminals
        L1a = Fe1 + np.array([0,  d_L, 0])
        L1b = Fe1 + np.array([0, 0,  d_L])
        # Fe2 terminals
        L2a = Fe2 + np.array([0, -d_L, 0])
        L2b = Fe2 + np.array([0, 0, -d_L])

        atoms = [
            ('Fe', Fe1), ('Fe', Fe2),
            ('S',  S1),  ('S',  S2),
            (atom_sym, L1a), (atom_sym, L1b),
            (atom_sym, L2a), (atom_sym, L2b),
        ]

        # Add H for SH and OH
        if terminal in ['SH', 'OH']:
            H1a = L1a + np.array([0,  0.97, 0])
            H1b = L1b + np.array([0, 0, 0.97])
            H2a = L2a + np.array([0, -0.97, 0])
            H2b = L2b + np.array([0, 0, -0.97])
            atoms += [('H',H1a),('H',H1b),
                      ('H',H2a),('H',H2b)]

    elif terminal == 'CN':
        # Fe-C-N linear
        C1a = Fe1 + np.array([0,  d_L, 0])
        N1a = Fe1 + np.array([0,  d_L+1.16, 0])
        C1b = Fe1 + np.array([0, 0,  d_L])
        N1b = Fe1 + np.array([0, 0,  d_L+1.16])
        C2a = Fe2 + np.array([0, -d_L, 0])
        N2a = Fe2 + np.array([0, -d_L-1.16, 0])
        C2b = Fe2 + np.array([0, 0, -d_L])
        N2b = Fe2 + np.array([0, 0, -d_L-1.16])
        atoms = [
            ('Fe', Fe1), ('Fe', Fe2),
            ('S',  S1),  ('S',  S2),
            ('C', C1a), ('N', N1a),
            ('C', C1b), ('N', N1b),
            ('C', C2a), ('N', N2a),
            ('C', C2b), ('N', N2b),
        ]

    # Apply distortion
    atom_str = ''
    for i, (sym, pos) in enumerate(atoms):
        p = pos + noise[i] if i < len(noise) else pos
        atom_str += (f'{sym}  {p[0]:.4f}  '
                     f'{p[1]:.4f}  {p[2]:.4f}\n')
    return atom_str.strip()

# ── GENERATE ALL SYSTEMS ───────────────────────────────────────
systems  = []
skipped  = 0

for terminal in TERMINALS:
    for fe_fe in FE_FE_DISTS:
        for charge in CHARGE_SPINS:
            for spin in CHARGE_SPINS[charge]:
                for distort in [0.0, 0.05]:

                    name = (f'Fe2S2_{terminal}_'
                            f'chg{charge}_'
                            f'spin{spin}_'
                            f'r{int(fe_fe*100)}_'
                            f'd{int(distort*100)}')

                    # Skip if file exists
                    outf = f'{outdir}/{name}.json'
                    if os.path.exists(outf):
                        skipped += 1
                        continue

                    try:
                        atom_str = build_fe2s2_geometry(
                            terminal, fe_fe,
                            charge, distort)
                    except Exception as e:
                        print(f'Geom failed {name}: {e}')
                        continue

                    record = {
                        'name'       : name,
                        'cluster'    : 'Fe2S2',
                        'terminal'   : terminal,
                        'charge'     : charge,
                        'spin'       : spin,
                        'mult'       : spin + 1,
                        'fe_fe_dist' : fe_fe,
                        'distort'    : distort,
                        'atom_str'   : atom_str,
                        'n_fe'       : 2,
                        'n_s_bridge' : 2,
                        'n_terminal' : 4,
                        'status'     : 'generated',
                    }

                    json.dump(record,
                              open(outf, 'w'),
                              indent=2)
                    systems.append(name)

print(f'Generated: {len(systems)} new systems')
print(f'Skipped (exist): {skipped}')
print(f'Total in directory: '
      f'{len(os.listdir(outdir))}')

# Write commands file for array jobs
cmdfile = os.path.expanduser(
    '~/activeml/scripts/fe2s2_commands.txt')
all_files = sorted([
    f'{outdir}/{f}'
    for f in os.listdir(outdir)
    if f.endswith('.json')
])
with open(cmdfile, 'w') as f:
    for fp in all_files:
        f.write(fp + '\n')
print(f'Commands written: {cmdfile}')
print(f'Total systems to process: {len(all_files)}')

# Summary
print('\nBreakdown:')
by_terminal = {}
by_charge   = {}
by_spin     = {}
for fp in all_files:
    d = json.load(open(fp))
    t = d['terminal']
    c = d['charge']
    s = d['spin']
    by_terminal[t] = by_terminal.get(t, 0) + 1
    by_charge[c]   = by_charge.get(c, 0) + 1
    by_spin[s]     = by_spin.get(s, 0) + 1

print('By terminal:', dict(sorted(by_terminal.items())))
print('By charge:',
      dict(sorted(by_charge.items())))
print('By spin (2S):',
      dict(sorted(by_spin.items())))
