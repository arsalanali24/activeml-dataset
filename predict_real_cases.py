"""
predict_real_cases.py
=====================
Trains ML1 model on full dataset, then tests on 10 real benchmark cases.
Generates a comparison table: predicted vs published n_active.

Usage:
  python predict_real_cases.py

Output:
  ~/activeml/data/real_cases/prediction_results.json
  ~/activeml/data/real_cases/prediction_table.txt
"""
import json, glob, os, sys, logging
import numpy as np
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import GroupKFold, cross_val_score
    import sklearn
    log.info(f"sklearn version: {sklearn.__version__}")
except ImportError:
    log.error("sklearn not installed — run: pip install scikit-learn --user")
    sys.exit(1)

BASE     = os.path.expanduser('~/activeml/data')
REAL_DIR = os.path.join(BASE, 'real_cases')

# ══════════════════════════════════════════════════════════════
# FEATURE DEFINITIONS — must match training data exactly
# ══════════════════════════════════════════════════════════════

NUMERIC_FEATURES = [
    'spin_contamination', 'homo_lumo_gap', 'homo_energy',
    'lumo_energy', 'homo_ab_gap', 'alpha_beta_overlap',
    'delta_E_HS_LS', 'mp2_corr', 'corr_energy',
    'largest_t2', 'n_frac_uno_001', 'n_frac_uno_005',
    'n_frac_uno_010', 'n_frac_uno_020',
    'n_frac_mp2_002', 'n_frac_mp2_005', 'n_frac_mp2_010',
    'mayer_bond_order_mean', 'mayer_bond_order_std',
    'mulliken_metal_charge', 'n_electrons',
    'charge', 'spin', 'n_ligands', 'dist_ang',
    'z_eff', 'zeta_so_cm1',
]

CATEGORICAL_FEATURES = ['metal_row', 'ligand', 'geometry']

# ── Ligand group encoding (same as training notebook) ────────
LIG_GROUPS = {
    'Cl':'halide','Br':'halide','F':'halide','I':'halide',
    'N':'N_donor','O':'O_donor','S':'S_donor',
    'P':'P_donor','C':'pi_acceptor',
    'CN':'pi_acceptor','CO':'pi_acceptor',
    'PH3':'P_donor',
    'bipy':'bidentate_N','en':'bidentate_N',
    'phen':'bidentate_N','acac':'bidentate_O',
    'ox':'bidentate_O','dtc':'bidentate_S',
    'PNP':'pincer','NNN':'pincer',
    'N4':'porphyrin','N4Cl':'porphyrin','N4Cl2':'porphyrin',
    'N4O':'porphyrin',
}

def get_lig_group(lig):
    return LIG_GROUPS.get(lig, 'other')

ROW_ENC  = {'3d': 0, '4d': 1, '5d': 2}
GEOM_ENC = {
    'oct':0,'sq_pl':1,'tet':2,'sqpyr':3,'tbp':4,
    'linear':5,'csd_real':6,'cod_real':7,
    'bidentate':8,'porphyrin':9,'mixed':10,'unknown':11,
}


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_folder(folder):
    records = []
    for f in glob.glob(f'{folder}/*.json'):
        try:
            d = json.load(open(f))
            if d.get('status') != 'ok': continue
            if d.get('n_active', -1) < 0: continue
            records.append(d)
        except: pass
    return records


def extract_X(records):
    rows = []
    for d in records:
        row = []
        for feat in NUMERIC_FEATURES:
            val = d.get(feat, np.nan)
            row.append(float(val) if val is not None else np.nan)
        # Categorical encoded
        row.append(ROW_ENC.get(d.get('metal_row','3d'), 0))
        lig_grp = get_lig_group(d.get('ligand','Cl'))
        # Simple ordinal encoding for ligand group
        lig_enc = list(set(LIG_GROUPS.values()))
        row.append(lig_enc.index(lig_grp) if lig_grp in lig_enc else 0)
        row.append(GEOM_ENC.get(d.get('geometry','oct'), 11))
        rows.append(row)
    return np.array(rows, dtype=float)


def extract_y(records):
    return np.array([d['n_active'] for d in records], dtype=float)


def extract_groups(records):
    return np.array([d.get('metal','?') + '_' +
                     d.get('ligand','?') for d in records])


# ══════════════════════════════════════════════════════════════
# LOAD ALL TRAINING DATA
# ══════════════════════════════════════════════════════════════

log.info("Loading training data...")
training_folders = [
    'generated300', 'generated_4d5d', 'generated_polyatomic',
    'generated_csd', 'generated_cod', 'generated_csd_extra',
    'generated_cod_extra', 'generated_bidentate',
    'generated_phosphine', 'generated_row5d_extra',
    'generated_porphyrin',
]

all_records = []
for folder in training_folders:
    path    = os.path.join(BASE, folder)
    records = load_folder(path)
    log.info(f"  {folder}: {len(records)}")
    all_records.extend(records)

log.info(f"Total training records: {len(all_records)}")

X_train = extract_X(all_records)
y_train = extract_y(all_records)
groups  = extract_groups(all_records)

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)

log.info(f"Feature matrix: {X_train_imp.shape}")
log.info(f"Target range: {y_train.min():.0f} – {y_train.max():.0f}")


# ══════════════════════════════════════════════════════════════
# TRAIN MODEL
# ══════════════════════════════════════════════════════════════

log.info("\nTraining Random Forest...")
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    min_samples_split=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train_imp, y_train)
log.info("  Training complete")

# Quick CV estimate
cv = GroupKFold(n_splits=5)
cv_scores = cross_val_score(rf, X_train_imp, y_train,
                             groups=groups, cv=cv,
                             scoring='neg_mean_absolute_error')
log.info(f"  CV MAE: {-cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Within±1 accuracy
train_pred = rf.predict(X_train_imp)
within1 = np.mean(np.abs(train_pred - y_train) <= 1.0)
log.info(f"  Training within±1: {100*within1:.1f}%")


# ══════════════════════════════════════════════════════════════
# PREDICT ON REAL CASES
# ══════════════════════════════════════════════════════════════

log.info("\nLoading real benchmark cases...")
real_records = load_folder(REAL_DIR)
log.info(f"  Loaded: {len(real_records)} cases")

X_real     = extract_X(real_records)
X_real_imp = imputer.transform(X_real)
y_pred_raw = rf.predict(X_real_imp)
y_pred     = np.round(y_pred_raw).astype(int)

# ── Results table ─────────────────────────────────────────────
log.info("\n" + "="*75)
log.info("BENCHMARK RESULTS: Predicted vs Published n_active")
log.info("="*75)
log.info(f"  {'Case':<35} {'Metal':<4} {'Pub':>4} {'Pred':>5} "
         f"{'Raw':>6} {'Err':>4} {'OK?'}")
log.info("-"*75)

results = []
n_correct = n_within1 = 0

for i, (d, pred, pred_raw) in enumerate(
        zip(real_records, y_pred, y_pred_raw)):
    pub    = d['published_n_active']
    err    = int(pred) - pub
    ok     = abs(err) == 0
    within = abs(err) <= 1
    flag   = "✓" if ok else ("~" if within else "✗")
    conv   = "conv" if d.get('converged', True) else "no-conv"

    log.info(f"  {d['name'][:35]:<35} {d['metal']:<4} {pub:>4} "
             f"{pred:>5} {pred_raw:>6.2f} {err:>+4}  {flag}  [{conv}]")

    if ok: n_correct += 1
    if within: n_within1 += 1

    results.append({
        'name':             d['name'],
        'metal':            d['metal'],
        'ligand':           d['ligand'],
        'published_n_active': pub,
        'predicted_n_active': int(pred),
        'predicted_raw':    float(pred_raw),
        'error':            err,
        'exact_correct':    bool(ok),
        'within_1':         bool(within),
        'converged':        bool(d.get('converged', True)),
        'reference':        d.get('reference',''),
        'spin_contamination': float(d.get('spin_contamination',0)),
    })

log.info("-"*75)
log.info(f"  Exact correct:  {n_correct}/{len(results)} "
         f"({100*n_correct/len(results):.0f}%)")
log.info(f"  Within ±1:      {n_within1}/{len(results)} "
         f"({100*n_within1/len(results):.0f}%)")
log.info("="*75)

# ── AutoCAS comparison ────────────────────────────────────────
log.info("\nAutoCAS COMPARISON (Microsoft Fe-PNP pincer case):")
for r in results:
    if 'Microsoft' in r['name'] or 'PNP' in r['name']:
        log.info(f"  Published expert selection:  {r['published_n_active']} orbitals")
        log.info(f"  Your model prediction:       {r['predicted_n_active']} orbitals")
        log.info(f"  AutoCAS (entropy bloat):     59 orbitals")
        reduction = 59 - r['predicted_n_active']
        log.info(f"  Reduction vs AutoCAS:        {reduction} orbitals "
                 f"({100*reduction/59:.0f}% smaller)")

# ── Save ─────────────────────────────────────────────────────
out_json = os.path.join(REAL_DIR, 'prediction_results.json')
out_txt  = os.path.join(REAL_DIR, 'prediction_table.txt')

json.dump({
    'n_training':  len(all_records),
    'cv_mae':      float(-cv_scores.mean()),
    'train_within1': float(within1),
    'n_cases':     len(results),
    'exact_correct': n_correct,
    'within_1':    n_within1,
    'results':     results,
}, open(out_json,'w'), indent=2)

with open(out_txt,'w') as f:
    f.write("REAL CASE BENCHMARK RESULTS\n")
    f.write("="*75 + "\n")
    f.write(f"Training set: {len(all_records)} structures\n")
    f.write(f"CV MAE: {-cv_scores.mean():.3f}\n\n")
    f.write(f"{'Case':<35} {'Metal':<5} {'Published':>9} "
            f"{'Predicted':>9} {'Error':>5} {'OK'}\n")
    f.write("-"*75 + "\n")
    for r in results:
        ok = "✓" if r['exact_correct'] else \
             ("~" if r['within_1'] else "✗")
        f.write(f"  {r['name'][:33]:<33} {r['metal']:<5} "
                f"{r['published_n_active']:>9} "
                f"{r['predicted_n_active']:>9} "
                f"{r['error']:>+5}  {ok}\n")
    f.write("-"*75 + "\n")
    f.write(f"Exact: {n_correct}/{len(results)}  "
            f"Within±1: {n_within1}/{len(results)}\n")

log.info(f"\nSaved: {out_json}")
log.info(f"Saved: {out_txt}")
log.info("Done!")
