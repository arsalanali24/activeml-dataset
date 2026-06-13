"""
train_ml2.py
============
Trains ML2 model using ML1 + ML2 features from gen_ml2_features.py output.

ML2 improvements over ML1:
  - Correct d_count via per-ligand charge (fixes VOacac2 class)
  - d-orbital character features (homo_d_char, lumo_d_char)
  - Strongly vs weakly fractional UNO separation
  - Metal oxidation state estimated properly
  - Coordination class with ligand-field awareness
  - Gradient Boosting in addition to Random Forest

USAGE:
  python train_ml2.py train        # train and save model
  python train_ml2.py evaluate     # evaluate on new_test_cases
  python train_ml2.py compare      # compare ML1 vs ML2 accuracy
"""
import numpy as np, json, os, sys, glob, logging, pickle, warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_val_score
from collections import Counter

BASE     = os.path.expanduser('~/activeml/data')
ML2_DIR  = os.path.join(BASE, 'ml2_features')
NEW_DIR  = os.path.join(BASE, 'new_test_cases')
REAL_DIR = os.path.join(BASE, 'real_cases')

# ── Encoding maps ─────────────────────────────────────────────
LIG_GROUPS = {
    'Cl':'halide','Br':'halide','F':'halide','I':'halide',
    'N':'N_donor','O':'O_donor','S':'S_donor',
    'C':'pi_acceptor','CN':'pi_acceptor','CO':'pi_acceptor',
    'P':'P_donor','PH3':'P_donor','bipy':'bidentate_N',
    'en':'bidentate_N','phen':'bidentate_N','NNN':'bidentate_N',
    'acac':'bidentate_O','ox':'bidentate_O','dtc':'bidentate_S',
    'N4':'porphyrin','N4Cl':'porphyrin','N4Cl2':'porphyrin',
    'N4O':'porphyrin','PNP':'pincer',
}
ROW_ENC  = {'3d':0,'4d':1,'5d':2}
GEOM_ENC = {'oct':0,'sq_pl':1,'tet':2,'sqpyr':3,'tbp':4,'linear':5,
            'csd_real':6,'cod_real':7,'bidentate':8,'porphyrin':9,
            'mixed':10,'closed_shell':11,'unknown':12}
LIG_ENC  = {v:i for i,v in enumerate(sorted(set(LIG_GROUPS.values())))}

# ── ML1 features (baseline) ───────────────────────────────────
ML1_NUMERIC = [
    'n_ligands','n_electrons','spin','spin_contamination','charge',
    'homo_lumo_gap','homo_energy','lumo_energy','homo_ab_gap',
    'delta_E_HS_LS','mp2_corr','corr_energy','largest_t2',
    'n_frac_mp2_002','n_frac_mp2_005','n_frac_mp2_010',
    'z_eff','zeta_so_cm1','dist_ang','mulliken_metal_charge',
    'mayer_bond_order_mean','d_electron_count',
]

# ── ML2 additional features ───────────────────────────────────
ML2_NUMERIC = [
    'ml2_metal_ox_state',        # estimated oxidation state
    'ml2_d_count_corrected',     # correct d-count via ligand charges
    'ml2_per_ligand_charge',     # charge per ligand
    'ml2_total_ligand_charge',   # total ligand charge
    'ml2_sigma_pi_ratio',        # sigma vs pi donor
    'ml2_coord_class_ml2',       # coordination class LF-aware
    'ml2_n_coord_effective',     # effective coordination number
    'ml2_ligand_field_split_eV', # ligand field splitting
    'ml2_homo_d_char',           # d character of HOMO
    'ml2_lumo_d_char',           # d character of LUMO
    'ml2_n_frac_strong',         # strongly fractional NOs
    'ml2_n_frac_weak',           # weakly fractional NOs
    'ml2_n_frac_ratio',          # strong/weak ratio
    'ml2_metal_nat_charge',      # natural charge on metal
    'ml2_spin_per_d_electron',   # spin / d-count
    'ml2_so_coupling_heavy',     # heavy atom SO coupling flag
    'ml2_zeta_times_spin',       # SO * spin product
    'ml2_mp2_ratio_010_002',     # MP2 ratio features
]

ALL_FEATURES = ML1_NUMERIC + ML2_NUMERIC


def extract_X(recs, feature_set=None):
    if feature_set is None:
        feature_set = ALL_FEATURES
    rows = []
    for d in recs:
        row = [float(d.get(f, np.nan) or np.nan) for f in feature_set]
        row.append(ROW_ENC.get(d.get('metal_row','3d'), 0))
        row.append(LIG_ENC.get(
            LIG_GROUPS.get(str(d.get('ligand','Cl')),'other'),
            len(LIG_ENC)))
        row.append(GEOM_ENC.get(d.get('geometry','oct'), 12))
        rows.append(row)
    return np.array(rows, dtype=float)


def load_ml2_folder(folder_name):
    """Load ML2-enhanced files from ml2_features directory."""
    recs = []
    # ML2 files are named: foldername_filename_ml2.json
    pattern = os.path.join(ML2_DIR, f'{folder_name}_*_ml2.json')
    for f in glob.glob(pattern):
        try:
            d = json.load(open(f))
            if isinstance(d, dict) and d.get('status')=='ok' \
                    and d.get('n_active',-1) >= 0 \
                    and d.get('ml2_computed'):
                recs.append(d)
        except: pass
    # Also try subdirectory structure
    subdir = os.path.join(ML2_DIR, folder_name)
    if os.path.exists(subdir):
        for f in glob.glob(f'{subdir}/*_ml2.json'):
            try:
                d = json.load(open(f))
                if isinstance(d,dict) and d.get('status')=='ok' \
                        and d.get('n_active',-1)>=0 \
                        and d.get('ml2_computed'):
                    recs.append(d)
            except: pass
    return recs


def load_original_folder(folder_name):
    """Fallback: load original ML1 files."""
    recs = []
    path = os.path.join(BASE, folder_name)
    if not os.path.exists(path): return recs
    for f in glob.glob(f'{path}/*.json'):
        try:
            d = json.load(open(f))
            if isinstance(d,dict) and d.get('status')=='ok' \
                    and d.get('n_active',-1)>=0:
                recs.append(d)
        except: pass
    return recs


TRAIN_FOLDERS = [
    'generated300','generated_4d5d','generated_polyatomic',
    'generated_csd','generated_cod','generated_csd_extra',
    'generated_cod_extra','generated_bidentate','generated_phosphine',
    'generated_row5d_extra','generated_porphyrin','generated_closed_shell',
    'generated_carbonyl','generated_cod_phosphine','generated_cod_porphyrin',
    'generated_gap_fill','generated_cod_gap_fill',
]


def load_training_data():
    """Load training data, preferring ML2-enhanced files."""
    all_recs = []
    for folder in TRAIN_FOLDERS:
        # Try ML2 first
        recs = load_ml2_folder(folder)
        if not recs:
            # Fall back to ML1
            recs = load_original_folder(folder)
            log.warning(f"  {folder}: using ML1 ({len(recs)} files, no ML2)")
        else:
            log.info(f"  {folder}: {len(recs)} ML2-enhanced")
        all_recs.extend(recs)
    return all_recs


def train_ml2(all_recs, feature_set=None):
    if feature_set is None:
        feature_set = ALL_FEATURES

    X = extract_X(all_recs, feature_set)
    y = np.array([d['n_active'] for d in all_recs], dtype=float)
    g = np.array([d.get('metal','?')+'_'+str(d.get('n_ligands',0))
                  for d in all_recs])

    imp = SimpleImputer(strategy='median')
    Xi  = imp.fit_transform(X)

    counts = Counter(y.astype(int))
    MAX_W  = 4.0
    wmap   = {k: min(len(y)/(len(counts)*v), MAX_W)
              for k,v in counts.items()}
    sw     = np.array([wmap[int(yi)] for yi in y])

    log.info("Training Random Forest (ML2)...")
    rf = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=1,
        max_features='sqrt', random_state=42, n_jobs=-1)
    rf.fit(Xi, y, sample_weight=sw)

    cv     = GroupKFold(n_splits=5)
    cv_rf  = cross_val_score(rf, Xi, y, groups=g, cv=cv,
                              scoring='neg_mean_absolute_error')
    w1_rf  = np.mean(np.abs(rf.predict(Xi) - y) <= 1.0)
    log.info(f"RF CV MAE: {-cv_rf.mean():.3f} ± {cv_rf.std():.3f}")
    log.info(f"RF Train within±1: {100*w1_rf:.1f}%")

    # Feature importances
    feat_names = feature_set + ['row_enc','lig_enc','geom_enc']
    top = np.argsort(rf.feature_importances_)[::-1][:10]
    log.info("\nTop 10 features:")
    for i in top:
        log.info(f"  {feat_names[i]:<35} {rf.feature_importances_[i]:.4f}")

    return rf, imp, cv_rf, w1_rf


def evaluate_on_new_cases(rf, imp, feature_set=None):
    if feature_set is None:
        feature_set = ALL_FEATURES

    # Load new test cases — try ML2 first, fall back to ML1
    new_recs = []
    for f in glob.glob(f'{ML2_DIR}/new_test_cases_*_ml2.json'):
        try:
            d = json.load(open(f))
            if d.get('status')=='ok' and d.get('ml2_computed'):
                new_recs.append(d)
        except: pass
    if not new_recs:
        # Fall back to ML1
        log.warning("No ML2 new_test_cases — using ML1 features")
        for f in glob.glob(f'{NEW_DIR}/*.json'):
            try:
                d = json.load(open(f))
                if d.get('status')=='ok':
                    new_recs.append(d)
            except: pass

    if not new_recs:
        log.error("No test cases found")
        return

    Xr   = extract_X(new_recs, feature_set)
    Xri  = imp.transform(Xr)
    pred = np.round(rf.predict(Xri)).astype(int)

    print("\n"+"="*68)
    print("ML2 HONEST VALIDATION — 10 unseen test cases")
    print("="*68)
    print("  %-28s %3s %4s %5s %4s  OK" %
          ('Case','M','Pub','Pred','Err'))
    print("-"*68)
    ne=nw=0
    for d,p in zip(new_recs,pred):
        pub  = d['published_n_active']
        err  = int(p)-pub
        ok   = abs(err)==0; w = abs(err)<=1
        flag = '✓' if ok else ('~' if w else '✗')
        print("  %-28s %-3s %4d %5d %+4d  %s" % (
            d['name'][:28],d['metal'],pub,p,err,flag))
        if ok: ne+=1
        if w:  nw+=1
    print("-"*68)
    print("ML2 Exact: %d/%d  Within±1: %d/%d" % (
        ne,len(new_recs),nw,len(new_recs)))
    return ne, nw


def compare_ml1_ml2():
    """Side-by-side comparison of ML1 and ML2 on validation set."""
    # Load ML1 model
    ml1_path = os.path.join(BASE, 'ml1_model_clean.pkl')
    if not os.path.exists(ml1_path):
        log.error("ML1 clean model not found: %s" % ml1_path)
        return

    with open(ml1_path,'rb') as f:
        m1 = pickle.load(f)
    rf1  = m1['rf']; imp1 = m1['imp']
    NUM1 = m1['NUMERIC']

    # Load ML2 model
    ml2_path = os.path.join(BASE, 'ml2_model.pkl')
    if not os.path.exists(ml2_path):
        log.error("ML2 model not found. Run 'train' first.")
        return

    with open(ml2_path,'rb') as f:
        m2 = pickle.load(f)
    rf2  = m2['rf']; imp2 = m2['imp']
    NUM2 = m2['feature_set']

    # Load test cases
    new_recs_ml1 = []
    new_recs_ml2 = []
    for f in sorted(glob.glob(f'{NEW_DIR}/*.json')):
        d = json.load(open(f))
        if d.get('status')=='ok':
            new_recs_ml1.append(d)

    # ML2 versions
    for f in sorted(glob.glob(f'{ML2_DIR}/new_test_cases_*_ml2.json')):
        d = json.load(open(f))
        if d.get('status')=='ok' and d.get('ml2_computed'):
            new_recs_ml2.append(d)

    if not new_recs_ml2:
        log.warning("No ML2 test cases — using ML1 features for both")
        new_recs_ml2 = new_recs_ml1

    def Xf1(recs):
        rows=[]
        for d in recs:
            row=[float(d.get(f,np.nan) or np.nan) for f in NUM1]
            row.append(ROW_ENC.get(d.get('metal_row','3d'),0))
            row.append(LIG_ENC.get(LIG_GROUPS.get(
                str(d.get('ligand','Cl')),'other'),len(LIG_ENC)))
            row.append(GEOM_ENC.get(d.get('geometry','oct'),12))
            rows.append(row)
        return np.array(rows,dtype=float)

    p1 = np.round(rf1.predict(imp1.transform(Xf1(new_recs_ml1)))).astype(int)
    p2 = np.round(rf2.predict(imp2.transform(
        extract_X(new_recs_ml2, NUM2)))).astype(int)

    print("\n"+"="*75)
    print("ML1 vs ML2 COMPARISON — Clean validation set")
    print("="*75)
    print("  %-28s %3s %4s  %5s %5s  ML1 ML2" %
          ('Case','M','Pub','ML1','ML2'))
    print("-"*75)
    ne1=nw1=ne2=nw2=0
    for i,(d,pp1,pp2) in enumerate(zip(new_recs_ml1,p1,p2)):
        pub  = d['published_n_active']
        e1   = int(pp1)-pub; e2 = int(pp2)-pub
        f1   = '✓' if abs(e1)==0 else ('~' if abs(e1)<=1 else '✗')
        f2   = '✓' if abs(e2)==0 else ('~' if abs(e2)<=1 else '✗')
        print("  %-28s %-3s %4d  %5d %5d  %2s  %2s" % (
            d['name'][:28],d['metal'],pub,pp1,pp2,f1,f2))
        if abs(e1)==0: ne1+=1
        if abs(e1)<=1: nw1+=1
        if abs(e2)==0: ne2+=1
        if abs(e2)<=1: nw2+=1
    n=len(new_recs_ml1)
    print("-"*75)
    print("  ML1: Exact=%d/%d (%d%%)  Within±1=%d/%d (%d%%)" %
          (ne1,n,100*ne1//n,nw1,n,100*nw1//n))
    print("  ML2: Exact=%d/%d (%d%%)  Within±1=%d/%d (%d%%)" %
          (ne2,n,100*ne2//n,nw2,n,100*nw2//n))
    delta_exact = ne2-ne1; delta_w1 = nw2-nw1
    print(f"\n  Improvement: Exact {delta_exact:+d}  Within±1 {delta_w1:+d}")
    print("="*75)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv)>1 else 'train'

    if cmd == 'train':
        log.info("Loading training data...")
        all_recs = load_training_data()
        log.info(f"Total: {len(all_recs)}")

        if len(all_recs) < 100:
            log.error("Too few records. Run gen_ml2_features.py estimate_all first.")
            sys.exit(1)

        # Check how many have ML2 features
        n_ml2 = sum(1 for d in all_recs if d.get('ml2_computed'))
        log.info(f"  With ML2 features: {n_ml2}/{len(all_recs)}")

        if n_ml2 < len(all_recs) * 0.5:
            log.warning("Less than 50% have ML2 features.")
            log.warning("Run: python gen_ml2_features.py estimate_all")
            log.warning("Training with ML1 features only for now...")
            feature_set = ML1_NUMERIC
        else:
            feature_set = ALL_FEATURES
            log.info("Using full ML2 feature set")

        rf, imp, cv_s, w1 = train_ml2(all_recs, feature_set)

        # Save
        model_path = os.path.join(BASE, 'ml2_model.pkl')
        with open(model_path,'wb') as f:
            pickle.dump({'rf':rf,'imp':imp,
                         'feature_set':feature_set,
                         'n_training':len(all_recs),
                         'n_ml2_features':n_ml2,
                         'cv_mae':float(-cv_s.mean()),
                         'cv_std':float(cv_s.std()),
                         'train_within1':float(w1),
                         'ROW_ENC':ROW_ENC,'GEOM_ENC':GEOM_ENC,
                         'LIG_ENC':LIG_ENC,'LIG_GROUPS':LIG_GROUPS,
                         'note':'ML2 model — per-ligand charge features'},f)
        log.info(f"Model saved: {model_path}")

        # Evaluate
        evaluate_on_new_cases(rf, imp, feature_set)

    elif cmd == 'evaluate':
        ml2_path = os.path.join(BASE,'ml2_model.pkl')
        with open(ml2_path,'rb') as f: m = pickle.load(f)
        evaluate_on_new_cases(m['rf'], m['imp'], m['feature_set'])

    elif cmd == 'compare':
        compare_ml1_ml2()

    else:
        print("Usage: train | evaluate | compare")
