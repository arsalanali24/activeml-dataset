"""
predict_with_rules.py
=====================
ML1 prediction with chemistry rule layer.
Rules encode physical laws that RF cannot learn reliably.
Run after closed_shell and carbonyl jobs finish.
"""
import json, glob, os, logging, warnings
import numpy as np
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_val_score
from collections import Counter

BASE     = os.path.expanduser('~/activeml/data')
REAL_DIR = os.path.join(BASE, 'real_cases')

# ── Features ──────────────────────────────────────────────────
LIG_GROUPS = {
    'Cl':'halide','Br':'halide','F':'halide','I':'halide',
    'N':'N_donor','O':'O_donor','S':'S_donor',
    'C':'pi_acceptor','CN':'pi_acceptor','CO':'pi_acceptor',
    'P':'P_donor','PH3':'P_donor',
    'bipy':'bidentate_N','en':'bidentate_N',
    'phen':'bidentate_N','NNN':'bidentate_N',
    'acac':'bidentate_O','ox':'bidentate_O','dtc':'bidentate_S',
    'N4':'porphyrin','N4Cl':'porphyrin','N4Cl2':'porphyrin',
    'N4O':'porphyrin','PNP':'pincer',
}
ROW_ENC  = {'3d':0,'4d':1,'5d':2}
GEOM_ENC = {
    'oct':0,'sq_pl':1,'tet':2,'sqpyr':3,'tbp':4,
    'linear':5,'csd_real':6,'cod_real':7,'bidentate':8,
    'porphyrin':9,'mixed':10,'closed_shell':11,'unknown':12,
}
LIG_ENC  = {v:i for i,v in enumerate(sorted(set(LIG_GROUPS.values())))}
NUMERIC  = [
    'spin_contamination','homo_lumo_gap','homo_energy','lumo_energy',
    'homo_ab_gap','alpha_beta_overlap','delta_E_HS_LS','mp2_corr',
    'corr_energy','largest_t2','n_frac_uno_001','n_frac_uno_005',
    'n_frac_uno_010','n_frac_uno_020','n_frac_mp2_002','n_frac_mp2_005',
    'n_frac_mp2_010','mayer_bond_order_mean','mayer_bond_order_std',
    'mulliken_metal_charge','n_electrons','charge','spin','n_ligands',
    'dist_ang','z_eff','zeta_so_cm1',
]

def extract_X(recs):
    rows = []
    for d in recs:
        row = [float(d.get(f,np.nan) or np.nan) for f in NUMERIC]
        row.append(ROW_ENC.get(d.get('metal_row','3d'),0))
        row.append(LIG_ENC.get(LIG_GROUPS.get(str(d.get('ligand','Cl')),'other'), len(LIG_ENC)))
        row.append(GEOM_ENC.get(d.get('geometry','oct'),12))
        rows.append(row)
    return np.array(rows, dtype=float)

def load_folder(folder):
    recs = []
    for f in glob.glob(f'{folder}/*.json'):
        try:
            d = json.load(open(f))
            if d.get('status')=='ok' and d.get('n_active',-1)>=0:
                recs.append(d)
        except: pass
    return recs

# ── CHEMISTRY RULE LAYER ──────────────────────────────────────
def apply_rules(records, y_pred_rf):
    """
    Hard physics rules that override RF predictions.
    Applied AFTER RF prediction, these encode inviolable constraints.
    """
    y_pred = y_pred_rf.copy()
    rules_applied = []

    for i, d in enumerate(records):
        metal  = d.get('metal','?')
        ligand = str(d.get('ligand','?'))
        spin   = d.get('spin', 0)
        charge = d.get('charge', 0)
        sc     = d.get('spin_contamination', 0)
        geom   = d.get('geometry', '')
        hl     = d.get('homo_lumo_gap', 0)

        orig = int(y_pred[i])
        rule = None

        # Rule 1 — d0 high-oxidation oxo complexes
        # Os(VIII), Re(VII), W(VI), Mo(VI), Cr(VI), Mn(VII), V(V), Ti(IV)
        # with O or F ligands → n_active = 0
        if (metal in {'Os','Re','W','Mo','Cr','Mn','V','Ti'}
                and ligand in {'O','F'} and spin == 0):
            y_pred[i] = 0; rule = 'd0_oxo'

        # Rule 2 — genuine d10 closed shell
        # Zn(II), Cu(I) always d10 → n_active = 0
        elif metal in {'Zn'} and spin == 0:
            y_pred[i] = 0; rule = 'd10_Zn'

        # Rule 3 — Pd(0)/Pt(0) neutral d10
        # charge=0, spin=0, no halide → catalytic resting state
        elif (metal in {'Pd','Pt'} and spin == 0
                and charge == 0 and ligand in {'P','CO'}):
            y_pred[i] = min(orig, 2); rule = 'd10_Pd0'

        # Rule 4 — CO complexes, low spin, decent HOMO-LUMO gap
        # π-backbonding strongly reduces active space
        elif (ligand == 'CO' and spin == 0
                and sc < 0.5 and hl > 0.10):
            y_pred[i] = min(orig, 4); rule = 'CO_pi_back'

        # Rule 5 — d8 square planar Ir(I)/Rh(I) with CO or PR3
        # Closed shell d8 → small active space
        elif (metal in {'Ir','Rh'} and spin == 0
                and ligand in {'P','CO','PH3'}
                and geom in {'sq_pl','linear','sqpyr','csd_real'}):
            y_pred[i] = min(orig, 4); rule = 'd8_sqpl'

        # Rule 6 — d6 low spin 4d/5d with 6 N donors (bipy type)
        # Strong field ligands → always LS d6 → n_active=6-8
        elif (metal in {'Ru','Os','Ir'} and spin == 0
                and ligand in {'N','bipy','phen'}
                and geom == 'oct'):
            y_pred[i] = max(min(orig, 8), 4); rule = 'd6_4d5d_LS'

        if rule:
            rules_applied.append((i, d.get('name','?'),
                                   orig, int(y_pred[i]), rule))

    return y_pred, rules_applied


# ── LOAD TRAINING DATA ────────────────────────────────────────
log.info("Loading training data...")
FOLDERS = [
    'generated300','generated_4d5d','generated_polyatomic',
    'generated_csd','generated_cod','generated_csd_extra',
    'generated_cod_extra','generated_bidentate','generated_phosphine',
    'generated_row5d_extra','generated_porphyrin',
    'generated_closed_shell','generated_carbonyl',  # new!
]
all_recs = []
for folder in FOLDERS:
    path = os.path.join(BASE, folder)
    if not os.path.exists(path): continue
    recs = load_folder(path)
    log.info(f"  {folder}: {len(recs)}")
    all_recs.extend(recs)
log.info(f"Total: {len(all_recs)}")

X = extract_X(all_recs)
y = np.array([d['n_active'] for d in all_recs], dtype=float)
g = np.array([d.get('metal','?')+'_'+d.get('ligand','?') for d in all_recs])

imp = SimpleImputer(strategy='median')
Xi  = imp.fit_transform(X)

# Capped weights
counts = Counter(y.astype(int))
MAX_W  = 4.0
wmap   = {k: min(len(y)/(len(counts)*v), MAX_W) for k,v in counts.items()}
sw     = np.array([wmap[int(yi)] for yi in y])

log.info("Training RF...")
rf = RandomForestRegressor(n_estimators=500,min_samples_leaf=2,
                            max_features='sqrt',random_state=42,n_jobs=-1)
rf.fit(Xi, y, sample_weight=sw)

cv = GroupKFold(n_splits=5)
sc_cv = cross_val_score(rf, Xi, y, groups=g, cv=cv,
                         scoring='neg_mean_absolute_error',
                         params={'sample_weight':sw})
w1 = np.mean(np.abs(rf.predict(Xi)-y)<=1.0)
log.info(f"CV MAE: {-sc_cv.mean():.3f}  Train within±1: {100*w1:.1f}%")

# ── PREDICT REAL CASES ────────────────────────────────────────
real_recs  = load_folder(REAL_DIR)
log.info(f"Real cases: {len(real_recs)}")

Xr  = extract_X(real_recs)
Xri = imp.transform(Xr)
y_rf_raw = rf.predict(Xri)
y_rf     = np.round(y_rf_raw).astype(int)

# Apply chemistry rules
y_final, rules = apply_rules(real_recs, y_rf)

if rules:
    log.info("\nRules applied:")
    for idx, name, orig, new, rule in rules:
        log.info(f"  {name[:35]:<35} {orig}->{new} [{rule}]")

# ── RESULTS TABLE ─────────────────────────────────────────────
log.info("\n"+"="*75)
log.info("FINAL RESULTS: RF + Chemistry Rules")
log.info("="*75)
log.info(f"  {'Case':<35} {'M':<3} {'Pub':>4} {'RF':>4} {'Final':>6} {'Err':>4}  OK")
log.info("-"*75)

n_exact=n_w1=0; results=[]
for i,(d,rf_p,fin) in enumerate(zip(real_recs,y_rf,y_final)):
    pub  = d['published_n_active']
    err  = int(fin)-pub
    ok   = abs(err)==0; w1b = abs(err)<=1
    flag = "✓" if ok else ("~" if w1b else "✗")
    conv = "[nc]" if not d.get('converged',True) else ""
    log.info(f"  {d['name'][:35]:<35} {d['metal']:<3} {pub:>4} "
             f"{rf_p:>4} {fin:>6} {err:>+4}  {flag} {conv}")
    if ok: n_exact+=1
    if w1b: n_w1+=1
    results.append({'name':d['name'],'metal':d['metal'],
                    'published':pub,'rf_pred':int(rf_p),
                    'final_pred':int(fin),'error':err,
                    'exact':bool(ok),'within_1':bool(w1b)})

log.info("-"*75)
log.info(f"  Exact:    {n_exact}/10 ({10*n_exact}%)")
log.info(f"  Within±1: {n_w1}/10  ({10*n_w1}%)")
log.info("="*75)

log.info("\nAutoRAS COMPARISON:")
for r in results:
    if 'Microsoft' in r['name']:
        log.info(f"  Expert:   {r['published']} orbitals")
        log.info(f"  ML+rules: {r['final_pred']} orbitals")
        log.info(f"  AutoCAS:  59 orbitals")
        log.info(f"  Reduction vs AutoCAS: "
                 f"{59-r['final_pred']} orbitals "
                 f"({100*(59-r['final_pred'])/59:.0f}%)")

json.dump({'n_training':len(all_recs),'cv_mae':float(-sc_cv.mean()),
           'train_within1':float(w1),'exact':n_exact,'within_1':n_w1,
           'results':results},
          open(os.path.join(REAL_DIR,'final_results.json'),'w'),indent=2)
log.info("Done!")
