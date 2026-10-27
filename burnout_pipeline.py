"""
Burnout risk prediction: preprocessing and encoding first, then models, RQ1-RQ3 analyses
and all figures/tables used in main_burnout_lncs.tex.

Run in Google Colab (pip install shap imbalanced-learn xgboost lightgbm if missing).
Outputs: figures/*.png and results/*.csv, plus LaTeX table rows printed to the console.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')

# Real libraries when available; clearly labelled stand-ins otherwise.
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMB = True
except ImportError:
    HAS_IMB = False

# ============================================================
# 0. Configuration
# ============================================================
PATH = os.environ.get('DATA_PATH', '/content/drive/MyDrive/Google Colab/tech_mental_health_burnout.csv')
OUT_ROOT = os.environ.get('OUT_ROOT', '.')
FIG_DIR = os.path.join(OUT_ROOT, 'figures')
RES_DIR = os.path.join(OUT_ROOT, 'results')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

SEED = 42
QUICK = os.environ.get('QUICK', '0') == '1'      # tiny fast run, only for testing the code
TUNE = os.environ.get('TUNE', '0') == '1'        # random search on the training set (slow)
N_SPLITS = 5
N_BOOT = 50 if QUICK else int(os.environ.get('N_BOOT', '2000'))
SHAP_N = 500 if QUICK else 2000                  # test records used for SHAP
PERM_N = 3000 if QUICK else 10000                # test records used for permutation importance
PERM_REPEATS = 2 if QUICK else 5
PATEL = dict(f1=0.7081, roc=0.9604, prec=0.7156, rec=0.7008)   # original study, XGBoost + SMOTE

print('xgboost:', HAS_XGB, '| lightgbm:', HAS_LGBM, '| shap:', HAS_SHAP, '| imblearn:', HAS_IMB)
if not (HAS_XGB and HAS_LGBM):
    raise ImportError('Install xgboost and lightgbm: benchmark models must not be silently substituted.')

# ============================================================
# 1. Load data and basic checks
# ============================================================
df = pd.read_csv(PATH)
if QUICK:
    df = df.sample(20000, random_state=SEED).reset_index(drop=True)
print('Shape:', df.shape)
print('Missing values:', int(df.isna().sum().sum()), '| duplicate rows:', int(df.duplicated().sum()))
print(df['burnout_level'].value_counts())

# ============================================================
# 2. Target, leakage control and feature categories
# ============================================================
LEAK = ['burnout_level', 'burnout_score']            # binned copy / continuous source of the target
df['at_risk'] = (df['burnout_level'] != 'Low').astype(int)   # 0 = Low, 1 = Moderate or High
y = df['at_risk']
X = df.drop(columns=LEAK + ['at_risk'])

CATEGORIES = {
    'Work environment': ['job_role', 'company_size', 'work_mode', 'work_hours_per_week',
                         'overtime_hours', 'meetings_per_day', 'deadlines_missed',
                         'job_satisfaction', 'manager_support', 'work_life_balance',
                         'social_support_score'],
    'Lifestyle': ['sleep_hours', 'physical_activity_days', 'screen_time_hours', 'caffeine_intake'],
    'Psychological': ['stress_level', 'anxiety_score', 'depression_score'],
    'Demographic': ['age', 'gender', 'experience_years'],
    'Treatment': ['has_therapy', 'seeks_professional_help'],
}
ALL = [c for cols in CATEGORIES.values() for c in cols]
missing_cols = set(ALL) ^ set(X.columns)
assert not missing_cols, f'Column mismatch with the category map: {missing_cols}'
assert not set(LEAK) & set(X.columns), 'Leakage columns are still in X'
X = X[ALL]
CAT_OF = {c: cat for cat, cols in CATEGORIES.items() for c in cols}

NOMINAL = ['gender', 'job_role', 'company_size', 'work_mode']
BINARY = ['has_therapy', 'seeks_professional_help']
assert set(np.unique(X[BINARY].values)) <= {0, 1}, 'Binary columns are not 0/1'

F1_SET = ALL                                                         # all 23 features
F2_SET = [c for c in ALL if c not in CATEGORIES['Psychological']]   # without stress/anxiety/depression
F3_SET = CATEGORIES['Work environment'] + CATEGORIES['Lifestyle']    # work environment + lifestyle only
print(f'Feature sets: F1={len(F1_SET)}, F2={len(F2_SET)}, F3={len(F3_SET)}')

# ============================================================
# 3. Train/test split (stratified 80/20, as in the original study)
# ============================================================
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
SPW = (ytr == 0).sum() / (ytr == 1).sum()            # imbalance ratio used for class weighting
print(f'Train {len(Xtr)} (At Risk {int(ytr.sum())}) | Test {len(Xte)} (At Risk {int(yte.sum())}) | ratio {SPW:.2f}:1')
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# ============================================================
# 4. ENCODING AND PREPROCESSING (done first; fitted on training data only)
# ============================================================
def make_pre(features, scale):
    """One-hot encode nominal columns, keep binary 0/1, standardise numeric columns if scale=True."""
    nominal = [c for c in NOMINAL if c in features]
    binary = [c for c in BINARY if c in features]
    numeric = [c for c in features if c not in nominal + binary]
    return ColumnTransformer([
        ('num', StandardScaler() if scale else 'passthrough', numeric),
        ('bin', 'passthrough', binary),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal),
    ])

demo = make_pre(F1_SET, scale=True).fit(Xtr)
enc_names = list(demo.get_feature_names_out())
enc_demo = pd.DataFrame(demo.transform(Xtr.head(5)), columns=enc_names)
print(f'\nEncoding: {len(F1_SET)} original columns -> {len(enc_names)} model inputs')
print('One-hot columns:', [n for n in enc_names if n.startswith('cat__')])
print(enc_demo.round(2).iloc[:3, :8])
print('Encoded train mean/std check (numeric should be ~0/~1):')
enc_tr = pd.DataFrame(demo.transform(Xtr), columns=enc_names)
print(enc_tr.filter(like='num__').agg(['mean', 'std']).round(2).iloc[:, :4])

# ============================================================
# 5. Exploratory figures
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
counts = df['burnout_level'].value_counts().reindex(['Low', 'Moderate', 'High'])
ax[0].bar(counts.index, counts.values, color=['#4c9f70', '#e08a4b', '#c44e52'])
ax[0].set_title('Burnout level (3 classes)'); ax[0].set_ylabel('Records')
for i, v in enumerate(counts.values):
    ax[0].text(i, v, f'{v:,}\n({v / len(df):.2%})', ha='center', va='bottom', fontsize=8)
bc = df['at_risk'].value_counts().sort_index()
ax[1].bar(['Not at Risk', 'At Risk'], bc.values, color=['#4c72b0', '#c44e52'])
ax[1].set_title('Binary target')
for i, v in enumerate(bc.values):
    ax[1].text(i, v, f'{v:,}\n({v / len(df):.2%})', ha='center', va='bottom', fontsize=8)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/class_distribution.png', dpi=300); plt.show()

num_cols = [c for c in X.columns if c not in NOMINAL]
corr = df[num_cols + ['at_risk']].corr()
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
fig.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/correlation_heatmap.png', dpi=300); plt.show()
print('Top correlations with at_risk:\n', corr['at_risk'].drop('at_risk').abs().sort_values(ascending=False).head(8).round(3))

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for a, (col, title) in zip(ax[:2], [('work_mode', 'Work mode'), ('company_size', 'Company size')]):
    g = df.groupby(col)['at_risk'].mean().sort_values()
    a.bar(g.index.astype(str), g.values, color='#4c72b0'); a.set_title(title)
    for i, v in enumerate(g.values):
        a.text(i, v, f'{v:.1%}', ha='center', va='bottom', fontsize=8)
hours = pd.qcut(df['work_hours_per_week'], 4)
g = df.groupby(hours)['at_risk'].mean()
ax[2].bar([str(i) for i in g.index], g.values, color='#c44e52'); ax[2].set_title('Weekly work hours (quartiles)')
for i, v in enumerate(g.values):
    ax[2].text(i, v, f'{v:.1%}', ha='center', va='bottom', fontsize=8)
ax[2].tick_params(axis='x', labelsize=7)
ax[0].set_ylabel('At Risk rate')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/burnout_by_group.png', dpi=300); plt.show()

# ============================================================
# 6. Models
# ============================================================
MODELS = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'MLP']
SCALED = {'Logistic Regression', 'MLP'}       # scaled inputs; tree models use raw numeric values

DEFAULTS = {
    'Logistic Regression': dict(C=1.0),
    'Random Forest': dict(n_estimators=40 if QUICK else 300, max_depth=8, min_samples_leaf=50, max_features='sqrt'),
    'XGBoost': dict(n_estimators=60 if QUICK else 300, max_depth=3, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, reg_lambda=5, min_child_weight=10),
    'LightGBM': dict(n_estimators=60 if QUICK else 300, num_leaves=8, learning_rate=0.05, reg_lambda=5,
                     min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8),
    'MLP': dict(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=30 if QUICK else 100),
}
SEARCH = {
    'Logistic Regression': dict(C=[0.01, 0.1, 1, 10]),
    'Random Forest': dict(max_depth=[6, 8, 12], min_samples_leaf=[20, 50, 100]),
    'XGBoost': dict(max_depth=[2, 3, 4, 5], learning_rate=[0.03, 0.05, 0.1], n_estimators=[200, 300, 500]),
    'LightGBM': dict(num_leaves=[4, 8, 16], learning_rate=[0.03, 0.05, 0.1], n_estimators=[200, 300, 500],
                     min_child_samples=[100, 200, 400]),
    'MLP': dict(hidden_layer_sizes=[(32,), (64, 32), (128, 64)], alpha=[1e-4, 1e-3, 1e-2]),
}

def _hgb_standin(kind, p, weighted):
    if kind == 'xgb':
        kw = dict(max_depth=p.get('max_depth', 3), min_samples_leaf=100)
    else:
        kw = dict(max_leaf_nodes=p.get('num_leaves', 8), min_samples_leaf=p.get('min_child_samples', 200))
    return HistGradientBoostingClassifier(
        learning_rate=p.get('learning_rate', 0.05), max_iter=p.get('n_estimators', 300),
        l2_regularization=p.get('reg_lambda', 5), class_weight='balanced' if weighted else None,
        random_state=SEED, **kw)

def make_clf(name, params=None, weighted=True):
    """Weighted classical models; MLP is an explicit unweighted probability reference.

    This compares configured pipelines, not architecture effects under identical
    losses. Modern sklearn MLP supports sample_weight; absence of class_weight
    does not make weighted MLP impossible. See paper limitations.
    """
    p = dict(DEFAULTS[name]); p.update(params or {})
    if name == 'Logistic Regression':
        return LogisticRegression(max_iter=2000, class_weight='balanced' if weighted else None, **p)
    if name == 'Random Forest':
        return RandomForestClassifier(class_weight='balanced_subsample' if weighted else None,
                                      n_jobs=-1, random_state=SEED, **p)
    if name == 'XGBoost':
        if HAS_XGB:
            return XGBClassifier(scale_pos_weight=SPW if weighted else 1.0, eval_metric='logloss',
                                 n_jobs=-1, random_state=SEED, **p)
        return _hgb_standin('xgb', p, weighted)
    if name == 'LightGBM':
        if HAS_LGBM:
            return LGBMClassifier(scale_pos_weight=SPW if weighted else 1.0, verbosity=-1,
                                  n_jobs=-1, random_state=SEED, **p)
        return _hgb_standin('lgbm', p, weighted)
    if name == 'MLP':
        return MLPClassifier(early_stopping=True, random_state=SEED, **p)
    raise ValueError(name)

def build(name, features, params=None, weighted=True, smote=False):
    pre = make_pre(features, name in SCALED)
    if smote:
        clf = make_clf(name, params, weighted=False)
        return ImbPipeline([('pre', pre), ('smote', SMOTE(random_state=SEED)), ('clf', clf)])
    return Pipeline([('pre', pre), ('clf', make_clf(name, params, weighted))])

SCORING = {'ap': 'average_precision', 'roc': 'roc_auc', 'f1': 'f1', 'prec': 'precision', 'rec': 'recall'}

def cv_eval(pipe, Xd, yd):
    r = cross_validate(pipe, Xd, yd, cv=skf, scoring=SCORING, n_jobs=1)
    return {k: r[f'test_{k}'] for k in SCORING}

def fmt(a):
    return f'{np.mean(a):.3f} ± {np.std(a, ddof=1):.3f}'

# ============================================================
# 7. Cross-validation on the training set (default settings) and optional tuning
# ============================================================
print('\n=== 5-fold stratified CV on the training set ===')
PARAMS, CV = {}, {}
for name in MODELS:
    PARAMS[name] = {}
    CV[name] = cv_eval(build(name, F1_SET), Xtr, ytr)
    print(f'{name:20s} CV AP {fmt(CV[name]["ap"])} | ROC {fmt(CV[name]["roc"])}')

tuning_rows = []
if TUNE and HAS_XGB and HAS_LGBM:
    print('\n=== Random search (3-fold, 8 draws, 40,000-record training subsample) ===')
    sub = Xtr.sample(min(40000, len(Xtr)), random_state=SEED)
    for name in MODELS:
        rs = RandomizedSearchCV(build(name, F1_SET), {f'clf__{k}': v for k, v in SEARCH[name].items()},
                                n_iter=8, scoring='average_precision', cv=3, random_state=SEED,
                                n_jobs=1, refit=False)
        rs.fit(sub, ytr.loc[sub.index])
        tuned = {k.replace('clf__', ''): v for k, v in rs.best_params_.items()}
        cv_t = cv_eval(build(name, F1_SET, tuned), Xtr, ytr)
        keep = tuned if np.mean(cv_t['ap']) > np.mean(CV[name]['ap']) else {}
        tuning_rows.append(dict(Model=name, Before_CV_AP=np.mean(CV[name]['ap']),
                                After_CV_AP=np.mean(cv_t['ap']), Tuned_params=str(tuned), Kept=bool(keep)))
        if keep:
            PARAMS[name], CV[name] = tuned, cv_t
        print(f'{name:20s} before {np.mean(tuning_rows[-1]["Before_CV_AP"]):.3f} -> after {np.mean(cv_t["ap"]):.3f}')
    pd.DataFrame(tuning_rows).to_csv(f'{RES_DIR}/tuning_log.csv', index=False)
elif TUNE:
    print('TUNE requested but xgboost/lightgbm are missing: tuning skipped.')

# Model choice uses cross-validation on the TRAINING set, never the test set.
best_name = max(MODELS, key=lambda n: np.mean(CV[n]['ap']))
print('\nBest model by CV AP:', best_name)

# ============================================================
# 8. Final fit and test evaluation (RQ2)
# ============================================================
print('\n=== Test set evaluation ===')
fitted, proba = {}, {}
for name in MODELS:
    fitted[name] = build(name, F1_SET, PARAMS[name]).fit(Xtr, ytr)
    proba[name] = fitted[name].predict_proba(Xte)[:, 1]
yte_np = yte.values

rng = np.random.default_rng(SEED)
boot = {n: [] for n in MODELS}
for _ in range(N_BOOT):
    idx = rng.integers(0, len(yte_np), len(yte_np))
    for n in MODELS:
        boot[n].append(average_precision_score(yte_np[idx], proba[n][idx]))
boot = {n: np.array(v) for n, v in boot.items()}

rows = []
for n in MODELS:
    p, pred = proba[n], (proba[n] >= 0.5).astype(int)
    diff = boot[n] - boot['Logistic Regression']
    rows.append(dict(
        Model=n,
        PR_AUC=average_precision_score(yte_np, p),
        PR_AUC_CI=f'[{np.percentile(boot[n], 2.5):.3f}, {np.percentile(boot[n], 97.5):.3f}]',
        ROC_AUC=roc_auc_score(yte_np, p), F1=f1_score(yte_np, pred), Precision=precision_score(yte_np, pred),
        Recall=recall_score(yte_np, pred), Brier=brier_score_loss(yte_np, p),
        CV_AP=fmt(CV[n]['ap']),
        Diff_vs_LR=f'{np.mean(diff):+.3f} [{np.percentile(diff, 2.5):+.3f}, {np.percentile(diff, 97.5):+.3f}]'))
res = pd.DataFrame(rows)
res.to_csv(f'{RES_DIR}/model_results.csv', index=False)
print(res.round(3).to_string(index=False))
print(f'Chance-level PR-AUC (prevalence): {yte.mean():.4f}')
best_by_test = res.loc[res['PR_AUC'].idxmax(), 'Model']
if best_by_test != best_name:
    print(f'NOTE: best by CV is {best_name}, best on test is {best_by_test}; report the CV-based choice.')
r = res.set_index('Model').loc[best_name]
print(f'Compared with Patel et al.: F1 {r.F1:.3f} vs {PATEL["f1"]}, ROC-AUC {r.ROC_AUC:.3f} vs {PATEL["roc"]}')

print('\nLaTeX rows for Table (model comparison):')
for _, x in res.iterrows():
    print(f'{x.Model} & {x.PR_AUC:.3f} & {x.ROC_AUC:.3f} & {x.F1:.3f} & {x.Precision:.3f} & {x.Recall:.3f} \\\\')

# PR curves
plt.figure(figsize=(6.5, 5))
for n in MODELS:
    pr, rc, _ = precision_recall_curve(yte_np, proba[n])
    plt.plot(rc, pr, label=f'{n} (AP {average_precision_score(yte_np, proba[n]):.3f})')
plt.axhline(yte.mean(), color='grey', ls='--', label=f'Chance ({yte.mean():.3f})')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(fontsize=8); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/pr_curves.png', dpi=300); plt.show()

# Confusion matrix of the best model at threshold 0.5
def plot_cm(cm, title, path):
    pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(pct, cmap='Blues', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}\n({pct[i, j]:.1%})', ha='center', va='center',
                    color='white' if pct[i, j] > 0.5 else 'black')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Not at Risk', 'At Risk'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Not at Risk', 'At Risk'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(title, fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.show()

cm05 = confusion_matrix(yte_np, (proba[best_name] >= 0.5).astype(int), labels=[0, 1])
plot_cm(cm05, f'{best_name}, threshold 0.5', f'{FIG_DIR}/confusion_matrix.png')
pd.DataFrame(cm05, index=['actual_0', 'actual_1'], columns=['pred_0', 'pred_1']).to_csv(f'{RES_DIR}/confusion_matrix_best.csv')

# ============================================================
# 9. RQ1: feature importance (SHAP + permutation) for the best model
# ============================================================
print('\n=== RQ1: feature importance ===')
best_pipe = fitted[best_name]
pre_fit, clf_fit = best_pipe.named_steps['pre'], best_pipe.named_steps['clf']
names = list(pre_fit.get_feature_names_out())

def to_original(enc_name):
    s = enc_name.split('__', 1)[1]
    for nom in NOMINAL:
        if s.startswith(nom + '_'):
            return nom
    return s

shap_by_feature = None
if HAS_SHAP and best_name != 'MLP':
    Xs = Xte.sample(min(SHAP_N, len(Xte)), random_state=SEED)
    Xs_enc = pd.DataFrame(pre_fit.transform(Xs), columns=names)
    if best_name == 'Logistic Regression':
        bg = pd.DataFrame(pre_fit.transform(Xtr.sample(min(1000, len(Xtr)), random_state=SEED)), columns=names)
        masker = shap.maskers.Independent(bg.values, max_samples=100)
        sv = shap.LinearExplainer(clf_fit, masker).shap_values(Xs_enc.values)
    else:
        sv = shap.TreeExplainer(clf_fit).shap_values(Xs_enc.values)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) == 2 else sv[0]
    elif getattr(sv, 'ndim', 2) == 3:
        sv = sv[:, :, 1]
    # Group SIGNED one-hot contributions per observation before abs and mean.
    # Taking abs first spuriously counts cancelling dummy contributions.
    original_cols = list(dict.fromkeys(map(to_original, names)))
    grouped_sv = np.column_stack([
        sv[:, [to_original(n) == c for n in names]].sum(axis=1)
        for c in original_cols
    ])
    shap_by_feature = pd.Series(np.abs(grouped_sv).mean(axis=0), index=original_cols).sort_values(ascending=False)
    shap_by_feature.rename('mean_abs_shap').to_csv(f'{RES_DIR}/shap_feature_importance.csv')
    top_shap = shap_by_feature.head(15).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh([c.replace('_', ' ').capitalize() for c in top_shap.index], top_shap.values)
    plt.xlabel('Mean absolute grouped SHAP (model output units)')
    plt.tight_layout(); plt.savefig(f'{FIG_DIR}/shap_summary.png', dpi=300, bbox_inches='tight'); plt.show()
    print('Mean |SHAP| by original feature:\n', shap_by_feature.head(10).round(4))
else:
    print('SHAP skipped (shap not installed, or the best model is the MLP). Permutation importance is still computed.')

# Permutation importance per original feature (drop in PR-AUC on the test set)
Xp = Xte.sample(min(PERM_N, len(Xte)), random_state=SEED)
yp = yte.loc[Xp.index]
perm = permutation_importance(best_pipe, Xp, yp, scoring='average_precision',
                              n_repeats=PERM_REPEATS, random_state=SEED, n_jobs=1)
perm_df = pd.DataFrame({'feature': Xp.columns, 'mean': perm.importances_mean,
                        'std': perm.importances_std}).sort_values('mean', ascending=False)
perm_df.to_csv(f'{RES_DIR}/permutation_importance.csv', index=False)
top = perm_df.head(15).iloc[::-1]
plt.figure(figsize=(6.5, 5))
plt.barh(top['feature'], top['mean'], xerr=top['std'], color='#4c72b0')
plt.xlabel('Drop in PR-AUC when the feature is permuted'); plt.grid(axis='x', alpha=0.3)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/permutation_importance.png', dpi=300); plt.show()
print(perm_df.head(10).round(4).to_string(index=False))

# Category-level importance: permute all columns of a category together
def group_permutation(cols, repeats):
    base = average_precision_score(yp, best_pipe.predict_proba(Xp)[:, 1])
    g = np.random.default_rng(SEED)
    drops = []
    for _ in range(repeats):
        Xq = Xp.copy()
        order = g.permutation(len(Xq))
        Xq[cols] = Xp[cols].values[order]
        drops.append(base - average_precision_score(yp, best_pipe.predict_proba(Xq)[:, 1]))
    return np.mean(drops), np.std(drops)

cat_rows = []
total_shap = shap_by_feature.sum() if shap_by_feature is not None else np.nan
for cat, cols in CATEGORIES.items():
    d_mean, d_std = group_permutation(cols, PERM_REPEATS)
    share = shap_by_feature.reindex(cols).sum() / total_shap if shap_by_feature is not None else np.nan
    pool = shap_by_feature.reindex(cols).dropna().sort_values(ascending=False) if shap_by_feature is not None \
        else perm_df.set_index('feature').loc[cols, 'mean'].sort_values(ascending=False)
    cat_rows.append(dict(Category=cat, SHAP_share=share, Perm_drop_PR_AUC=d_mean, Perm_std=d_std,
                         Top_feature=pool.index[0]))
cat_df = pd.DataFrame(cat_rows).sort_values('Perm_drop_PR_AUC', ascending=False)
cat_df.to_csv(f'{RES_DIR}/category_importance.csv', index=False)
print(cat_df.round(4).to_string(index=False))
print('\nLaTeX rows for Table (importance by category):')
for _, x in cat_df.iterrows():
    share_txt = '--' if pd.isna(x.SHAP_share) else f'{x.SHAP_share:.1%}'.replace('%', '\\%')
    print(f'{x.Category} & {share_txt} & {x.Perm_drop_PR_AUC:.3f} & \\texttt{{{x.Top_feature.replace("_", chr(92) + "_")}}} \\\\')

# ============================================================
# 10. RQ3: remove stress, anxiety and depression (same model, same split)
# ============================================================
print('\n=== RQ3: feature-set ablation with', best_name, '===')
sets = [('F1: all features', F1_SET), ('F2: without stress, anxiety, depression', F2_SET),
        ('F3: work environment and lifestyle only', F3_SET),
        ('F4: without treatment', [c for c in ALL if c not in CATEGORIES['Treatment']]),
        ('F5: without psychological and treatment',
         [c for c in ALL if c not in CATEGORIES['Psychological'] + CATEGORIES['Treatment']])]
abl_proba, abl_rows = {}, []
for label, feats in sets:
    pipe = build(best_name, feats, PARAMS[best_name]).fit(Xtr[feats], ytr)
    abl_proba[label] = pipe.predict_proba(Xte[feats])[:, 1]
    cvr = cv_eval(build(best_name, feats, PARAMS[best_name]), Xtr[feats], ytr)
    abl_rows.append(dict(Feature_set=label, N_features=len(feats),
                         PR_AUC=average_precision_score(yte_np, abl_proba[label]),
                         ROC_AUC=roc_auc_score(yte_np, abl_proba[label]), CV_AP=fmt(cvr['ap'])))
abl = pd.DataFrame(abl_rows)
base_ap = abl.loc[0, 'PR_AUC']
abl['Delta_PR_AUC'] = abl['PR_AUC'] - base_ap
abl['Relative_change'] = abl['Delta_PR_AUC'] / base_ap

def paired_ci(pa, pb):
    g = np.random.default_rng(SEED + 1)
    d = []
    for _ in range(N_BOOT):
        idx = g.integers(0, len(yte_np), len(yte_np))
        d.append(average_precision_score(yte_np[idx], pa[idx]) - average_precision_score(yte_np[idx], pb[idx]))
    return np.percentile(d, [2.5, 97.5])

labels = [s[0] for s in sets]
abl['Drop_in_PR_AUC'] = -abl['Delta_PR_AUC']
ablation_intervals = [paired_ci(abl_proba[labels[0]], abl_proba[label]) for label in labels[1:]]
abl['Drop_CI_95'] = ['--'] + [f'[{lo:.3f}, {hi:.3f}]' for lo, hi in ablation_intervals]
abl.to_csv(f'{RES_DIR}/ablation_results.csv', index=False)
print(abl.round(3).to_string(index=False))
print('\nLaTeX rows for Table (feature sets):')
for _, x in abl.iterrows():
    print(f'{x.Feature_set} & {x.N_features} & {x.PR_AUC:.3f} & {x.ROC_AUC:.3f} & {x.Delta_PR_AUC:+.3f} \\\\')

# ============================================================
# 11. Supplementary: imbalance handling and decision threshold (best model)
# ============================================================
print('\n=== Supplementary: imbalance strategies (5-fold CV on training set) ===')
imb_rows = []
for label, kw in [('No correction', dict(weighted=False)), ('Class weights', dict(weighted=True)),
                  ('SMOTE', dict(smote=True))]:
    if kw.get('smote') and not HAS_IMB:
        print('SMOTE skipped: imbalanced-learn is not installed')
        continue
    cvr = cv_eval(build(best_name, F1_SET, PARAMS[best_name], **kw), Xtr, ytr)
    imb_rows.append(dict(Strategy=label, PR_AUC=fmt(cvr['ap']), Recall=fmt(cvr['rec']), Precision=fmt(cvr['prec'])))
imb = pd.DataFrame(imb_rows)
imb.to_csv(f'{RES_DIR}/imbalance_results.csv', index=False)
print(imb.to_string(index=False))

# Threshold chosen on out-of-fold TRAINING predictions, then evaluated once on the test set
oof = cross_val_predict(build(best_name, F1_SET, PARAMS[best_name]), Xtr, ytr, cv=skf,
                        method='predict_proba', n_jobs=1)[:, 1]
pr, rc, th = precision_recall_curve(ytr, oof)
f1c = 2 * pr * rc / (pr + rc + 1e-12)
t_f1 = th[np.argmax(f1c[:-1])]
t_rec = th[np.where(rc[:-1] >= 0.80)[0][-1]]
plt.figure(figsize=(6.5, 4.5))
plt.plot(th, pr[:-1], label='Precision'); plt.plot(th, rc[:-1], label='Recall'); plt.plot(th, f1c[:-1], label='F1')
plt.axvline(t_f1, color='grey', ls='--', label=f'F1-optimal ({t_f1:.2f})')
plt.axvline(t_rec, color='red', ls=':', label=f'Recall >= 80% ({t_rec:.2f})')
plt.xlabel('Decision threshold (out-of-fold, training set)'); plt.legend(fontsize=8); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/threshold_curve.png', dpi=300); plt.show()
for label, t in [('Default 0.5', 0.5), ('F1-optimal', t_f1), ('Recall >= 80%', t_rec)]:
    pred = (proba[best_name] >= t).astype(int)
    print(f'{label:15s} t={t:.2f} | test precision {precision_score(yte_np, pred):.3f} | '
          f'recall {recall_score(yte_np, pred):.3f} | F1 {f1_score(yte_np, pred):.3f}')

print('\nDone. Figures in', FIG_DIR, '| tables in', RES_DIR)
print('Still to make by hand: figures/pipeline.png (diagram) and figures/dashboard.png (screenshot).')
