"""Auditable corrections to the saved Group 15 experiment.

Run with .venv/Scripts/python.exe revision_analysis.py. The five-model benchmark
is verified and retained; only changed analyses are refitted/recomputed. Original
artifacts are preserved in revision_20261019/before_revision.zip.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import hashlib, json, sys, time, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'revision_20261019'
RES = OUT / 'results'
FIG = OUT / 'figures'
MODELS = OUT / 'models'
for p in [RES, FIG, MODELS]: p.mkdir(parents=True, exist_ok=True)
SEED = 42
N_BOOT = int(os.environ.get('REVISION_BOOTSTRAPS', '2000'))
CATEGORIES = {
    'Work environment': ['job_role','company_size','work_mode','work_hours_per_week','overtime_hours',
                         'meetings_per_day','deadlines_missed','job_satisfaction','manager_support',
                         'work_life_balance','social_support_score'],
    'Lifestyle': ['sleep_hours','physical_activity_days','screen_time_hours','caffeine_intake'],
    'Psychological': ['stress_level','anxiety_score','depression_score'],
    'Demographic': ['age','gender','experience_years'],
    'Treatment': ['has_therapy','seeks_professional_help'],
}
NOMINAL = ['gender','job_role','company_size','work_mode']
BINARY = CATEGORIES['Treatment']
FEATURES = [f for v in CATEGORIES.values() for f in v]
LABELS = {f: f.replace('_',' ').capitalize() for f in FEATURES}
LABELS.update(work_hours_per_week='Weekly working hours', work_life_balance='Work-life balance',
              has_therapy='Therapy status', seeks_professional_help='Professional help',
              work_mode='Work mode', gender='Gender', physical_activity_days='Physical activity')

def preprocessor(features):
    return ColumnTransformer([
        ('num', StandardScaler(), [c for c in features if c not in NOMINAL+BINARY]),
        ('bin','passthrough',[c for c in BINARY if c in features]),
        ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),[c for c in NOMINAL if c in features]),
    ])

def lr_pipeline(features):
    return Pipeline([('pre',preprocessor(features)),('clf',LogisticRegression(C=1.,class_weight='balanced',max_iter=2000))])

def original_name(name):
    s = name.split('__',1)[1]
    return next((n for n in NOMINAL if s.startswith(n+'_')), s)

def aggregate_signed_shap(values, names):
    """Sum encoded SHAP values per record/original variable BEFORE abs/mean."""
    cols = list(dict.fromkeys(map(original_name,names)))
    grouped = np.column_stack([values[:,[original_name(n)==c for n in names]].sum(axis=1) for c in cols])
    return cols, grouped

def ap_bootstrap_setup(y, scores):
    order = np.argsort(-scores,kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(scores[order]) != 0),len(scores)-1]
    return order, ends, y[order]

def weighted_ap_from_counts(counts, setup):
    order, ends, ys = setup
    c = counts[order]
    tp = np.cumsum(c*ys)[ends]
    total = np.cumsum(c)[ends]
    inc = np.diff(np.r_[0.,tp])
    return float(np.sum(np.divide(tp,total,out=np.zeros_like(tp,dtype=float),where=total>0)*inc)/tp[-1])

def plot_bars(series, path, xlabel, title, n=12):
    s=series.nlargest(n).iloc[::-1]
    fig,ax=plt.subplots(figsize=(7,4.7))
    ax.barh([LABELS.get(x,x) for x in s.index],s.values,color='#294662')
    ax.set(xlabel=xlabel,title=title)
    ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(path,dpi=220,bbox_inches='tight');plt.close(fig)

def main():
    start=time.time()
    manifest={'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running','seed':SEED,'n_bootstrap':N_BOOT,
              'data_sha256':hashlib.sha256((ROOT/'tech_mental_health_burnout.csv').read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'retained_benchmark':'Five fitted pipelines and saved test/CV/OOF predictions; verified before changes',
              'packages':{m.__name__:m.__version__ for m in [np,pd,shap]},'python':sys.version}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    data=pd.read_csv(ROOT/'tech_mental_health_burnout.csv')
    X=data[FEATURES]; y=(data.burnout_level!='Low').astype(int)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
    saved=pd.read_csv(ROOT/'training_results/results/test_predictions.csv')
    assert np.array_equal(Xte.index,saved.row_index)
    assert np.array_equal(yte,saved.actual)
    pipe=joblib.load(ROOT/'training_results/models/logistic_regression.joblib')
    np.testing.assert_allclose(pipe.predict_proba(Xte)[:,1],saved['Logistic Regression'],atol=1e-12,rtol=0)
    # Full effective parameter dumps from fitted estimators, not assumed defaults.
    params={}
    for name in ['Logistic Regression','Random Forest','XGBoost','LightGBM','MLP']:
        if name=='XGBoost':
            from xgboost import XGBClassifier
            clf=XGBClassifier();clf.load_model(ROOT/'training_results/models/xgboost_native.json')
            # Native JSON preserves trees, but not all sklearn training kwargs.
            # Recover configured values from the archived, hashed training source.
            params[name]={'n_estimators':300,'max_depth':3,'learning_rate':.05,
                          'subsample':.8,'colsample_bytree':.8,'reg_lambda':5,
                          'min_child_weight':10,'scale_pos_weight':float((ytr==0).sum()/(ytr==1).sum()),
                          'objective':'binary:logistic','eval_metric':'logloss',
                          'n_jobs':-1,'random_state':42,
                          'parameter_provenance':'archived original training source; native-model probabilities verified'}
            continue
        else:
            clf=joblib.load(ROOT/'training_results/models'/(name.lower().replace(' ','_')+'.joblib')).named_steps['clf']
        params[name]=clf.get_params()
    (RES/'hyperparameters.json').write_text(json.dumps(params,indent=2,default=str))
    print('Verified split, saved LR probabilities and estimator parameters.',flush=True)

    # Same background and test sample as original; grouping is the changed operation.
    pre,clf=pipe.named_steps['pre'],pipe.named_steps['clf']
    sample=Xte.sample(2000,random_state=SEED)
    enc=pre.transform(sample)
    bg=pre.transform(Xtr.sample(1000,random_state=SEED))
    names=list(pre.get_feature_names_out())
    explainer=shap.LinearExplainer(clf,bg)
    sv=np.asarray(explainer.shap_values(enc))
    cols,grouped=aggregate_signed_shap(sv,names)
    np.testing.assert_allclose(grouped.sum(axis=1)+explainer.expected_value,pipe.decision_function(sample),atol=1e-10)
    # Independent linear-model formula and a cancellation regression check.
    expected=(enc-explainer.masker.data.mean(axis=0))*clf.coef_[0]
    np.testing.assert_allclose(sv,expected,atol=1e-12)
    tc,tv=aggregate_signed_shap(np.array([[2.,-2.,3.],[-1.,1.,-4.]]),['cat__gender_A','cat__gender_B','num__age'])
    assert np.allclose(tv[:,tc.index('gender')],0.)
    correct=pd.Series(np.abs(grouped).mean(axis=0),index=cols).sort_values(ascending=False)
    old=pd.Series(np.abs(sv).mean(axis=0),index=[original_name(n) for n in names]).groupby(level=0).sum()
    comparison=pd.DataFrame({'corrected_mean_abs_shap':correct,'previous_sum_of_abs_shap':old})
    comparison['difference']=comparison.corrected_mean_abs_shap-comparison.previous_sum_of_abs_shap
    comparison.to_csv(RES/'shap_aggregation_audit.csv')
    correct.rename('mean_abs_shap').to_csv(RES/'shap_feature_importance.csv')
    np.savez_compressed(RES/'grouped_shap_values.npz',row_index=sample.index.to_numpy(),features=np.array(cols),values=grouped,
                        expected_value=explainer.expected_value,background_indices=Xtr.sample(1000,random_state=SEED).index.to_numpy())
    plot_bars(correct,FIG/'shap_summary.png','Mean absolute grouped SHAP (log-odds)','Corrected original-feature SHAP')
    print('SHAP correction:',correct.head(8).to_dict(),flush=True)

    xp=Xte.sample(10000,random_state=SEED); yp=yte.loc[xp.index]
    pi=permutation_importance(pipe,xp,yp,scoring='average_precision',n_repeats=5,random_state=SEED,n_jobs=1)
    perm=pd.DataFrame({'feature':FEATURES,'mean':pi.importances_mean,'std':pi.importances_std}).sort_values('mean',ascending=False)
    perm.to_csv(RES/'permutation_importance.csv',index=False)
    plot_bars(perm.set_index('feature')['mean'],FIG/'permutation_importance.png','Drop in average precision','Original-feature permutation importance')
    catrows=[]; base=average_precision_score(yp,pipe.predict_proba(xp)[:,1])
    for cat,features in CATEGORIES.items():
        rng=np.random.default_rng(SEED);drops=[]
        for _ in range(5):
            q=xp.copy();q[features]=xp[features].values[rng.permutation(len(xp))]
            drops.append(base-average_precision_score(yp,pipe.predict_proba(q)[:,1]))
        catrows.append({'Category':cat,'SHAP_share':correct.reindex(features).sum()/correct.sum(),
                        'Perm_drop_PR_AUC':np.mean(drops),'Perm_std':np.std(drops),
                        'Top_feature':correct.reindex(features).idxmax()})
    cats=pd.DataFrame(catrows);cats.to_csv(RES/'category_importance.csv',index=False)
    print('Category results:',cats.to_dict('records'),flush=True)

    sets={'F1: all features':FEATURES,
          'F2: without psychological':[f for f in FEATURES if f not in CATEGORIES['Psychological']],
          'F3: work environment and lifestyle':CATEGORIES['Work environment']+CATEGORIES['Lifestyle'],
          'F4: without treatment':[f for f in FEATURES if f not in CATEGORIES['Treatment']],
          'F5: without psychological and treatment':[f for f in FEATURES if f not in CATEGORIES['Treatment']+CATEGORIES['Psychological']]}
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    preds={}; rows=[];foldrows=[]
    for label,features in sets.items():
        model=lr_pipeline(features).fit(Xtr[features],ytr)
        prob=model.predict_proba(Xte[features])[:,1];preds[label]=prob
        scores=cross_val_score(lr_pipeline(features),Xtr[features],ytr,cv=cv,scoring='average_precision',n_jobs=1)
        joblib.dump(model,MODELS/(label[:2]+'.joblib'),compress=3)
        rows.append({'Feature_set':label,'N_features':len(features),'PR_AUC':average_precision_score(yte,prob),
                     'ROC_AUC':roc_auc_score(yte,prob),'CV_AP_mean':scores.mean(),'CV_AP_std':scores.std(ddof=1)})
        foldrows.extend({'Feature_set':label,'fold':i+1,'AP':s} for i,s in enumerate(scores))
        print('Ablation',rows[-1],flush=True)
    np.testing.assert_allclose(preds['F1: all features'],saved['Logistic Regression'],atol=1e-10,rtol=0)
    original_abl=pd.read_csv(ROOT/'training_results/results/ablation_results.csv')
    for i in range(3): assert abs(rows[i]['PR_AUC']-original_abl.iloc[i].PR_AUC)<1e-10
    pd.DataFrame(foldrows).to_csv(RES/'ablation_cv_folds.csv',index=False)
    pd.DataFrame({'row_index':Xte.index,'actual':yte.values,**preds}).to_csv(RES/'ablation_predictions.csv',index=False)

    # Paired ordinary bootstrap: multinomial multiplicities implement exact row
    # resampling; grouped score ties match sklearn average_precision_score.
    all_probs={name:saved[name].to_numpy() for name in ['Logistic Regression','Random Forest','XGBoost','LightGBM','MLP']}
    all_probs.update(preds)
    yy=yte.to_numpy();setups={n:ap_bootstrap_setup(yy,p) for n,p in all_probs.items()}
    rng=np.random.default_rng(SEED+1)
    # Validate the fast estimator against literal bootstrap rows, including ties.
    for _ in range(3):
        ix=rng.integers(0,len(yy),len(yy));counts=np.bincount(ix,minlength=len(yy))
        for n,p in all_probs.items():
            assert abs(weighted_ap_from_counts(counts,setups[n])-average_precision_score(yy[ix],p[ix]))<1e-12
    rng=np.random.default_rng(SEED+1)
    boots={n:np.empty(N_BOOT) for n in all_probs}
    for b in range(N_BOOT):
        ix=rng.integers(0,len(yy),len(yy));counts=np.bincount(ix,minlength=len(yy))
        for n,setup in setups.items(): boots[n][b]=weighted_ap_from_counts(counts,setup)
        if (b+1)%500==0: print('Bootstrap',b+1,'/',N_BOOT,flush=True)
    bootframe=pd.DataFrame(boots);bootframe.to_csv(RES/'bootstrap_replicates.csv',index=False)
    metrics=pd.read_csv(ROOT/'training_results/results/model_results.csv')
    for i,row in metrics.iterrows():
        n=row.Model;lo,hi=np.quantile(boots[n],[.025,.975]);metrics.loc[i,'PR_AUC_CI']=f'[{lo:.3f}, {hi:.3f}]'
        d=boots[n]-boots['Logistic Regression'];dl,dh=np.quantile(d,[.025,.975])
        metrics.loc[i,'Diff_vs_LR']=f'{row.PR_AUC-metrics.iloc[0].PR_AUC:+.3f} [{dl:+.3f}, {dh:+.3f}]'
        metrics.loc[i,'AP_CI_low']=lo;metrics.loc[i,'AP_CI_high']=hi
    metrics.to_csv(RES/'model_results.csv',index=False)
    abl=pd.DataFrame(rows);base_ap=abl.iloc[0].PR_AUC
    for i,row in abl.iterrows():
        lo,hi=np.quantile(boots[row.Feature_set],[.025,.975])
        d=boots['F1: all features']-boots[row.Feature_set];dl,dh=np.quantile(d,[.025,.975])
        abl.loc[i,'AP_CI_low']=lo;abl.loc[i,'AP_CI_high']=hi
        abl.loc[i,'Drop_in_PR_AUC']=base_ap-row.PR_AUC
        abl.loc[i,'Relative_drop']=(base_ap-row.PR_AUC)/base_ap
        abl.loc[i,'Drop_CI_low']=dl;abl.loc[i,'Drop_CI_high']=dh
        abl.loc[i,'CV_AP']=f'{row.CV_AP_mean:.3f} ± {row.CV_AP_std:.3f}'
        abl.loc[i,'Drop_CI_95']=f'[{dl:.3f}, {dh:.3f}]'
    abl.to_csv(RES/'ablation_results.csv',index=False)
    # Exact low-stress counts distinguish rounded 0.0% from observed zeros.
    low=data.stress_level<4
    low_result={'n':int(low.sum()),'at_risk':int(y[low].sum()),'max_burnout_score':float(data.loc[low,'burnout_score'].max()),
                'minimum_high_or_moderate_score':float(data.loc[y==1,'burnout_score'].min())}
    (RES/'low_stress_check.json').write_text(json.dumps(low_result,indent=2))
    manifest.update(status='complete',elapsed_seconds=time.time()-start,train_records=len(Xtr),test_records=len(Xte),
                    corrected_shap_additivity=True,fast_bootstrap_matches_sklearn=True,
                    original_F1_F2_F3_AP_reproduced=True,completed_utc=datetime.now(timezone.utc).isoformat())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('COMPLETE',abl.to_string(index=False),low_result,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2): main()
