"""Independent scientific and cross-artifact checks for the revised project."""
import hashlib,json,re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from revision_analysis import ROOT,OUT,RES,CATEGORIES


def verify_independent(pred, abl, boots):
    """Check the saved artifacts from source data and literal sklearn calculations.

    This does not refit estimators or reuse the production SHAP/bootstrap helpers.
    """
    import joblib
    import shap
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, brier_score_loss
    source=pd.read_csv(ROOT/'tech_mental_health_burnout.csv')
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert hashlib.sha256((ROOT/'tech_mental_health_burnout.csv').read_bytes()).hexdigest()==manifest['data_sha256']
    assert source.shape==(150000,25) and not source.isna().any().any() and not source.duplicated().any()
    # Independent feature-set specification rather than importing production F1--F5.
    work=['job_role','company_size','work_mode','work_hours_per_week','overtime_hours',
          'meetings_per_day','deadlines_missed','job_satisfaction','manager_support',
          'work_life_balance','social_support_score']
    lifestyle=['sleep_hours','physical_activity_days','screen_time_hours','caffeine_intake']
    psychological=['stress_level','anxiety_score','depression_score']
    demographics=['age','gender','experience_years']
    treatment=['has_therapy','seeks_professional_help']
    features=work+lifestyle+psychological+demographics+treatment
    assert set(features)==set(source)-{'burnout_score','burnout_level'}
    target=(source.burnout_level!='Low').astype(int)
    Xtr,Xte,ytr,yte=train_test_split(source[features],target,test_size=.2,stratify=target,random_state=42)
    saved=pd.read_csv(ROOT/'training_results/results/test_predictions.csv')
    oof=pd.read_csv(ROOT/'training_results/results/oof_best_model.csv')
    for frame in [pred,saved]:
        assert frame.row_index.is_unique and len(frame)==30000
        np.testing.assert_array_equal(frame.row_index,Xte.index)
        np.testing.assert_array_equal(frame.actual,yte)
    assert oof.row_index.is_unique and len(oof)==120000
    np.testing.assert_array_equal(oof.row_index,Xtr.index)
    np.testing.assert_array_equal(oof.actual,ytr)
    assert not set(oof.row_index)&set(pred.row_index)
    sets={'F1: all features':features,
          'F2: without psychological':[f for f in features if f not in psychological],
          'F3: work environment and lifestyle':work+lifestyle,
          'F4: without treatment':[f for f in features if f not in treatment],
          'F5: without psychological and treatment':[f for f in features if f not in psychological+treatment]}
    assert set(abl.Feature_set)==set(sets)
    cv=pd.read_csv(RES/'ablation_cv_folds.csv')
    close=lambda a,b,atol=1e-12:np.testing.assert_allclose(a,b,atol=atol,rtol=0)
    for row in abl.itertuples():
        fs=sets[row.Feature_set]
        pipe=joblib.load(OUT/'models'/(row.Feature_set[:2]+'.joblib'))
        assert list(pipe.feature_names_in_)==fs and row.N_features==len(fs)
        clf=pipe.named_steps['clf']
        assert clf.C==1 and clf.class_weight=='balanced' and clf.max_iter==2000
        close(pipe.predict_proba(Xte[fs])[:,1],pred[row.Feature_set])
        close(abl.iloc[0].PR_AUC-row.PR_AUC,row.Drop_in_PR_AUC)
        close(row.Drop_in_PR_AUC/abl.iloc[0].PR_AUC,row.Relative_drop)
        fold=cv.loc[cv.Feature_set==row.Feature_set,'AP'];assert len(fold)==5
        close([fold.mean(),fold.std(ddof=1)],[row.CV_AP_mean,row.CV_AP_std])
    close(saved['Logistic Regression'],pred['F1: all features'])
    models=['Logistic Regression','Random Forest','XGBoost','LightGBM','MLP']
    scores=pd.read_csv(RES/'model_results.csv').set_index('Model')
    folds=pd.read_csv(ROOT/'training_results/results/cv_fold_results.csv')
    assert folds.groupby('Model').ap.mean().idxmax()=='Logistic Regression'
    for name in models:
        row=scores.loc[name];p=saved[name];label=p>=.5
        close([average_precision_score(yte,p),roc_auc_score(yte,p),f1_score(yte,label),
               precision_score(yte,label),recall_score(yte,label),brier_score_loss(yte,p)],
              row[['PR_AUC','ROC_AUC','F1','Precision','Recall','Brier']].astype(float),atol=1e-7)
        close(np.quantile(boots[name],[.025,.975]),[row.AP_CI_low,row.AP_CI_high])
        lo,hi=np.quantile(boots[name]-boots['Logistic Regression'],[.025,.975])
        delta=row.PR_AUC-scores.loc['Logistic Regression','PR_AUC']
        assert row.Diff_vs_LR==f'{delta:+.3f} [{lo:+.3f}, {hi:+.3f}]'
    # Regenerate row samples and check selected saved bootstrap replicates literally.
    # This independently tests RNG, pairing and AP (including tied scores).
    probabilities={name:saved[name].to_numpy() for name in models}
    probabilities.update({name:pred[name].to_numpy() for name in sets})
    rng=np.random.default_rng(43);checked=[0,1,2,499,1999];yy=yte.to_numpy()
    for b in range(len(boots)):
        ix=rng.integers(0,len(yy),len(yy))
        if b in checked:
            for name,p in probabilities.items():close(average_precision_score(yy[ix],p[ix]),boots.loc[b,name])
    pr,rc,th=precision_recall_curve(oof.actual,oof.oof_probability)
    f1=2*pr*rc/(pr+rc+1e-12)
    thresholds={'Default 0.5':.5,'F1-optimal':th[np.argmax(f1[:-1])],
                'Recall >= 80%':th[np.where(rc[:-1]>=.8)[0][-1]]}
    for row in pd.read_csv(ROOT/'training_results/results/threshold_results.csv').itertuples():
        close(row.Threshold,thresholds[row.Strategy]);flag=saved['Logistic Regression']>=row.Threshold
        close([precision_score(yte,flag),recall_score(yte,flag),f1_score(yte,flag)],
              [row.Precision,row.Recall,row.F1])
    pipe=joblib.load(ROOT/'training_results/models/logistic_regression.joblib')
    pre,clf=pipe.named_steps['pre'],pipe.named_steps['clf']
    sample=Xte.sample(2000,random_state=42);candidates=Xtr.sample(1000,random_state=42)
    phi=np.load(RES/'grouped_shap_values.npz')
    np.testing.assert_array_equal(phi['row_index'],sample.index)
    np.testing.assert_array_equal(phi['background_indices'],candidates.index)
    enc=pre.transform(sample)
    # Historical runner used the Independent masker's default of 100. Specify it
    # explicitly here and reproduce the fitted model's explanation from scratch.
    masker=shap.maskers.Independent(pre.transform(candidates),max_samples=100)
    assert len(masker.data)==100
    explainer=shap.LinearExplainer(clf,masker)
    signed=(enc-masker.data.mean(axis=0))*clf.coef_[0]
    close(explainer.shap_values(enc),signed)
    close(phi['expected_value'],explainer.expected_value)
    names=list(pre.get_feature_names_out())
    original=[next((f for f in features if n.split('__',1)[1]==f or
                    n.split('__',1)[1].startswith(f+'_')),None) for n in names]
    assert None not in original
    grouped=np.column_stack([signed[:,np.array(original)==f].sum(axis=1) for f in phi['features']])
    close(grouped,phi['values'])
    close(grouped.sum(axis=1)+phi['expected_value'],pipe.decision_function(sample),atol=1e-10)
    audit=pd.read_csv(RES/'shap_aggregation_audit.csv',index_col=0)
    old=pd.Series(np.abs(signed).mean(axis=0),index=original).groupby(level=0).sum()
    close(audit.previous_sum_of_abs_shap,old.reindex(audit.index))
    low=json.loads((RES/'low_stress_check.json').read_text());mask=source.stress_level<4
    assert low['n']==int(mask.sum()) and low['at_risk']==int(target[mask].sum())
    close(low['max_burnout_score'],source.loc[mask,'burnout_score'].max())
    close(low['minimum_high_or_moderate_score'],source.loc[target==1,'burnout_score'].min())
    dashboard=json.loads((ROOT/'dashboard/data.json').read_text(encoding='utf-8'))
    html=(ROOT/'dashboard/burnout-dashboard.html').read_text(encoding='utf-8')
    embedded=re.search(r'<script id="research-data" type="application/json">(.*?)</script>',html,re.S)
    assert embedded and json.loads(embedded.group(1))==dashboard
    for key,frame,id_column in [('models',scores.reset_index(),'Model'),
                                 ('ablation',abl,'Feature_set'),
                                 ('categories',pd.read_csv(RES/'category_importance.csv'),'Category')]:
        canonical=frame.set_index(id_column)
        assert {row[id_column] for row in dashboard[key]}==set(canonical.index)
        for row in dashboard[key]:
            expected=canonical.loc[row[id_column]]
            assert set(row)==set(frame.columns)
            for column,value in row.items():
                if column==id_column:continue
                if isinstance(value,(int,float)):close(value,expected[column])
                else:assert value==expected[column]
    return {'data_hash_and_exact_split_verified':True,'saved_ablation_estimators_verified':True,
            'literal_bootstrap_replicates_checked':checked,'model_comparison_intervals_verified':True,
            'training_OOF_thresholds_verified':True,'SHAP_local_additivity_verified':True,
            'SHAP_effective_background_records':len(masker.data),'SHAP_candidate_background_records':len(candidates),
            'dashboard_all_result_fields_verified':True,'dashboard_embedded_JSON_matches_file':True}

def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['status']=='complete' and manifest['n_bootstrap']==2000
    assert hashlib.sha256((ROOT/'revision_analysis.py').read_bytes()).hexdigest()==manifest['script_sha256']
    pred=pd.read_csv(RES/'ablation_predictions.csv');abl=pd.read_csv(RES/'ablation_results.csv')
    boots=pd.read_csv(RES/'bootstrap_replicates.csv')
    assert len(pred)==30000 and len(boots)==2000 and len(abl)==5
    for _,row in abl.iterrows():
        name=row.Feature_set
        assert abs(average_precision_score(pred.actual,pred[name])-row.PR_AUC)<1e-12
        assert abs(roc_auc_score(pred.actual,pred[name])-row.ROC_AUC)<1e-12
        lo,hi=np.quantile(boots[name],[.025,.975])
        np.testing.assert_allclose([lo,hi],[row.AP_CI_low,row.AP_CI_high],atol=1e-12)
        drop=boots['F1: all features']-boots[name]
        np.testing.assert_allclose(np.quantile(drop,[.025,.975]),[row.Drop_CI_low,row.Drop_CI_high],atol=1e-12)
    sh=pd.read_csv(RES/'shap_feature_importance.csv',index_col=0).iloc[:,0]
    cats=pd.read_csv(RES/'category_importance.csv')
    assert abs(cats.SHAP_share.sum()-1)<1e-12
    for row in cats.itertuples():assert abs(row.SHAP_share-sh.reindex(CATEGORIES[row.Category]).sum()/sh.sum())<1e-12
    assert sh['work_mode']<.021 and sh['gender']<.027
    audit=pd.read_csv(RES/'shap_aggregation_audit.csv',index_col=0)
    assert audit.loc['work_mode','previous_sum_of_abs_shap']>.63
    phi=np.load(RES/'grouped_shap_values.npz')
    np.testing.assert_allclose(np.abs(phi['values']).mean(axis=0),sh.reindex(phi['features']),atol=1e-12)
    data=json.loads((ROOT/'dashboard/data.json').read_text(encoding='utf-8'))
    np.testing.assert_allclose([x['value'] for x in data['shap']],sh.values,atol=1e-12)
    assert data['revision']['bootstrap_resamples']==2000 and len(data['ablation'])==5
    q=pd.read_csv(ROOT/'sql_analysis/query_2_results.csv')
    assert q.loc[q.feature=='social_support_score','group_name'].item()=='Work environment'
    low=json.loads((RES/'low_stress_check.json').read_text());assert low['n']==12749 and low['at_risk']==0
    src=(ROOT/'paper_revised/main.tex').read_text(encoding='utf-8')
    assert '55.08' in src and '32.16' in src and r'\Phi_{ij}' in src
    assert not any(x in src for x in ['\\todo{','\\guide{','Dashboard (planned','num\\_\\_','cat\\_\\_'])
    refids=re.findall(r'\\bibitem\{([^}]+)\}',src);seen=[]
    for group in re.findall(r'\\cite\{([^}]+)\}',src):
        for v in group.split(','):
            if v not in seen:seen.append(v)
    assert refids==seen
    report={'status':'passed','bootstrap_resamples':len(boots),'ablation_configurations':len(abl),
            'ablation_training_fits':30,'prediction_metrics_recomputed':True,'paired_intervals_verified':True,
            'signed_grouped_SHAP_verified':True,'corrected_category_shares_sum_to_one':True,
            'SQL_category_consistent':True,'dashboard_matches_results':True,'references_in_first_citation_order':True,
            'references':len(refids),'retained_limitations':['synthetic data','no temporal/external/subgroup validation',
                    'retrospective ablations','unweighted MLP reference','conditional bootstrap; no repeated CV']}
    report.update(verify_independent(pred,abl,boots))
    # A packaged verification result must identify the exact source and generated
    # artifacts it checked, so a later manuscript/dashboard rebuild is detectable.
    tracked=[ROOT/'verify_revision.py',ROOT/'build_revision_artifacts.py',ROOT/'revision_analysis.py',
             ROOT/'paper_revised/main.tex',ROOT/'tech_mental_health_burnout.csv',
             ROOT/'dashboard/data.json',ROOT/'dashboard/burnout-dashboard.html',
             OUT/'manifest.json',RES/'grouped_shap_values.npz']
    tracked+=sorted(RES.glob('*.csv'))
    tracked+=sorted((ROOT/'training_results/results').glob('*.csv'))
    report['checked_at_utc']=datetime.now(timezone.utc).isoformat()
    report['checked_artifact_sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in tracked}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
