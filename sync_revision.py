"""Promote reviewed numerical outputs and regenerate consistent local artifacts."""
from pathlib import Path
import json, shutil, hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parent
REV=ROOT/'revision_20261019'

def main():
    sql=ROOT/'sql_analysis/sql_queries.sql'
    s=sql.read_text(encoding='utf-8')
    s=s.replace("'Workplace'", "'Work environment'").replace("'social_support_score','Lifestyle'", "'social_support_score','Work environment'")
    sql.write_text(s,encoding='utf-8')
    for name in ['model_results.csv','shap_feature_importance.csv','category_importance.csv',
                 'permutation_importance.csv','ablation_results.csv']:
        shutil.copy2(REV/'results'/name,ROOT/'training_results/results'/name)
    for name in ['shap_summary.png','permutation_importance.png']:
        shutil.copy2(REV/'figures'/name,ROOT/'training_results/figures'/name)
    m=json.loads((ROOT/'training_results/run_manifest.json').read_text())
    m['analysis_revision']={'date':'2026-10-19','manifest':'revision_20261019/manifest.json',
                            'original_source_path':'training_results/reproducibility/burnout_pipeline.py',
                            'current_source_sha256':hashlib.sha256((ROOT/'burnout_pipeline.py').read_bytes()).hexdigest(),
                            'note':'Original model fit retained; current source has corrected SHAP and added ablations. Revised AP intervals use 2,000 paired row bootstraps.'}
    (ROOT/'training_results/run_manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    d=json.loads((ROOT/'dashboard/data.json').read_text(encoding='utf-8'))
    d['models']=pd.read_csv(REV/'results/model_results.csv').sort_values('PR_AUC',ascending=False).to_dict('records')
    sh=pd.read_csv(REV/'results/shap_feature_importance.csv',index_col=0).iloc[:,0]
    d['shap']=[{'name':k,'value':float(v)} for k,v in sh.items()]
    d['categories']=pd.read_csv(REV/'results/category_importance.csv').to_dict('records')
    d['ablation']=pd.read_csv(REV/'results/ablation_results.csv').to_dict('records')
    d['revision']={'date':'2026-10-19','bootstrap_resamples':2000,'shap_aggregation':'sum signed dummy SHAP within each observation, then absolute value and mean',
                   'explanation_units':'log-odds','shap_test_records':2000,'candidate_background':1000,'effective_background':100,
                   'source':'revision_20261019/results','dataset':'Employee Mental Health & Burnout Dataset'}
    (ROOT/'dashboard/data.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Promoted corrected numerical outputs and dashboard data.')

if __name__=='__main__':main()
