"""Check saved Colab artifacts without fitting or changing any model."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score, brier_score_loss

root = Path(__file__).resolve().parent
out = root / 'training_results'
manifest = json.loads((out / 'run_manifest.json').read_text())
assert manifest['status'] == 'complete' and manifest['source_unchanged']
original_source = root / manifest.get('analysis_revision', {}).get('original_source_path', 'burnout_pipeline.py')
assert hashlib.sha256(original_source.read_bytes()).hexdigest() == manifest['source_sha256']
if 'analysis_revision' in manifest:
    assert hashlib.sha256((root/'burnout_pipeline.py').read_bytes()).hexdigest() == manifest['analysis_revision']['current_source_sha256']
assert hashlib.sha256((root / 'tech_mental_health_burnout.csv').read_bytes()).hexdigest() == manifest['data_sha256']
preds = pd.read_csv(out / 'results/test_predictions.csv')
metrics = pd.read_csv(out / 'results/model_results.csv').set_index('Model')
df = pd.read_csv(root / 'tech_mental_health_burnout.csv')
assert len(preds) == manifest['test_records'] == 30000
assert preds['row_index'].is_unique
assert np.array_equal(preds['actual'], (df.loc[preds['row_index'], 'burnout_level'] != 'Low').astype(int))
train = pd.read_csv(out / 'results/oof_best_model.csv')
assert len(train) == manifest['train_records'] == 120000
assert not set(train['row_index']) & set(preds['row_index'])
checks = {}
for name in metrics.index:
    p = preds[name].values
    y = preds['actual'].values
    lab = p >= 0.5
    calculated = {'PR_AUC': average_precision_score(y, p), 'ROC_AUC': roc_auc_score(y, p),
                  'F1': f1_score(y, lab), 'Precision': precision_score(y, lab),
                  'Recall': recall_score(y, lab), 'Brier': brier_score_loss(y, p)}
    for metric, value in calculated.items():
        # XGBoost probabilities are float32; CSV reload converts them to float64.
        assert np.isclose(value, metrics.loc[name, metric], atol=1e-7, rtol=0), (name, metric)
    if name == 'XGBoost':
        from xgboost import XGBClassifier
        from sklearn.pipeline import Pipeline
        clf = XGBClassifier()
        clf.load_model(out / 'models/xgboost_native.json')
        model = Pipeline([('pre', joblib.load(out / 'models/xgboost_preprocessor.joblib')), ('clf', clf)])
    else:
        model = joblib.load(out / 'models' / (name.lower().replace(' ', '_') + '.joblib'))
    rows = preds.head(100)
    sample = df.loc[rows['row_index'], list(model.feature_names_in_)]
    error = float(np.max(np.abs(model.predict_proba(sample)[:, 1] - rows[name].values)))
    assert error < 1e-7, (name, error)
    checks[name] = {'metrics_match': True, 'saved_model_max_probability_error_100_test_rows': error}
folds = pd.read_csv(out / 'results/cv_fold_results.csv')
assert len(folds) == 25 and (folds.groupby('Model').size() == 5).all()
report = {'status': 'passed', 'source_and_data_hashes_match': True, 'test_records': len(preds),
          'train_records': len(train), 'train_test_disjoint': True, 'cv_fold_records': len(folds), 'models': checks}
(out / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
