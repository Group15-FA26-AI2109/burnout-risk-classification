"""Execute the supplied pipeline unchanged and preserve its trained artifacts."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'training_run'
OUT.mkdir(exist_ok=True)
os.environ.update(DATA_PATH=str(ROOT / 'tech_mental_health_burnout.csv'),
                  OUT_ROOT=str(OUT), QUICK='0', TUNE='0', MPLBACKEND='Agg')

class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log
    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        self.log.flush()
    def flush(self):
        self.console.flush()
        self.log.flush()

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if __name__ == '__main__':
    started = time.time()
    manifest = {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version,
        'configuration': {k: os.environ[k] for k in ['DATA_PATH', 'OUT_ROOT', 'QUICK', 'TUNE', 'MPLBACKEND']},
        'source_sha256': digest(ROOT / 'burnout_pipeline.py'),
        'data_sha256': digest(ROOT / 'tech_mental_health_burnout.csv'),
        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'status': 'running',
    }
    def save_manifest():
        (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save_manifest()
    with (OUT / 'training.log').open('w', encoding='utf-8') as log:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(sys.stdout, log)
        sys.stderr = Tee(sys.stderr, log)
        try:
            ns = runpy.run_path(str(ROOT / 'burnout_pipeline.py'), run_name='__main__')
            import joblib
            import pandas as pd
            from sklearn.metrics import precision_score, recall_score, f1_score
            model_dir = OUT / 'models'
            model_dir.mkdir(exist_ok=True)
            for name, pipeline in ns['fitted'].items():
                filename = name.lower().replace(' ', '_') + '.joblib'
                joblib.dump(pipeline, model_dir / filename, compress=3)
            pd.DataFrame([
                {'Model': name, 'Fold': i + 1, **{metric: float(values[i]) for metric, values in scores.items()}}
                for name, scores in ns['CV'].items() for i in range(ns['N_SPLITS'])
            ]).to_csv(OUT / 'results' / 'cv_fold_results.csv', index=False)
            pd.DataFrame({'row_index': ns['yte'].index, 'actual': ns['yte'].values,
                          **ns['proba']}).to_csv(OUT / 'results' / 'test_predictions.csv', index=False)
            pd.DataFrame({'row_index': ns['ytr'].index, 'actual': ns['ytr'].values,
                          'oof_probability': ns['oof']}).to_csv(OUT / 'results' / 'oof_best_model.csv', index=False)
            thresholds = []
            for label, t in [('Default 0.5', 0.5), ('F1-optimal', ns['t_f1']), ('Recall >= 80%', ns['t_rec'])]:
                pred = (ns['proba'][ns['best_name']] >= t).astype(int)
                thresholds.append({'Strategy': label, 'Threshold': float(t),
                                   'Precision': precision_score(ns['yte_np'], pred),
                                   'Recall': recall_score(ns['yte_np'], pred),
                                   'F1': f1_score(ns['yte_np'], pred)})
            pd.DataFrame(thresholds).to_csv(OUT / 'results' / 'threshold_results.csv', index=False)
            if ns['shap_by_feature'] is not None:
                ns['shap_by_feature'].rename('mean_abs_shap').to_csv(OUT / 'results' / 'shap_feature_importance.csv')
            manifest.update(status='complete', best_model_by_cv=ns['best_name'],
                            records=len(ns['df']), train_records=len(ns['Xtr']), test_records=len(ns['Xte']),
                            n_bootstrap=ns['N_BOOT'], source_unchanged=digest(ROOT / 'burnout_pipeline.py') == manifest['source_sha256'])
            print('\nSaved all five fitted pipelines, predictions, fold scores and threshold results.')
        except BaseException:
            manifest['status'] = 'failed'
            traceback.print_exc()
            raise
        finally:
            manifest['elapsed_seconds'] = time.time() - started
            manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save_manifest()
            sys.stdout, sys.stderr = original_stdout, original_stderr
