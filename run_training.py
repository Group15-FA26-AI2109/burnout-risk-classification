"""Run the revised full benchmark into a new directory with explicit provenance.

This is a NEW fit; it does not replace the retained benchmark in training_results.
Use --check for a dependency/data preflight without fitting or writing outputs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
MODULES = {'numpy': 'numpy', 'pandas': 'pandas', 'scikit-learn': 'sklearn',
           'scipy': 'scipy', 'shap': 'shap', 'joblib': 'joblib',
           'matplotlib': 'matplotlib', 'duckdb': 'duckdb', 'xgboost': 'xgboost',
           'lightgbm': 'lightgbm', 'imbalanced-learn': 'imblearn',
           'threadpoolctl': 'threadpoolctl'}


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


def preflight():
    """Check imported modules, not only potentially duplicated pip metadata."""
    pinned = {}
    for line in (ROOT / 'requirements_revision.txt').read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            name, version = line.split('==')
            pinned[name] = version
    imported = {}
    for distribution, module_name in MODULES.items():
        module = importlib.import_module(module_name)
        imported[distribution] = getattr(module, '__version__', importlib.metadata.version(distribution))
        if imported[distribution] != pinned[distribution]:
            raise RuntimeError(f'{distribution}: imported {imported[distribution]}, expected {pinned[distribution]}. '
                               'Install requirements_revision.txt in a clean environment.')
    data = ROOT / 'tech_mental_health_burnout.csv'
    manifest_path = ROOT / 'training_results/run_manifest.json'
    expected_data = json.loads(manifest_path.read_text(encoding='utf-8'))['data_sha256'] if manifest_path.exists() else '0bf19913d26b49546a96de481de04474a551b7b24cb579549f7c61f6365a92e0'
    if digest(data) != expected_data:
        raise RuntimeError('Dataset SHA-256 differs from the retained benchmark.')
    return imported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'training_run',
                        help='New or empty directory; existing results are never overwritten.')
    parser.add_argument('--check', action='store_true', help='Check inputs/dependencies only; no fit or output writes.')
    args = parser.parse_args()
    out = args.output.resolve()
    if not args.check and out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f'Output directory is not empty: {out}. Choose a new --output directory.')
    os.environ.update(DATA_PATH=str(ROOT / 'tech_mental_health_burnout.csv'),
                      OUT_ROOT=str(out), QUICK='0', TUNE='0', N_BOOT='2000', MPLBACKEND='Agg')
    imported = preflight()
    if args.check:
        print(json.dumps({'status': 'preflight_passed', 'data_sha256': digest(ROOT / 'tech_mental_health_burnout.csv'),
                          'source_sha256': digest(ROOT / 'burnout_pipeline.py'), 'imported_modules': imported,
                          'configuration': {k: os.environ[k] for k in ['QUICK', 'TUNE', 'N_BOOT']},
                          'fit_performed': False}, indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest = {
        'started_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
        'purpose': 'new full benchmark; retained submission benchmark is unchanged',
        'configuration': {k: os.environ[k] for k in ['DATA_PATH', 'OUT_ROOT', 'QUICK', 'TUNE', 'N_BOOT', 'MPLBACKEND']},
        'source_sha256': digest(ROOT / 'burnout_pipeline.py'),
        'runner_sha256': digest(Path(__file__)), 'requirements_sha256': digest(ROOT / 'requirements_revision.txt'),
        'data_sha256': digest(ROOT / 'tech_mental_health_burnout.csv'),
        'imported_modules': imported,
        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'status': 'running',
    }

    def save_manifest():
        (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    save_manifest()
    with (out / 'training.log').open('w', encoding='utf-8') as log:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
        try:
            ns = runpy.run_path(str(ROOT / 'burnout_pipeline.py'), run_name='__main__')
            import joblib
            import pandas as pd
            from sklearn.metrics import precision_score, recall_score, f1_score
            model_dir = out / 'models'
            model_dir.mkdir(exist_ok=True)
            for name, pipeline in ns['fitted'].items():
                joblib.dump(pipeline, model_dir / (name.lower().replace(' ', '_') + '.joblib'), compress=3)
            xgb = ns['fitted']['XGBoost']
            xgb.named_steps['clf'].save_model(model_dir / 'xgboost_native.json')
            joblib.dump(xgb.named_steps['pre'], model_dir / 'xgboost_preprocessor.joblib', compress=3)
            pd.DataFrame([
                {'Model': name, 'Fold': i + 1, **{metric: float(values[i]) for metric, values in scores.items()}}
                for name, scores in ns['CV'].items() for i in range(ns['N_SPLITS'])
            ]).to_csv(out / 'results/cv_fold_results.csv', index=False)
            pd.DataFrame({'row_index': ns['yte'].index, 'actual': ns['yte'].values,
                          **ns['proba']}).to_csv(out / 'results/test_predictions.csv', index=False)
            pd.DataFrame({'row_index': ns['ytr'].index, 'actual': ns['ytr'].values,
                          'oof_probability': ns['oof']}).to_csv(out / 'results/oof_best_model.csv', index=False)
            thresholds = []
            for label, threshold in [('Default 0.5', 0.5), ('F1-optimal', ns['t_f1']), ('Recall >= 80%', ns['t_rec'])]:
                pred = (ns['proba'][ns['best_name']] >= threshold).astype(int)
                thresholds.append({'Strategy': label, 'Threshold': float(threshold),
                                   'Precision': precision_score(ns['yte_np'], pred),
                                   'Recall': recall_score(ns['yte_np'], pred), 'F1': f1_score(ns['yte_np'], pred)})
            pd.DataFrame(thresholds).to_csv(out / 'results/threshold_results.csv', index=False)
            if ns['shap_by_feature'] is not None:
                ns['shap_by_feature'].rename('mean_abs_shap').to_csv(out / 'results/shap_feature_importance.csv')
            manifest.update(status='complete', best_model_by_cv=ns['best_name'],
                            records=len(ns['df']), train_records=len(ns['Xtr']), test_records=len(ns['Xte']),
                            n_bootstrap=ns['N_BOOT'], source_unchanged=digest(ROOT / 'burnout_pipeline.py') == manifest['source_sha256'])
            print('\nSaved all five fitted pipelines, native XGBoost, predictions, fold scores and threshold results.')
        except BaseException:
            manifest['status'] = 'failed'
            traceback.print_exc()
            raise
        finally:
            manifest['elapsed_seconds'] = time.time() - started
            manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save_manifest()
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == '__main__':
    main()
