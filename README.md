# Explainable burnout risk classification in technology workers

This repository contains code, synthetic data, fitted models and saved predictions for a five-model burnout classification benchmark, signed-grouped SHAP analysis and feature ablations. The accompanying paper is `output/pdf/Group15_Revised_Paper.pdf`; the editable LNCS sources are in `paper_revised/`.

The benchmark uses 150,000 synthetic records, a stratified 120,000/30,000 split and 23 predictors. Logistic Regression has test average precision (AP) 0.810967. Treatment-excluded F4 has AP 0.789913 and is the recommended benchmark configuration when treatment timing is unknown. These results have no external, temporal or subgroup validation and do not establish suitability for employee screening. F4 and F5 are post hoc sensitivity analyses on the existing test set. Bootstrap intervals are conditional on fitted models.

## Authors

| Author | ORCID |
|---|---|
| Le Hoang Ky Phuong | https://orcid.org/0009-0007-7325-0720 |
| Nguyen Anh Khoa | https://orcid.org/0009-0008-6007-9693 |
| Luu Gia Luong | https://orcid.org/0009-0002-2274-6121 |
| Thu Le | https://orcid.org/0009-0008-3480-8311 |
| Nhat-Tung Le | https://orcid.org/0000-0002-7210-3426 |

Le Hoang Ky Phuong is the first author. Nhat-Tung Le is the last author and corresponding author: tungln4@fpt.edu.vn. All authors are affiliated with FPT University, Ho Chi Minh City, Vietnam.

## Data and licenses

S. Gupta's [Employee Mental Health & Burnout Dataset](https://www.kaggle.com/datasets/suhanigupta04/employee-mental-health-and-burnout-dataset), version 1, was released on 15 March 2026. Kaggle metadata identifies its license as CC0: Public Domain. The included `tech_mental_health_burnout.csv` has SHA-256 `0bf19913d26b49546a96de481de04474a551b7b24cb579549f7c61f6365a92e0`; the metadata and hash comparison are retained under `revision_20261022/reference_sources/`. The records are synthetic. The data card identifies `seeks_professional_help` as a second intended target; its temporal availability for burnout prediction is unknown.

The CC0 statement applies to the dataset. It does not assign a license to the analysis code or manuscript. See `LICENSES.md` for scope; the bundled Springer `llncs.cls` retains its own notices.

## Check the snapshot

Use Python 3.12 in a separate environment. The following commands verify file integrity and the saved scientific results without fitting new models:

```sh
python -m venv .venv-reproduce
# Activate .venv-reproduce using the command for your shell.
python -m pip install -r requirements_revision.txt
python -m pip check
python verify_public_package.py
python run_training.py --check
python verify_training_results.py
python verify_revision.py
```

`verify_public_package.py` uses only the Python standard library and checks the complete shipped file inventory. Run it before other commands, because the scientific verifiers write their verification JSON files. The independent scientific checks recompute metrics, split membership, paired intervals, feature ablations and SHAP grouping from saved data and predictions. Pickled models are bundled research artifacts; the XGBoost native JSON and separate preprocessor support cross-platform verification.

`requirements_revision.txt` describes the local analysis environment. The retained benchmark was fitted in Colab using different versions recorded in `training_results/actual_package_versions.json` and `training_results/reproducibility/requirements_training.txt`. Do not install both sets of pins into one environment. A full refit may differ numerically with library versions or hardware; saved row indices and predictions preserve the reported experiment.

## Recompute analyses and figures

Run these commands in a copy of the repository if you want to preserve the shipped snapshot:

```sh
python revision_analysis.py
python sync_revision.py
python sql_analysis/run_supplied_queries.py
python build_revision_artifacts.py
```

The feature analysis fits 30 Logistic Regression models across F1-F5 and recomputes SHAP, permutation importance and 2,000 paired bootstrap resamples. The five benchmark pipelines and their CV, out-of-fold and test predictions are retained. For a new five-model training run use `python run_training.py --output training_run_new`. The archived training source is preserved separately so its historical source hash remains verifiable.

The document builder also needs `requirements_documents.txt`. Compile `paper_revised/main.tex` with Tectonic or upload the contents of `paper_revised/` to Overleaf. Charts are included, so compiling the supplied LaTeX source does not require retraining. `build_revision_artifacts.py` is the authoritative generator: update its bibliography and `paper_revised/manuscript_template.tex.in` before regenerating the manuscript.

## Files

| Path | Contents |
|---|---|
| `training_results/` | Fitted benchmark models, saved CV/OOF/test predictions, actual package versions and archived training source |
| `revision_20261019/` | F1-F5 estimators, paired bootstrap samples, grouped SHAP values and analysis provenance |
| `sql_analysis/` | Three supplied queries, their results and independent pandas checks |
| `dashboard/` | Interactive dashboard and its data |
| `dashboard_submission/` | Standalone HTML and printable dashboard |
| `paper_revised/` | Manuscript template, generated LaTeX, LNCS class and figures |
| `public_package_manifest.json` | SHA-256 inventory of every shipped file |

The dated analysis directories are retained because the executable scripts and provenance point to them. The snapshot excludes reviewer correspondence, private feedback, source backups, local software, unrelated coursework, historical paper exports and local execution logs. The project repository is https://github.com/Group15-FA26-AI2109/burnout-risk-classification . No DOI has been assigned.
