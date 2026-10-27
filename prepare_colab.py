"""Generate the revised Colab notebook and matching inputs from current sources."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
INPUT_NAMES = ('burnout_pipeline.py', 'tech_mental_health_burnout.csv', 'run_training.py', 'requirements_revision.txt')


def cell(kind, text):
    result = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook():
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in INPUT_NAMES}
    return {
        'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'colab': {'name': 'Burnout_5_models_CPU.ipynb', 'provenance': []},
                     'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                     'language_info': {'name': 'python'}, 'group15_inputs_sha256': hashes},
        'cells': [
            cell('markdown', '# Burnout: revised five-model full run on Colab CPU\n\n'
                 'Runtime → Change runtime type → Python 3, accelerator None, standard RAM.\n\n'
                 'Upload the supplied `colab_inputs.zip`. The pinned environment is installed after input verification. '
                 '`QUICK=0`, `TUNE=0`, seed 42, five CV folds and 2,000 bootstrap resamples. '
                 'Signed SHAP is grouped within each observation before taking absolute values; F1–F5 feature sets are evaluated.\n\n'
                 'This notebook performs a new full fit in a fresh output directory. The submitted benchmark is the retained '
                 'Colab fit; reproduce the focused revision locally with `revision_analysis.py`. Cross-platform reruns can differ '
                 'numerically and must be compared before promoting them into the paper.\n'),
            cell('code', 'import os, shutil, subprocess\n'
                 'assert not os.environ.get("COLAB_GPU", ""), "Select accelerator None."\n'
                 'if shutil.which("nvidia-smi"):\n'
                 '    check = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)\n'
                 '    assert check.returncode != 0 or not check.stdout.strip(), "Select accelerator None."\n'
                 'print("CPU runtime; logical CPUs:", os.cpu_count())\n'),
            cell('code', 'from google.colab import files\n'
                 'from pathlib import Path\n'
                 'import zipfile, hashlib\n'
                 'uploaded = files.upload()\n'
                 'assert "colab_inputs.zip" in uploaded, "Upload the supplied colab_inputs.zip."\n'
                 f'expected = {hashes!r}\n'
                 'with zipfile.ZipFile("colab_inputs.zip") as archive:\n'
                 '    assert len(archive.namelist()) == len(expected) and set(archive.namelist()) == set(expected), "Unexpected archive members."\n'
                 '    for name, digest in expected.items():\n'
                 '        payload = archive.read(name)\n'
                 '        assert hashlib.sha256(payload).hexdigest() == digest, name\n'
                 '        Path("/content", name).write_bytes(payload)\n'
                 'print("Verified revised pipeline, runner, dataset and dependency pins.")\n'),
            cell('code', '%pip install -q -r /content/requirements_revision.txt\n'
                 'import subprocess, sys\n'
                 'subprocess.run([sys.executable, "-m", "pip", "check"], check=True)\n'
                 'subprocess.run([sys.executable, "/content/run_training.py", "--check"], cwd="/content", check=True)\n'),
            cell('code', 'import subprocess, sys, shutil\n'
                 'from datetime import datetime, timezone\n'
                 'from google.colab import files\n'
                 'out = "/content/training_run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")\n'
                 'run = subprocess.run([sys.executable, "-u", "/content/run_training.py", "--output", out], cwd="/content")\n'
                 'if Path(out).exists():\n'
                 '    archive = shutil.make_archive(out, "zip", out)\n'
                 '    files.download(archive)\n'
                 'assert run.returncode == 0, "Pipeline failed; inspect the log and partial output. Do not use as completed results."\n'),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check notebook and input ZIP without changing them.')
    args = parser.parse_args()
    notebook = build_notebook()
    notebook_path = ROOT / 'Burnout_5_models_CPU.ipynb'
    archive_path = ROOT / 'colab_inputs.zip'
    if args.check:
        assert json.loads(notebook_path.read_text(encoding='utf-8')) == notebook, 'Notebook is stale. Run prepare_colab.py.'
        with zipfile.ZipFile(archive_path) as archive:
            assert len(archive.namelist()) == len(INPUT_NAMES) and set(archive.namelist()) == set(INPUT_NAMES)
            assert archive.testzip() is None
            for name in INPUT_NAMES:
                assert archive.read(name) == (ROOT / name).read_bytes(), f'Stale Colab input: {name}'
        print('Colab notebook and all four inputs match current sources.')
        return
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in INPUT_NAMES:
            archive.write(ROOT / name, name)
    print('Prepared revised Colab notebook and four verified inputs.')


if __name__ == '__main__':
    main()
