"""Verify the shipped replication snapshot before running mutating scripts."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'public_package_manifest.json').read_text(encoding='utf-8'))
failures = []
for relative, expected in manifest['files'].items():
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        failures.append(relative + ': missing or unsafe path')
    elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(relative + ': SHA-256 mismatch')
if failures:
    raise SystemExit('Snapshot verification failed: ' + '; '.join(failures))
print(json.dumps({'status': 'passed', 'files_checked': len(manifest['files'])}))
