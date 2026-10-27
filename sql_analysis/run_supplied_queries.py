"""Run the canonical corrected SQL file and independently cross-check with pandas."""
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import hashlib
import json
import re
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sql_text = (OUT / 'sql_queries.sql').read_text(encoding='utf-8')
queries = [q.strip() for q in re.sub(r'--[^\n]*', '', sql_text).split(';') if q.strip()]
assert len(queries) == 3
df = pd.read_csv(ROOT / 'tech_mental_health_burnout.csv')
df['at_risk'] = (df['burnout_level'] != 'Low').astype(int)
con = duckdb.connect(str(OUT / 'burnout_analysis.duckdb'))
con.register('source_data', df)
con.execute('CREATE OR REPLACE TABLE burnout_v AS SELECT * FROM source_data')
frames = [con.execute(q).df() for q in queries]
for i, frame in enumerate(frames, 1):
    frame.to_csv(OUT / f'query_{i}_results.csv', index=False)

q1, q2, q3 = frames
assert q1.n.sum() == len(df) == 150000
means = {'stress': ('stress_level',2), 'anxiety': ('anxiety_score',2),
         'depression': ('depression_score',2), 'work_hours': ('work_hours_per_week',1),
         'overtime': ('overtime_hours',2), 'wlb': ('work_life_balance',2),
         'mgr_support': ('manager_support',2), 'job_sat': ('job_satisfaction',2),
         'sleep': ('sleep_hours',2), 'screen': ('screen_time_hours',2),
         'social': ('social_support_score',2)}
for _, row in q1.iterrows():
    subset = df.loc[df.at_risk == int(row.grp == 'At Risk')]
    assert row.n == len(subset)
    for alias, (column, digits) in means.items():
        assert np.isclose(row[alias], round(subset[column].mean(), digits), atol=1e-10)
for _, row in q2.iterrows():
    assert np.isclose(row.corr_with_at_risk, round(df[row.feature].corr(df.at_risk), 3), atol=1e-10)
assert q2.loc[q2.feature == 'social_support_score', 'group_name'].item() == 'Work environment'
stress = pd.cut(df.stress_level, [-np.inf,4,6,8,np.inf], right=False, labels=['1. Stress < 4','2. Stress 4-6','3. Stress 6-8','4. Stress >= 8'])
hours = pd.cut(df.work_hours_per_week, [-np.inf,40,50,60,np.inf], labels=['a. <=40h','b. 41-50h','c. 51-60h','d. >60h'])
groups = df.groupby([stress,hours], observed=True).at_risk.agg(['size','mean'])
assert len(q3) == int((groups['size'] >= 30).sum())
for _, row in q3.iterrows():
    actual = groups.loc[(row.stress_band, row.hours_band)]
    assert row.n == actual['size']
    assert np.isclose(row.pct_at_risk, round(actual['mean']*100,1), atol=1e-10)
manifest = {'engine': 'DuckDB', 'version': duckdb.__version__, 'records': len(df),
            'data_sha256': hashlib.sha256((ROOT/'tech_mental_health_burnout.csv').read_bytes()).hexdigest(),
            'queries': 3, 'pandas_crosscheck': 'passed',
            'source_query_2_group_note': 'Corrected canonical SQL: social_support_score consistently belongs to Work environment.',
            'query_3_minimum_group_size': 30,
            'query_3_records_in_retained_cells': int(q3.n.sum())}
(OUT / 'sql_run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
con.close()
for i, f in enumerate(frames, 1):
    print(f'QUERY {i}\n{f.to_string(index=False)}\n')
print('Pandas cross-check passed for all three queries.')
