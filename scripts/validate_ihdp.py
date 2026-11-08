"""Reproduce all-100-replication data QA with aggregate-only output."""
import argparse
import json
from pathlib import Path
from causallens.data.validate import audit_release, render_quality_report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',default='data/raw')
    p.add_argument('--manifest',default='docs/data_manifest.json')
    p.add_argument('--output-dir',type=Path,default=Path('reports'))
    a=p.parse_args();r=audit_release(a.data_dir,a.manifest)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'data_quality_results.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    (a.output_dir/'data_quality_report.md').write_text(render_quality_report(r))
    print(f"{len(r['records'])} records; {r['failures']} failures; raw bytes unchanged")
    raise SystemExit(bool(r['failures']))
