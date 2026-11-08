"""Ground-truth and full-release QA, separate from estimator fitting."""
import json
import platform
from pathlib import Path
import numpy as np
from .acquisition import sha256
from .loader import archive_path, load_ihdp, load_ihdp_truth, validate_archive
from .preprocess import IHDPPreprocessor, _observed


def validate_truth(observed, truth):
    """Validate aligned oracle arrays and summarize the selected population.

    Shape checks cannot prove patient identity. The release audit additionally
    compares each loaded array with the same split/replication in the archive.
    """
    data = _observed(observed)
    if set(truth) != {'mu0', 'mu1', 'tau', 'ycf'}:
        raise ValueError('Expected exactly mu0, mu1, tau, ycf')
    n = len(data['T'])
    for key, a in truth.items():
        a = np.asarray(a)
        if a.shape != (n,) or a.dtype.kind not in 'biuf' or not np.isfinite(a).all():
            raise ValueError(f'Invalid oracle shape or values: {key}')
    tau = truth['mu1'] - truth['mu0']
    if not np.allclose(truth['tau'], tau, rtol=1e-12, atol=1e-12):
        raise ValueError('tau differs from mu1 - mu0')
    T, Y = data['T'], data['Y']
    residual = Y - np.where(T == 1, truth['mu1'], truth['mu0'])
    cf_residual = truth['ycf'] - np.where(T == 1, truth['mu0'], truth['mu1'])
    return dict(n=n, treated=int((T==1).sum()), control=int((T==0).sum()),
                ate_true=float(tau.mean()),
                att_true=float(tau[T==1].mean()) if (T==1).any() else None,
                atc_true=float(tau[T==0].mean()) if (T==0).any() else None,
                tau_min=float(tau.min()), tau_max=float(tau.max()),
                factual_residual_mean=float(residual.mean()),
                factual_residual_sd=float(residual.std(ddof=0)),
                counterfactual_residual_sd=float(cf_residual.std(ddof=0)))


def audit_release(data_dir, manifest_path):
    """Audit replication 0, then 1/2/9, then the remaining release. No fitting of models."""
    manifest = json.loads(Path(manifest_path).read_text())
    before = {}; arrays = {}
    for split in ('train', 'test'):
        path = archive_path(split, data_dir)
        before[split] = sha256(path)
        if before[split] != manifest['files'][split]['sha256']:
            raise ValueError(f'{split} differs from recorded release hash')
        schema = validate_archive(path)
        if schema != manifest['files'][split]['schema']:
            raise ValueError(f'{split} schema differs from recorded release')
        with np.load(path, allow_pickle=False) as a:
            arrays[split] = {k:a[k] for k in ('x','t','yf','ycf','mu0','mu1')}
    records=[]
    order=[0,1,2,9]+[r for r in range(100) if r not in (0,1,2,9)]
    for r in order:
        try:
            observed={s:load_ihdp(r,s,data_dir) for s in arrays}
            processor=IHDPPreprocessor(standardize=True).fit(observed['train'])
            mean=observed['train']['X'][:,:6].mean(0)
            scale=observed['train']['X'][:,:6].std(0)
            scale=np.where(scale==0,1,scale)
            np.testing.assert_allclose(processor.scaler_.mean_,mean)
            np.testing.assert_allclose(processor.scaler_.scale_,scale)
            fingerprints={}
            for split,d in observed.items():
                source=arrays[split]
                for public,key in (('X','x'),('T','t'),('Y','yf')):
                    expected=source[key][:,:,r] if key=='x' else source[key][:,r]
                    np.testing.assert_array_equal(d[public],expected)
                truth=load_ihdp_truth(r,split,data_dir)
                for k in ('mu0','mu1','ycf'):np.testing.assert_array_equal(truth[k],source[k][:,r])
                summary=validate_truth(d,truth)
                raw=IHDPPreprocessor().fit_transform(d)
                scaled=processor.transform(d)
                for k in d:
                    np.testing.assert_array_equal(raw[k],d[k])
                    if k!='X':np.testing.assert_array_equal(scaled[k],d[k])
                np.testing.assert_array_equal(scaled['X'][:,6:],d['X'][:,6:])
                np.testing.assert_allclose(scaled['X'][:,:6],(d['X'][:,:6]-mean)/scale)
                np.testing.assert_array_equal(processor.scaler_.mean_,mean)
                fingerprints[split]={tuple(row) for row in np.column_stack((d['X'],d['T']))}
                records.append(dict(replication=r,split=split,status='passed',**summary))
            shared=len(fingerprints['train']&fingerprints['test'])
            if shared:raise ValueError(f'{shared} exact shared X/T fingerprints; review required')
        except (ValueError, AssertionError, KeyError) as error:
            records.append(dict(replication=r,split='combined',status='failed',error=str(error)))
    after={s:sha256(archive_path(s,data_dir)) for s in arrays}
    if before!=after:raise ValueError('Raw archive bytes changed during validation')
    return dict(python=platform.python_version(),numpy=np.__version__,
                hashes=before,raw_bytes_unchanged=True,execution_order=order,
                records=records,failures=sum(x['status']=='failed' for x in records),
                scope='Data QA only; no model fitting or oracle-based tuning')


def render_quality_report(result):
    records=result['records']; good=[r for r in records if r['status']=='passed']
    lines=['# Phase 5 data quality report','',
           f"Executed on Python {result['python']}, NumPy {result['numpy']}. {len(good)} split-level records passed; {result['failures']} replication-level failures.",
           '', '## Checks and interpretation','',
           'Both archives matched the locally recorded SHA-256 baseline before and after QA; ZIP CRC, complete schema and finite-value checks passed. These hashes are not publisher-authenticated signatures.',
           'Replication 0 ran first, followed by 1, 2 and 9, then all remaining replications. Loaded X/T/Y and oracle arrays were compared directly to their corresponding source slices. Feature order, shapes, binary encodings, raw copies, unchanged T/Y, and training-only optional scaling were checked for every split.',
           'No shared exact X/T row fingerprints were accepted between splits. Absence of duplicate fingerprints is not proof of patient identity separation; the release lacks verified patient identifiers.',
           'ATE is mean(mu1−mu0) for the stated split. ATT averages this contrast among treated rows; ATC among controls. Heterogeneity is represented by the range of tau. Do not substitute scalar archive ate=4 for split-specific ATE.',
           'Residuals compare noisy factual/counterfactual outcomes with their corresponding expected means. Nonzero residual SD is consistent with simulated outcome noise; normality, independence and the exact generator are not established by these summaries. No equality between yf and its assigned mean is imposed.',
           'Oracle arrays are accessed solely by this QA/evaluation path. No estimator, preprocessing setting or tuning parameter is selected using them. Replications remain separate; they are not independent patient cohorts.',
           '', '## Per-replication results','',
           '| Replication | Split | n | Treated | Control | ATE true | ATT true | ATC true | Tau min | Tau max | Factual residual SD | Counterfactual residual SD |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted(good,key=lambda x:(x['replication'],x['split'])):
        keys=['replication','split','n','treated','control','ate_true','att_true','atc_true','tau_min','tau_max','factual_residual_sd','counterfactual_residual_sd']
        lines.append('| '+' | '.join(f'{r[k]:.6f}' if isinstance(r[k],float) else str(r[k]) for k in keys)+' |')
    lines+=['','## Exceptions and limitations','']
    failures=[r for r in records if r['status']=='failed']
    lines += [str(r) for r in failures] if failures else ['No validation exceptions.']
    lines+=['','Clinical feature mapping, patient identity, exporter revision and redistribution terms remain unresolved. Numeric correctness does not establish positivity, exchangeability, real clinical effects or valid model uncertainty. Pre-Phase-5 audit model findings remain with Lenin; this work does not resolve them.','','Reproduce: `python scripts/validate_ihdp.py`. Aggregate JSON is saved beside this report. No row-level arrays or raw archives are committed.']
    return '\n'.join(lines)+'\n'
