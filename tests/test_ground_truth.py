"""Independent analytic oracle cases, including deliberately corrupted inputs."""
import numpy as np
import pytest
from causallens.data.validate import validate_truth


def example():
    X=np.zeros((4,25));X[:,13]=1
    d=dict(X=X,T=np.array([0,0,1,1]),Y=np.array([2.,3.,9.,12.]))
    q=dict(mu0=np.array([1.,2.,3.,4.]),mu1=np.array([2.,4.,6.,8.]),
           tau=np.array([1.,2.,3.,4.]),ycf=np.array([3.,5.,4.,5.]))
    return d,q


def test_known_population_targets_and_noise():
    d,q=example();r=validate_truth(d,q)
    assert r['ate_true']==2.5 and r['att_true']==3.5 and r['atc_true']==1.5
    assert r['tau_min']==1 and r['tau_max']==4
    assert r['factual_residual_sd']==pytest.approx(np.std([1,1,3,4]))
    assert r['counterfactual_residual_sd']==0


@pytest.mark.parametrize('error',['permuted_tau','shape','nan','extra','missing'])
def test_invalid_truth_rejected(error):
    d,q=example()
    if error=='permuted_tau':q['tau']=q['tau'][::-1]
    elif error=='shape':q['mu1']=q['mu1'][:,None]
    elif error=='nan':q['mu0'][0]=np.nan
    elif error=='extra':q['other']=np.zeros(4)
    else:del q['mu0']
    with pytest.raises(ValueError):validate_truth(d,q)


def test_empty_treatment_subgroup_is_not_reported_as_zero_effect():
    d,q=example();d['T'][:]=0
    assert validate_truth(d,q)['att_true'] is None


def test_semantics_check_does_not_claim_to_detect_joint_row_permutation():
    d,q=example();q={k:v[::-1] for k,v in q.items()}
    # All oracle arrays consistently permuted still satisfy shape/contrast checks.
    # The release audit must additionally compare each source slice.
    assert validate_truth(d,q)['ate_true']==2.5
