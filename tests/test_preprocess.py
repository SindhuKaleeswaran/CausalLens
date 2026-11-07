import numpy as np
import pytest
from causallens.data.preprocess import IHDPPreprocessor, load_preprocessed_ihdp


def observed(offset=0):
    X=np.zeros((4,25));X[:,:6]=np.arange(4)[:,None]+offset;X[:,13]=[1,2,1,2]
    return dict(X=X,T=np.array([0.,1.,0.,1.]),Y=np.arange(4.))


def test_raw_default_preserves_values_dtypes_and_input_ownership():
    raw=observed();out=IHDPPreprocessor().fit_transform(raw)
    for k in raw:
        np.testing.assert_array_equal(raw[k],out[k]);assert raw[k].dtype==out[k].dtype
        assert not np.shares_memory(raw[k],out[k])


def test_train_only_scaling_and_unchanged_binary_t_y():
    train,test=observed(),observed(100)
    original={k:v.copy() for k,v in train.items()}
    p=IHDPPreprocessor(standardize=True);tr=p.fit_transform(train);te=p.transform(test)
    np.testing.assert_allclose(tr['X'][:,:6].mean(0),0,atol=1e-15)
    np.testing.assert_allclose(tr['X'][:,:6].std(0),1)
    np.testing.assert_allclose(te['X'][:,:6],(test['X'][:,:6]-1.5)/np.sqrt(1.25))
    np.testing.assert_allclose(p.scaler_.mean_,1.5)
    for raw,out in ((train,tr),(test,te)):
        np.testing.assert_array_equal(out['X'][:,6:],raw['X'][:,6:])
        for k in ('T','Y'):np.testing.assert_array_equal(out[k],raw[k])
    for k in train:np.testing.assert_array_equal(train[k],original[k])


def test_y_t_cannot_influence_fitted_statistics():
    a,b=observed(),observed();b['Y']*=1e6;b['T']=1-b['T']
    pa=IHDPPreprocessor(standardize=True).fit(a);pb=IHDPPreprocessor(standardize=True).fit(b)
    np.testing.assert_array_equal(pa.scaler_.mean_,pb.scaler_.mean_)
    np.testing.assert_array_equal(pa.scaler_.scale_,pb.scaler_.scale_)


@pytest.mark.parametrize('field',['mu0','mu1','tau','ycf'])
def test_oracle_fields_rejected(field):
    d=observed();d[field]=np.zeros(4)
    with pytest.raises(ValueError,match='oracle'):IHDPPreprocessor().fit(d)


@pytest.mark.parametrize('key',['X','T','Y'])
def test_nonfinite_rejected_without_imputation(key):
    d=observed();d[key].flat[0]=np.nan
    with pytest.raises(ValueError,match='finite'):IHDPPreprocessor().fit(d)


def test_constant_features_and_repeatability():
    d=observed();d['X'][:,0]=7
    a=IHDPPreprocessor(standardize=True).fit_transform(d)
    b=IHDPPreprocessor(standardize=True).fit_transform(d)
    np.testing.assert_array_equal(a['X'],b['X']);assert np.all(a['X'][:,0]==0)


def test_fit_required_and_configuration_changes_rejected():
    p=IHDPPreprocessor()
    with pytest.raises(ValueError,match='Fit'):p.transform(observed())
    p.fit(observed());p.standardize=True
    with pytest.raises(ValueError,match='Configuration'):p.transform(observed())


def test_wrapper_keeps_replication_and_split_state_isolated(monkeypatch):
    import causallens.data.preprocess as module
    calls=[]
    def load(r,s,data_dir):
        calls.append((r,s));return observed(r*10+(100 if s=='test' else 0))
    monkeypatch.setattr(module,'load_ihdp',load)
    a=load_preprocessed_ihdp(0,standardize=True)
    b=load_preprocessed_ihdp(3,standardize=True)
    assert calls==[(0,'train'),(0,'test'),(3,'train'),(3,'test')]
    for x,y in zip(a,b):np.testing.assert_allclose(x['X'],y['X'])


@pytest.mark.parametrize('value', [1, 0, 'yes', None])
def test_mutated_configuration_must_remain_boolean(value):
    p = IHDPPreprocessor().fit(observed())
    p.standardize = value
    with pytest.raises(ValueError):
        p.transform(observed())
    with pytest.raises(ValueError):
        p.fit(observed())
    p.standardize = False
    with pytest.raises(ValueError, match='Fit'):
        p.transform(observed())


@pytest.mark.parametrize('problem', ['features', 'alignment', 'encoding', 'treatment', 'extra', 'inf'])
def test_invalid_observed_inputs(problem):
    d = observed()
    if problem == 'features': d['X'] = d['X'][:, :24]
    if problem == 'alignment': d['Y'] = d['Y'][:-1]
    if problem == 'encoding': d['X'][0, 13] = 0
    if problem == 'treatment': d['T'][0] = 2
    if problem == 'extra': d['unknown'] = 1
    if problem == 'inf': d['X'][0, 0] = np.inf
    with pytest.raises(ValueError):
        IHDPPreprocessor().fit(d)
