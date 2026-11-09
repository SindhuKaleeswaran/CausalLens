"""Phase 2 tests: synthetic fixtures by default; no network or model fitting."""
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pytest
from causallens.data import acquisition
from causallens.data.loader import (FEATURE_NAMES, archive_path, load_ihdp,
                                    load_ihdp_truth, validate_archive)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_ihdp")
    for split, n in (("train",672),("test",75)):
        # Clearly artificial values encode row, feature and replication axes.
        x = (np.arange(n)[:,None,None] + np.arange(25)[None,:,None]/100
             + np.arange(100)[None,None,:]/10000).astype(np.float64)
        t = np.broadcast_to((np.arange(n)%2)[:,None],(n,100)).astype(float)
        mu0 = np.broadcast_to(np.arange(100)[None,:],(n,100)).astype(float)
        mu1 = mu0 + 3
        np.savez(archive_path(split,root), x=x,t=t,yf=mu0+7,ycf=mu1+9,
                 mu0=mu0,mu1=mu1,ate=np.array(0),yadd=np.array(0),ymul=np.array(1))
    return root


@pytest.mark.parametrize("rep",[0,2,99,np.int64(4)])
@pytest.mark.parametrize("split",["train","test"])
def test_loader_preserves_axes_values_and_dtypes(synthetic,rep,split):
    observed = load_ihdp(rep,split,synthetic)
    assert set(observed) == {"X","T","Y"}
    with np.load(archive_path(split,synthetic),allow_pickle=False) as a:
        for public,key in (("X","x"),("T","t"),("Y","yf")):
            expected = a[key][:,:,rep] if key=="x" else a[key][:,rep]
            np.testing.assert_array_equal(observed[public],expected)
            assert observed[public].dtype == expected.dtype
    assert FEATURE_NAMES == tuple(f"x{i:02d}" for i in range(1,26))
    observed["X"][0,0] = -999
    assert load_ihdp(rep,split,synthetic)["X"][0,0] != -999


@pytest.mark.parametrize("rep",[-1,100,0.5,"0",True,None])
def test_invalid_replication(synthetic,rep):
    with pytest.raises(ValueError,match="replication"):
        load_ihdp(rep,data_dir=synthetic)


@pytest.mark.parametrize("split",["validation","TRAIN","",None])
def test_invalid_split(synthetic,split):
    with pytest.raises(ValueError,match="split"):
        load_ihdp(split=split,data_dir=synthetic)


def test_truth_is_separate_and_noise_is_not_replaced(synthetic):
    observed=load_ihdp(data_dir=synthetic)
    truth=load_ihdp_truth(split="train",data_dir=synthetic)
    assert set(truth)=={"mu0","mu1","ycf","tau"}
    np.testing.assert_array_equal(truth["tau"],truth["mu1"]-truth["mu0"])
    assert not np.array_equal(observed["Y"],np.where(observed["T"],truth["mu1"],truth["mu0"]))
    np.testing.assert_array_equal(truth["tau"],np.full(672,3))


@pytest.mark.parametrize("problem",["shape","missing","nonfinite","treatment","object","xshape"])
def test_reject_invalid_archive(tmp_path,problem):
    a={"x":np.zeros((2,25,100)), **{k:np.zeros((2,100)) for k in ("t","yf","ycf","mu0","mu1")}}
    if problem=="shape": a["yf"]=np.zeros((1,100))
    elif problem=="missing": del a["mu1"]
    elif problem=="nonfinite": a["x"][0,0,0]=np.nan
    elif problem=="treatment": a["t"][0,0]=2
    elif problem=="object": a["yf"]=np.array([object()])
    elif problem=="xshape": a["x"]=np.zeros((2,100,25))
    file=tmp_path/"bad.npz";np.savez(file,**a)
    with pytest.raises(ValueError): validate_archive(file)


def test_reject_corrupt_zip(tmp_path):
    path=tmp_path/"bad.npz";path.write_bytes(b"incomplete archive")
    with pytest.raises(zipfile.BadZipFile): validate_archive(path)


class Response:
    headers={}
    def __init__(self,url,body): self.url,self.body=url,body
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def raise_for_status(self): pass
    def iter_content(self,size):
        for i in range(0,len(self.body),size): yield self.body[i:i+size]


def test_download_and_cache_without_network(tmp_path,synthetic,monkeypatch):
    calls=[]
    def get(url,**kwargs):
        split="train" if ".train." in url else "test"
        calls.append(url)
        body=archive_path(split,synthetic).read_bytes()
        response=Response(url,body);response.headers={"Content-Length":str(len(body))}
        return response
    monkeypatch.setattr(acquisition.requests,"get",get)
    expected={s:acquisition.sha256(archive_path(s,synthetic)) for s in ("train","test")}
    first=acquisition.download_ihdp(tmp_path,expected)
    second=acquisition.download_ihdp(tmp_path,expected)
    assert first==second and len(calls)==2
    assert not list(tmp_path.glob("*.part"))
    for split in ("train","test"):
        assert first["files"][split]["sha256"]==expected[split]
        assert archive_path(split,tmp_path).read_bytes()==archive_path(split,synthetic).read_bytes()


def test_download_mismatch_cleans_partial(tmp_path,synthetic,monkeypatch):
    monkeypatch.setattr(acquisition.requests,"get",lambda url,**kw:Response(url,archive_path("train",synthetic).read_bytes()))
    with pytest.raises(ValueError,match="SHA-256"):
        acquisition.download_ihdp(tmp_path,{"train":"0"*64})
    assert not archive_path("train",tmp_path).exists()
    assert not list(tmp_path.glob("*.part"))


def test_existing_hash_mismatch_refuses_overwrite(tmp_path,synthetic):
    path=archive_path("train",tmp_path);shutil.copyfile(archive_path("train",synthetic),path)
    before=path.read_bytes()
    with pytest.raises(ValueError,match="Hash mismatch"):
        acquisition.download_ihdp(tmp_path,{"train":"0"*64})
    assert path.read_bytes()==before


def test_content_length_mismatch(tmp_path,monkeypatch):
    def get(url,**kw):
        r=Response(url,b"incomplete");r.headers={"Content-Length":"999"};return r
    monkeypatch.setattr(acquisition.requests,"get",get)
    with pytest.raises(ValueError,match="Incomplete"):
        acquisition.download_ihdp(tmp_path)
    assert not list(tmp_path.iterdir())


def test_local_benchmark_integration():
    root=Path(__file__).resolve().parents[1]
    data=root/"data/raw"
    if not all(archive_path(s,data).exists() for s in ("train","test")):
        pytest.skip("Local benchmark not downloaded; offline synthetic tests still run")
    from causallens.data.inspection import inspect_ihdp
    result=inspect_ihdp(data,0)
    manifest=json.loads((root/"docs/data_manifest.json").read_text())
    for split,n in (("train",672),("test",75)):
        s=result["splits"][split]
        assert s["n"]==n
        assert s["sha256"]==manifest["files"][split]["sha256"]
    assert result["split_check"]["exact_shared_covariate_treatment_rows"]==0


def test_unpacked_size_limit_precedes_decompression(tmp_path, monkeypatch):
    from causallens.data import loader
    path = tmp_path / "compressed.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("x.npy", bytes(256000))
    monkeypatch.setattr(loader, "MAX_UNPACKED_BYTES", 100000)
    with pytest.raises(ValueError, match="Unpacked archive"):
        loader.validate_archive(path)
