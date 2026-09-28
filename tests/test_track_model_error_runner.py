"""Synthetic sampling contracts: native index, strict support, no implicit label swap."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

HERE=Path(__file__).resolve()
SCRIPT=HERE.with_name('diagnose_track_model_errors.py')
if not SCRIPT.exists(): SCRIPT=HERE.parents[1]/'scripts/diagnose_track_model_errors.py'
spec=importlib.util.spec_from_file_location('track_error_runner',SCRIPT)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def sample_fixture(monkeypatch):
    a={'primary_label':np.array([2,2,0,1,0,-1]),'legacy_label':np.array([2,2,0,1,0,-1]),
       'positive_depth':np.ones(6,bool),'reprojection_error':np.array([1.,1.01,.5,.5,.5,.5]),
       'primary_boundary_distance':np.array([11.,11.,11.,10.,11.,np.nan]),
       'legacy_valid':np.array([1,1,1,1,0,1],bool),'image_id':np.array([2,2,2,2,2,2]),
       'legacy_index':np.array([[17,8],[17,8],[18,8],[19,8],[19,8],[19,8]])}
    cached=np.zeros((989,1320,5),np.float32)
    cached[8,17,2]=1;cached[8,18,0]=1
    monkeypatch.setattr(runner.np,'load',lambda *_args,**_kwargs:cached)
    views=[{'image_id':2,'name':'002.png','probability':{'path':'synthetic.npy','sha256':'synthetic'}}]
    return a,views,cached


def test_queries_use_saved_native_index_and_strict_support(monkeypatch):
    a,views,_=sample_fixture(monkeypatch)
    strict,rows,prob,evidence=runner.sample_queries(a,views)
    assert strict.tolist()==[True,False,True,False,True,False]
    assert rows.tolist()==[0,2] and prob.argmax(1).tolist()==[2,0]
    assert evidence['queries']==evidence['unique_image_pixel_queries']==2


def test_sampling_refuses_coordinate_label_mismatch_and_bad_probs(monkeypatch):
    a,views,cached=sample_fixture(monkeypatch)
    a['legacy_label'][0]=0
    with pytest.raises(ValueError,match='labels differ'):runner.sample_queries(a,views)
    a['legacy_label'][0]=2;cached[8,17,2]=.8
    with pytest.raises(ValueError,match='Malformed'):runner.sample_queries(a,views)


def test_duplicate_pixels_reported_and_queries_not_reweighted(monkeypatch):
    a,views,_=sample_fixture(monkeypatch)
    a['legacy_index'][2]=a['legacy_index'][0]
    _,rows,_,evidence=runner.sample_queries(a,views)
    assert len(rows)==2 and evidence['duplicate_image_pixel_queries']==1


def test_prepare_refuses_overwrite_and_json_requires_finite(tmp_path):
    with pytest.raises(ValueError,match='overwrite'):runner.prepare(tmp_path)
    with pytest.raises(ValueError):runner.write_new(tmp_path/'bad.json',{'value':float('nan')})
    assert not (tmp_path/'bad.json').exists()


def association_module():
    path=HERE.parent/'bridge_rgs/track_error_association.py'
    if not path.exists():path=HERE.parents[1]/'src/bridge_rgs/track_error_association.py'
    return runner.load_module(path)


def test_whole_group_exclusion_and_point_sensitivity_recompute():
    m=association_module()
    points=np.repeat(np.arange(3),4)
    images=np.array([1,2,3,4,1,2,3,4,5,6,7,8])
    groups=np.array([0,1,1,1,0,1,1,1,0,0,0,0])
    labels=np.array([2,0,0,0,2,2,2,2,2,0,0,0])
    prob=np.array([[.6,0,.4,0,0],[0,0,1,0,0],[1,0,0,0,0]],np.float32)
    result=m.analyze_track_error_association(points,images,labels,np.ones(12,bool),groups,prob,
        np.array([0,4,8]),point_count=3,query_image_ids=np.array([1,5]),
        duplicate_touched_point_indices=np.array([0]),bootstrap_repeats=50)
    a=result['query_arrays']; main=result['report']['main']
    assert a['reference_count'].tolist()==[3,3,0]
    assert a['eligible'].tolist()==[True,True,False]
    assert main['association']['difference_exposed_minus_compatible']['brier']==pytest.approx(.72)
    assert a['sensitivity_eligible'].tolist()==[False,True,False]
    sensitivity=result['report']['duplicate_exclusion_sensitivity']
    assert not sensitivity['gate_evaluated'] and 'bootstrap' not in sensitivity['association']
    assert sensitivity['cross_group_eligible']['tracks']==1


def test_joint_track_bootstrap_preserves_shared_side_offsets():
    m=association_module()
    result=m.joint_track_bootstrap(np.repeat(np.arange(3),2),np.tile([False,True],3),
        np.zeros(6),np.array([.1,.4,.2,.5,.3,.6]),repeats=100)
    assert result['tracks_on_both_sides']==3
    np.testing.assert_allclose(result['metrics']['brier']['difference_95_interval'],[.3,.3],atol=1e-15)


def test_bootstrap_missing_side_is_undefined_not_zero():
    m=association_module()
    result=m.joint_track_bootstrap(np.array([0,1]),np.array([True,True]),np.ones(2),np.ones(2),repeats=10)
    assert result['metrics']['brier']['difference_95_interval'] is None
    assert result['metrics']['brier']['undefined_replicates']==10
