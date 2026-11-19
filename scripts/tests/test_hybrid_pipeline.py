"""Default offline path, real upstream solver and hardware-close ordering."""
import importlib.util
import sys
from types import ModuleType
from pathlib import Path
import numpy as np
import pytest
from camera_calibration import hybrid
from camera_calibration.dataset import write_json,read_json,digest
from camera_calibration.geometry import marker_points,CameraModel,inverse,distance
from camera_calibration.multipoint import grid_points
from camera_calibration.report import calibrate_run
from conftest import synthetic_observations
from test_post_capture_filtering import raw_run


def test_default_failure_preserves_capture_and_records_quality_rejections(tmp_path,cfg):
    raw_run(tmp_path,cfg)
    paths=[tmp_path/'manifest.json',*tmp_path.glob('samples/*/frames.jsonl'),*tmp_path.glob('samples/*/images/*.png')]
    hashes={p:digest(p) for p in paths}
    with pytest.raises(ValueError,match='Insufficient valid poses'):
        calibrate_run(tmp_path)
    out=next((tmp_path/'calibration').iterdir())
    assert out.name.endswith('_hybrid')
    assert (out/'failure.json').exists() and not (out/'result.json').exists()
    report=read_json(out/'filter_report.json')
    assert report['valid_samples']==0  # Fewer than six supported burst frames.
    assert all(row['status']=='rejected' for row in report['samples'])
    assert all(digest(p)==h for p,h in hashes.items())


def test_default_pipeline_exports_frozen_model_and_fold_membership(tmp_path,cfg,monkeypatch):
    base,spec,X,Y=synthetic_observations(count=25,noise=0)
    outer=[{**o,'points_M':marker_points().tolist()} for o in base];inner=[]
    P=grid_points([[i,j] for i in range(1,5) for j in range(2,6)])
    for o in base:
        C=inverse(X)@np.asarray(o['T_W_E'])@Y
        inner.append({**o,'points_M':P.tolist(),'uv':CameraModel(spec).project(P@C[:3,:3].T+C[:3,3]).tolist()})
    write_json(tmp_path/'manifest.json',{'status':'captured','config':cfg,'camera':spec,'robot':{'serial':'SYNTHETIC'}})
    h=digest(tmp_path/'manifest.json')
    monkeypatch.setattr(hybrid,'extract',lambda *args:(base,outer,inner))
    calls=[]
    def compare(*args):calls.append(True);return []
    monkeypatch.setattr(hybrid,'original_r3kit',compare)
    out,result,validation=calibrate_run(tmp_path)
    assert calls and result['method']=='hybrid' and result['status_solve']=='ok'
    assert result['information']['rank']==14
    assert result['status_physical_accuracy']=='unverified'
    assert validation['status_internal_validation']=='passed' and validation['rms_px']<1e-4
    np.testing.assert_allclose(result['camera']['K'],spec['K'])
    np.testing.assert_allclose(np.asarray(result['T_C_W'])@result['T_W_C'],np.eye(4),atol=1e-10)
    assert 'T_B_C' not in result and digest(tmp_path/'manifest.json')==h
    cv=read_json(out/'cross_validation.json');assert cv['all_folds_ok']
    seen=[]
    for f in cv['folds']:
        seen.extend(f['held_ids'])
        assert not set(f['held_ids']) & set(f['result']['train_ids'])
        assert not set(validation['test_ids']) & set(f['held_ids']+f['result']['train_ids'])
    assert seen==result['train_ids']


def test_pinned_r3kit_backend_and_marker_frame_conversion():
    from camera_calibration.reference_r3kit.algos.calib.handeye import HandEyeCalibor
    base,spec,X,Y=synthetic_observations(count=25,noise=0)
    cal=HandEyeCalibor(marker_type='aruco',ext_calib_params={'dict_type':'6X6_1000','marker_length':80})
    cal.ext_calibor.image_size=(spec['height'],spec['width'])
    for o in base:
        cal.ext_calibor.obj_points.append(cal.ext_calibor.objp_marker.copy())
        cal.ext_calibor.img_points.append(np.asarray(o['uv'],np.float32))
        cal.b2g.append(np.asarray(o['T_W_E']))
    result=cal.run(np.asarray(spec['K'],float),opt_intrinsics=False,opt_distortion=False)
    N_M=np.diag([1.,-1.,-1.,1.]);N_M[:3,3]=[.04,.04,0]
    assert distance(X,inverse(result['g2c']))[0]<1e-5
    assert distance(Y,inverse(result['b2w'])@N_M)[0]<1e-5


def test_auto_default_calibrates_only_after_both_devices_closed(tmp_path,cfg,monkeypatch):
    scripts=Path(__file__).resolve().parents[1];monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('auto_for_test',scripts/'auto_calibrate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    events=[]
    class Device:
        def __init__(self,name):self.name=name
        def close(self):events.append(self.name+'_closed')
    for name,cls in [('flexiv','Flexiv'),('realsense','RealSense')]:
        stub=ModuleType('camera_calibration.devices.'+name)
        setattr(stub,cls,lambda cfg,name=name:Device(name))
        monkeypatch.setitem(sys.modules,stub.__name__,stub)
    plan=tmp_path/'plan.json';write_json(plan,{'config':cfg,'sample_count':40,'route_indices':[0]})
    monkeypatch.setattr(module,'replay_and_collect',lambda *a:tmp_path)
    def forbidden(*args):raise AssertionError('Hybrid must extract its own image measurements once')
    monkeypatch.setattr(module,'filter_capture',forbidden)
    def calibrate(path,method):
        assert events==['flexiv_closed','realsense_closed'] and method=='hybrid'
        events.append('offline_calibration')
        return tmp_path,{'status_solve':'ok','comparison_status':'completed'},{'status_internal_validation':'passed','status_physical_accuracy':'unverified'}
    monkeypatch.setattr(module,'calibrate_run',calibrate)
    monkeypatch.setattr(sys,'argv',['auto_calibrate.py','--plan',str(plan),'--execute'])
    module.main()
    assert events[-1]=='offline_calibration'
