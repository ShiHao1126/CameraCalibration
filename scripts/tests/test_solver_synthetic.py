import numpy as np
from conftest import synthetic_observations
from camera_calibration.geometry import distance,CameraModel
from camera_calibration.solver import solve,information,unpack
from camera_calibration.validation import evaluate
from camera_calibration.detection import pnp_candidates


def test_exact_recovery_and_frame_direction(cfg):
    obs,spec,X,Y=synthetic_observations()
    train=[o for o in obs if o['split']=='train']; test=[o for o in obs if o['split']=='test']
    for method in ['baseline','joint']:
        r=solve(train,spec,cfg,method,multi=False)
        assert r['status_solve']=='ok'
        # OpenCV calibrateCamera requires float32 object/image arrays.
        assert max(distance(X,np.array(r['T_W_C'])))<1e-5
        assert max(distance(Y,np.array(r['T_E_M'])))<1e-5
        report=evaluate(r,test,cfg)
        assert report['rms_px']<1e-3
        assert report['status_physical_accuracy']=='unverified'


def test_noise_with_real_ippe_and_multistart(cfg):
    obs,spec,X,Y=synthetic_observations(count=24,noise=.12)
    camera=CameraModel(spec)
    for o in obs: o['candidates']=pnp_candidates(o['uv'],camera)
    r=solve(obs,spec,cfg)
    assert r['status_solve']=='ok'
    assert distance(X,np.array(r['T_W_C']))[0]<.003
    assert r['train_rms_px']<.35


def test_translation_only_does_not_constrain_all_parameters(cfg):
    obs,spec,X,Y=synthetic_observations(degenerate=True)
    G,uv,W=unpack(obs)
    assert information(X,Y,G,uv,W,CameraModel(spec),cfg)['rank']==9


def test_outlier_does_not_dominate_joint_fit(cfg):
    obs,spec,X,Y=synthetic_observations(count=30,noise=.1)
    obs[5]['uv'][0][0]+=25
    r=solve(obs,spec,cfg,multi=False)
    assert distance(X,np.array(r['T_W_C']))[0]<.003
    assert r['status_physical_accuracy']=='unverified'
