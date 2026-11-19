"""Offline trajectory and command-loop regressions; never connect to hardware."""
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import pytest
from camera_calibration import motion


@pytest.mark.parametrize('reverse',[False,True])
def test_smooth_path_respects_curve_and_derivative_bounds(cfg,reverse):
    q=np.array([[0,0],[.01,.005],[.02,.018],[.03,.01],[.04,.006]])
    if reverse: q=q[::-1]
    curve=motion.smooth_path(q,cfg)
    assert np.allclose(curve(curve.x),q)
    assert np.allclose(curve.derivative()(curve.x[[0,-1]]),0,atol=1e-12)
    assert np.allclose(curve.derivative(2)(curve.x[[0,-1]]),0,atol=1e-12)
    assert np.max(abs(curve.derivative()(curve.x[1:-1])))>1e-4
    limits=[motion.replay_limits(cfg)[k] for k in ('max_velocity_rad_s','max_acceleration_rad_s2','max_jerk_rad_s3')]
    for n,limit in enumerate(limits,1):
        assert np.max(abs(curve.derivative(n).c))<=limit*(1+1e-9)
    for n in (0,1,2):
        controls=curve.derivative(n).c
        assert np.allclose(controls[-1,:-1],controls[0,1:],atol=1e-12)
    for i,(a,b) in enumerate(zip(q[:-1],q[1:])):
        controls=curve.c[:,i,:]; d=b-a
        alpha=np.clip((controls-a)@d/(d@d),0,1)
        assert np.max(abs(controls-(a+alpha[:,None]*d)))<=cfg['motion']['smooth_path_tolerance_rad']+1e-12


def test_duplicate_points_and_reversal(cfg):
    assert motion.smooth_path([[0]*7]*3,cfg) is None
    curve=motion.smooth_path([[0]*7,[0]*7,[.01]*7,[0]*7],cfg)
    assert np.allclose(curve(curve.x),[[0]*7,[.01]*7,[0]*7])
    assert np.allclose(curve.derivative()(curve.x[1]),0)


class Clock:
    def __init__(self): self.now=0.
    def monotonic(self): return self.now
    def monotonic_ns(self): return int(self.now*1e9)
    def sleep(self,duration): self.now+=duration


@pytest.mark.parametrize('failure',[None,'tracking','fault','delay'])
def test_stream_uses_nonzero_transit_velocity_and_checks_feedback(cfg,monkeypatch,failure):
    clock=Clock(); monkeypatch.setattr(motion,'time',clock)
    q=np.array([[0]*7,[.01]*7,[.02]*7])
    state={'q':q[0],'fault':False,'operational':True}; sent=[]; logs=[]
    def send(target,vmax,amax,target_velocity=None):
        assert vmax==motion.replay_limits(cfg)['max_velocity_rad_s']
        assert amax==motion.replay_limits(cfg)['max_acceleration_rad_s2']
        sent.append((np.array(target),np.array(target_velocity)))
        state['q']=target
        if failure=='tracking': state['q']=np.array(target)+1
        if failure=='fault': state['fault']=True
        if failure=='delay': clock.now+=.2
    robot=SimpleNamespace(send_joint=send); recorder=SimpleNamespace(latest=lambda:state)
    args=(robot,recorder,q,cfg,logs,[0,1,2])
    if failure:
        with pytest.raises(RuntimeError): motion.stream_leg(*args)
        assert len(sent)==1
    else:
        motion.stream_leg(*args)
        assert len(sent)>3 and len(logs)==len(sent)
        assert any(np.max(abs(v))>1e-4 for _,v in sent[:-1])
        assert np.allclose(sent[-1][0],q[-1]) and np.allclose(sent[-1][1],0)
        assert logs[-1]['trajectory_interval']==[0,2]


def test_threefold_speed_preserves_path_and_scales_all_derivatives(cfg):
    baseline=deepcopy(cfg); baseline['motion'].pop('replay_speed_scale',None)
    faster=deepcopy(baseline); faster['motion']['replay_speed_scale']=3
    q=[[0,0],[.01,.005],[.02,-.01],[.03,.02]]
    slow=motion.smooth_path(q,baseline); fast=motion.smooth_path(q,faster)
    assert np.allclose(fast.x,slow.x/3)
    t=np.linspace(0,slow.x[-1],101)
    for n in range(4):
        assert np.allclose(fast.derivative(n)(t/3),slow.derivative(n)(t)*3**n)
    assert motion.replay_limits(baseline)['replay_speed_scale']==1
    assert faster['capture']==baseline['capture']


@pytest.mark.parametrize('speed',[0,-1,float('nan'),float('inf')])
def test_invalid_speed_rejected(cfg,speed):
    cfg['motion']['replay_speed_scale']=speed
    with pytest.raises(ValueError,match='replay_speed_scale'):
        motion.smooth_path([[0]*7,[.01]*7],cfg)
