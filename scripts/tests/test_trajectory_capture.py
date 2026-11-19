import numpy as np
import pytest
from camera_calibration.trajectory import path_indices,replay_direction,validate_recording,choose_indices
from camera_calibration.capture import stationary
from camera_calibration.geometry import se3_exp


def trajectory(n=300):
    t=np.linspace(0,1,n); states=[]
    for i,x in enumerate(t):
        q=np.array([.15*np.sin(3*x),.2*x,.1*np.cos(6*x),0,0,0,0])
        T=se3_exp([.4*np.sin(x*3),.4*np.cos(x*4),.5*x,.2*x,.1*np.sin(x*5),.7])
        states.append({'q':q.tolist(),'dq':[0]*7,'T_W_E':T.tolist(),'host_ns':int(i*20_000_000),
                       'state_age_s':0,'fault':False})
    return states


def test_route_preserves_samples_and_taught_curve(cfg):
    states=trajectory(); q=validate_recording(states,cfg)
    selected=choose_indices(states,list(range(0,len(states),3)),20,cfg)
    route=path_indices(q,selected,cfg)
    assert set(selected)<=set(route)
    assert route[0]==0 and route[-1]==len(q)-1
    assert np.max(abs(np.diff(q[route],axis=0)))<=cfg['motion']['max_waypoint_step_rad']
    for a,b in zip(route[:-1],route[1:]):
        d=q[b]-q[a]; alpha=np.clip((q[a:b+1]-q[a])@d/(d@d),0,1)
        assert np.max(abs(q[a:b+1]-(q[a]+alpha[:,None]*d)))<=cfg['motion']['path_simplification_rad']+1e-12


def test_replay_rejects_shortcut_from_middle():
    states=trajectory()
    assert replay_direction(states[0]['q'],states,.01)==1
    assert replay_direction(states[-1]['q'],states,.01)==-1
    with pytest.raises(ValueError,match='not at'): replay_direction(states[150]['q'],states,.01)


def test_recording_gap_and_stale_feedback_rejected(cfg):
    states=trajectory(); states[2]['host_ns']=500_000_000
    with pytest.raises(ValueError): validate_recording(states,cfg)
    states=trajectory(); states[2]['state_age_s']=1
    with pytest.raises(ValueError,match='stale'): validate_recording(states,cfg)


def test_motion_during_capture_rejected(cfg):
    s={'q':[0]*7,'dq':[0]*7,'host_ns':0,'T_W_E':np.eye(4).tolist()}
    m={**s,'host_ns':1_000_000_000,'T_W_E':np.eye(4).tolist()}
    assert stationary([s,m],cfg)
    m['T_W_E'][0][3]=.001
    assert not stationary([s,m],cfg)
    m={**s,'host_ns':1_000_000_000,'q':[.001]*7}
    assert not stationary([s,m],cfg)


def test_stationary_accepts_noisy_dq_but_rejects_joint_motion(cfg):
    # The failed real capture had dq spikes up to .00683 rad/s while q spanned
    # only .0000162 rad over ten seconds. Raw dq alone is not a stillness gate.
    states=[{'q':[1e-5*np.sin(i)]*7,'dq':[.007*(-1)**i]*7,
             'host_ns':i*20_000_000,'T_W_E':np.eye(4).tolist()} for i in range(51)]
    assert stationary(states,cfg)
    # Null-space motion can leave the flange fixed; joint feedback must gate it.
    states[25]['q']=[.002]*7
    assert not stationary(states,cfg)
    # Small displacement over a short window can still be fast motion.
    states=states[:2]; states[-1]['q']=[.0002]*7
    assert not stationary(states,cfg)
