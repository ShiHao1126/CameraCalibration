"""Control lifecycle tests use only fakes; no Flexiv SDK import or network."""
from pathlib import Path
import numpy as np
import pytest
from camera_calibration import capture
from camera_calibration.dataset import Journal,write_json,digest,read_json
from camera_calibration.trajectory import path_indices


def fixture_plan(tmp_path,cfg):
    session=tmp_path/'session'; states=[]
    for i in range(8):
        T=np.eye(4); T[0,3]=i*.001
        states.append({'q':[i*.003]*7,'dq':[0]*7,'T_W_E':T.tolist(),'host_ns':i*20_000_000,
                       'state_age_s':0,'fault':False,'operational':True})
    with Journal(session/'trajectory/states.jsonl') as f:
        for s in states: f.append(s)
    camera={'serial':cfg['camera']['serial'],'width':1920,'height':1080,'model':'none',
            'K':[[1000,0,960],[0,1000,540],[0,0,1]],'coeffs':[0]*5}
    write_json(session/'trajectory/camera.json',camera)
    sample={'id':'0000','trajectory_index':4,'target_q':states[4]['q'],'target_T_W_E':states[4]['T_W_E'],'split':'train'}
    plan={'session':str(session),'config':cfg,'robot_serial':cfg['robot']['serial'],'camera_serial':cfg['camera']['serial'],
          'trajectory_sha256':digest(session/'trajectory/states.jsonl'),
          'route_indices':path_indices(np.array([s['q'] for s in states]),[4],cfg),'samples':[sample]}
    path=session/'plan.json'; write_json(path,plan)
    return path,plan,states,camera


class FakeRobot:
    def __init__(self,state): self.state=dict(state); self.sent=[]; self.started=False; self.stopped=False
    def read(self): return self.state
    def metadata(self): return {'serial':'Rizon4R-062115'}
    def begin_motion(self,timeout_s): self.started=True
    def send_joint(self,q,*args): self.sent.append(list(q)); self.state={**self.state,'q':list(q)}
    def stop(self): self.stopped=True


class FakeRecorder:
    def __init__(self,robot,*args): self.robot=robot; self.closed=False
    def start(self): return self
    def latest(self): return self.robot.read()
    def close(self): self.closed=True


def test_motion_exception_always_stops_and_keeps_record(tmp_path,cfg,monkeypatch):
    path,plan,states,spec=fixture_plan(tmp_path,cfg)
    robot=FakeRobot(states[0]); camera=type('Camera',(),{'spec':spec})()
    monkeypatch.setattr(capture,'StateRecorder',FakeRecorder)
    def fail(*args): raise RuntimeError('simulated tracking fault')
    monkeypatch.setattr(capture,'_wait_waypoint',fail)
    with pytest.raises(RuntimeError,match='simulated'): capture.replay_and_collect(path,robot,camera,cfg)
    assert robot.started and robot.stopped
    manifests=list((Path(plan['session'])/'captures').glob('*/manifest.json'))
    assert read_json(manifests[0])['status']=='failed'


@pytest.mark.parametrize('legacy_camera_metadata',[False,True])
def test_reverse_route_and_capture_then_stop(tmp_path,cfg,monkeypatch,legacy_camera_metadata):
    path,plan,states,spec=fixture_plan(tmp_path,cfg)
    if not legacy_camera_metadata:
        (Path(plan['session'])/'trajectory/camera.json').unlink()
    robot=FakeRobot(states[-1]); camera=type('Camera',(),{'spec':spec})()
    monkeypatch.setattr(capture,'StateRecorder',FakeRecorder)
    waited=[]; legs=[]
    monkeypatch.setattr(capture,'_wait_waypoint',lambda robot,recorder,q,*args:waited.append(list(q)))
    def stream(robot,recorder,points,cfg,commands,indices):
        legs.append(indices)
        for q in points[1:]: robot.send_joint(q)
    monkeypatch.setattr(capture,'stream_leg',stream)
    monkeypatch.setattr(capture,'capture_sample',lambda folder,target,*args:{'sample_id':target['id']})
    out=capture.replay_and_collect(path,robot,camera,cfg)
    assert robot.stopped
    assert robot.sent==[states[i]['q'] for i in plan['route_indices'][::-1]]
    assert waited==[states[i]['q'] for i in (7,4,0)]
    assert legs[0][0]==7 and legs[0][-1]==legs[1][0]==4 and legs[1][-1]==0
    assert read_json(out/'manifest.json')['status']=='captured'
    assert read_json(out/'manifest.json')['filter_status']=='pending'
    assert not (out/'observations.json').exists()
    assert len(read_json(out/'raw_samples.json'))==1


def test_modified_route_fails_before_enabling(tmp_path,cfg):
    path,plan,states,spec=fixture_plan(tmp_path,cfg)
    plan['route_indices']=[0,7]; write_json(path,plan)
    robot=FakeRobot(states[0]); camera=type('Camera',(),{'spec':spec})()
    with pytest.raises(ValueError,match='verified taught path'): capture.replay_and_collect(path,robot,camera,cfg)
    assert not robot.started and not robot.sent
