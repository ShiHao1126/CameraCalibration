"""Image visibility is not required when choosing/replaying teaching poses."""
from pathlib import Path
import cv2
import numpy as np
import pytest
from camera_calibration import capture
from camera_calibration.dataset import Journal,write_json,read_json,read_jsonl,digest
from camera_calibration.detection import filter_capture
from camera_calibration.report import calibrate_run
from camera_calibration.trajectory import make_plan
from test_trajectory_capture import trajectory


@pytest.mark.parametrize('write_invisible_frames',[False,True])
def test_planning_does_not_depend_on_teaching_images(tmp_path,cfg,write_invisible_frames):
    write_json(tmp_path/'manifest.json',{'status':'trajectory_recorded',
               'robot_serial':cfg['robot']['serial'],'camera_serial':cfg['camera']['serial']})
    with Journal(tmp_path/'trajectory/states.jsonl') as log:
        for s in trajectory(): log.append(s)
    if write_invisible_frames:
        with Journal(tmp_path/'trajectory/frames.jsonl') as log:
            for i in (0,50,100): log.append({'robot_index':i,'detection':None,'rejection':'marker_not_found'})
    _,plan,_=make_plan(tmp_path,20,cfg)
    assert len(plan['samples'])==20
    assert plan['selection_basis']=='robot_pose_only'
    assert plan['visibility_filter_stage']=='after_capture'


def test_raw_capture_keeps_all_invisible_frames_without_detection(tmp_path,cfg,monkeypatch):
    cfg['capture'].update(fresh_frames=1,burst_frames=3,min_valid_frames=2)
    state={'q':[0.]*7,'dq':[0.]*7,'host_ns':0,'T_W_E':np.eye(4).tolist()}
    class Recorder:
        def latest(self): return state
        def window(self,*args): return [{**state,'host_ns':1_000_000_000}]
    class Camera:
        def read(self): return np.full((480,640,3),255,np.uint8),{'frame_number':1}
    monkeypatch.setattr(capture,'wait_stable',lambda *args:state)
    def forbidden(*args): raise AssertionError('Detection must not happen during point capture')
    monkeypatch.setattr('camera_calibration.detection.MarkerDetector',forbidden)
    target={'id':'0000','split':'train','target_q':state['q'],'target_T_W_E':state['T_W_E']}
    result=capture.capture_sample(tmp_path,target,Recorder(),Camera(),cfg)
    assert result['status']=='raw_captured'
    assert len(list((tmp_path/'images').glob('*.png')))==3
    assert (tmp_path/'target.json').exists() and (tmp_path/'actual.json').exists()
    assert all('detection' not in f for f in read_jsonl(tmp_path/'frames.jsonl'))
    assert not (tmp_path/'observation.json').exists() and not (tmp_path/'rejected.json').exists()


def image_fixture(kind):
    image=np.full((480,640,3),255,np.uint8)
    tag=cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000),0,160)
    tag=cv2.cvtColor(tag,cv2.COLOR_GRAY2BGR)
    if kind=='valid': image[160:320,240:400]=tag
    elif kind=='cropped': image[160:320,0:100]=tag[:,60:]
    return image


def raw_run(root,cfg):
    cfg['capture'].update(burst_frames=3,min_valid_frames=2)
    spec={'serial':'SYNTHETIC','width':640,'height':480,'K':[[500,0,320],[0,500,240],[0,0,1]],
          'model':'none','coeffs':[0]*5}
    write_json(root/'manifest.json',{'status':'captured','filter_status':'pending',
               'config':cfg,'camera':spec,'robot':{'serial':'SYNTHETIC'}})
    for sample_id,kinds,split in [('0000',['valid','valid','cropped'],'train'),
                                 ('0001',['cropped','blank','blank'],'test')]:
        folder=root/'samples'/sample_id
        write_json(folder/'target.json',{'id':sample_id,'split':split,'target_T_W_E':np.eye(4)})
        with Journal(folder/'frames.jsonl') as log:
            for i,kind in enumerate(kinds):
                rel=f'images/{i:03d}.png'; capture.save_image(folder/rel,image_fixture(kind))
                log.append({'image':rel,'actual':{'T_W_E':np.eye(4).tolist()}})


def test_offline_filter_preserves_raw_and_reports_rejections(tmp_path,cfg):
    raw_run(tmp_path,cfg)
    # Stale successful output must not survive when its sample is now rejected.
    write_json(tmp_path/'samples/0001/observation.json',{'stale':True})
    write_json(tmp_path/'samples/0000/rejected.json',{'stale':True})
    paths=[*tmp_path.glob('samples/*/frames.jsonl'),*tmp_path.glob('samples/*/images/*.png')]
    original={p:digest(p) for p in paths}
    report=filter_capture(tmp_path)
    assert (report['total_samples'],report['valid_samples'],report['rejected_samples'])==(2,1,1)
    assert report['valid_train_samples']==1 and report['valid_test_samples']==0
    assert report['enough_samples_for_calibration'] is False
    obs=read_json(tmp_path/'observations.json')
    assert [o['sample_id'] for o in obs]==['0000'] and obs[0]['valid_frames']==2
    rejected=read_json(tmp_path/'samples/0001/rejected.json')
    assert rejected['reason']=='insufficient_valid_frames'
    assert sum(rejected['frame_rejection_counts'].values())==3
    assert not (tmp_path/'samples/0001/observation.json').exists()
    assert not (tmp_path/'samples/0000/rejected.json').exists()
    assert all(digest(p)==h for p,h in original.items())
    assert read_json(tmp_path/'manifest.json')['filter_status']=='completed'
    # Re-filtering is deterministic and leaves raw images/logs intact.
    assert filter_capture(tmp_path)==report
    assert all(digest(p)==h for p,h in original.items())


def test_calibration_filters_first_and_refuses_insufficient_valid_poses(tmp_path,cfg):
    raw_run(tmp_path,cfg)
    with pytest.raises(ValueError,match='Insufficient valid poses after filtering'):
        calibrate_run(tmp_path,method='joint')
    assert read_json(tmp_path/'filter_report.json')['valid_samples']==1
    assert list((tmp_path/'calibration').glob('*/failure.json'))
    assert not list((tmp_path/'calibration').glob('*/result.json'))
