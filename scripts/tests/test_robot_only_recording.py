"""The teaching CLI must work without opening a camera or saving images."""
from pathlib import Path
import runpy
import sys
import time
import numpy as np
from camera_calibration import capture
from camera_calibration.dataset import read_json,read_jsonl


def test_teaching_cli_is_robot_only(tmp_path,cfg,monkeypatch,capsys):
    root=Path(__file__).resolve().parents[2]
    cfg['outputs']=str(tmp_path)
    monkeypatch.syspath_prepend(str(root/'scripts'))
    monkeypatch.setattr(sys,'argv',['record_trajectory.py','--duration','0.15'])
    closed=[]
    class Robot:
        def __init__(self,*args): pass
        def metadata(self): return {'serial':cfg['robot']['serial']}
        def read(self):
            return {'q':[0.]*7,'dq':[0.]*7,'T_W_E':np.eye(4).tolist(),
                    'host_ns':time.monotonic_ns(),'state_age_s':0.,'fault':False}
        def close(self): closed.append(True)
    def forbidden(*args,**kwargs):
        raise AssertionError('Teaching must not access a camera or save images')
    monkeypatch.setattr('camera_calibration.devices.realsense.RealSense.__init__',forbidden)
    monkeypatch.setattr(capture,'save_image',forbidden)
    namespace=runpy.run_path(str(root/'scripts/record_trajectory.py'))
    monkeypatch.setitem(namespace['main'].__globals__,'Flexiv',Robot)
    monkeypatch.setitem(namespace['main'].__globals__,'load_config',lambda *args:cfg)
    namespace['main']()
    session=next(tmp_path.iterdir()); trajectory=session/'trajectory'
    assert set(p.name for p in trajectory.iterdir())=={'robot.json','states.jsonl','summary.json'}
    assert len(read_jsonl(trajectory/'states.jsonl'))>=3
    assert read_json(session/'manifest.json')['status']=='trajectory_recorded'
    assert read_json(trajectory/'summary.json')['recording_mode']=='robot_only'
    output=capsys.readouterr().out.lower()
    assert 'robot states' in output and 'images' not in output and 'marker' not in output
    assert closed==[True]
