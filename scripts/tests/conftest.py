import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from camera_calibration.models import load_config
from camera_calibration.geometry import CameraModel,marker_points,inverse,se3_exp


@pytest.fixture
def cfg():
    c=load_config(); c['solver']['subset_count']=2; c['solver']['max_initializations']=2
    return c


def synthetic_observations(count=30,noise=0,seed=13,degenerate=False):
    rng=np.random.default_rng(seed)
    spec={'K':[[1365,0,960],[0,1364,540],[0,0,1]],'coeffs':[0]*5,'model':'none',
          'width':1920,'height':1080,'fps':8,'serial':'SYNTHETIC','name':'Synthetic pinhole'}
    camera=CameraModel(spec)
    X=se3_exp([.3,-.2,.1,.4,-.3,.2]); Y=se3_exp([.1,.2,-.1,.03,.02,.12])
    observations=[]
    for i in range(count):
        C=np.eye(4)
        C[:3,:3]=Rotation.from_rotvec(np.zeros(3) if degenerate else rng.uniform(-.55,.55,3)).as_matrix()@Rotation.from_euler('x',180,degrees=True).as_matrix()
        C[:3,3]=rng.uniform([-.15,-.12,.6],[.15,.12,1.])
        uv=camera.project(marker_points()@C[:3,:3].T+C[:3,3])+rng.normal(0,noise,(4,2))
        G=X@C@inverse(Y)
        observations.append({'sample_id':str(i),'T_W_E':G.tolist(),'uv':uv.tolist(),
            'covariance_px2':np.tile(np.eye(2)*.15**2,(4,1,1)).tolist(),
            'candidates':[{'T_C_M':C.tolist(),'rms_px':0}], 'split':'test' if i%5==0 else 'train'})
    return observations,spec,X,Y
