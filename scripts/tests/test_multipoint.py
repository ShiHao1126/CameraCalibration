import copy
import numpy as np
import pytest
from camera_calibration.geometry import marker_points,inverse,se3_exp,CameraModel,distance
from camera_calibration.multipoint import grid_points,aggregate_internal,fit,evaluate,pack,weighted_huber
from conftest import synthetic_observations


def test_grid_coordinates_match_decoded_marker_frame():
    np.testing.assert_allclose(grid_points([[0,0],[8,0],[8,8],[0,8]]),marker_points())
    np.testing.assert_allclose(grid_points([[4,4],[1,7]]),[[0,0,0],[-.03,-.03,0]])
    with pytest.raises(ValueError):grid_points([[9,0]])


def test_burst_uses_identified_measured_points_not_homography_predictions():
    grid=[[i,j] for i in range(1,5) for j in range(2,6)]
    frames=[]
    for i in range(10):
        pixels=np.asarray(grid)*10.+[100,200]+(5 if i==0 else 0)
        frames.append({'detection':{'uv':[[999,999]]*4},'internal_fit':{'accepted':True,
                       'grid_points':grid,'image_points':pixels.tolist()}})
    sample={'sample_id':'test','split':'train','T_W_E':np.eye(4).tolist()}
    o=aggregate_internal(sample,frames)
    np.testing.assert_allclose(o['uv'],np.asarray(grid)*10+[100,200])
    assert len(o['uv'])==16 and all(len(p['frames'])==10 for p in o['point_support'])
    with pytest.raises(ValueError,match='without dropping'):aggregate_internal(sample,frames[:5])


@pytest.mark.parametrize('radial',[False,True])
def test_shared_transform_and_radial_recovery(radial):
    poses,spec,X,Y=synthetic_observations(count=28,noise=0,seed=42)
    true=copy.deepcopy(spec)
    if radial:true.update(model='brown_conrady',coeffs=[.04,-.06,0,0,0])
    camera=CameraModel(true);observations=[]
    for i,o in enumerate(poses):
        P=grid_points([[x,y] for x in range(1,8) for y in range(1,8) if (x+y)%2==i%2])
        C=inverse(X)@np.asarray(o['T_W_E'])@Y
        uv=camera.project(P@C[:3,:3].T+C[:3,3])
        observations.append({**o,'points_M':P.tolist(),'uv':uv.tolist()})
    seed={'T_W_C':X@se3_exp([.005,-.004,.006,.002,-.001,.003]),
          'T_E_M':Y@se3_exp([-.003,.005,-.001,.001,.003,-.002])}
    train=observations[:22];held=observations[22:]
    result=fit(train,seed,spec,radial)
    assert result['converged']
    assert evaluate(result,held)['rms_px']<1e-5
    assert distance(X,np.asarray(result['T_W_C']))[0]<1e-6
    assert distance(Y,np.asarray(result['T_E_M']))[0]<1e-6
    if radial:np.testing.assert_allclose(result['camera']['coeffs'],true['coeffs'],atol=1e-5)


def test_pose_cost_does_not_grow_with_duplicate_points():
    sample={'T_W_E':np.eye(4),'points_M':np.zeros((4,3)),'uv':np.zeros((4,2))}
    many={**sample,'points_M':np.zeros((12,3)),'uv':np.zeros((12,2))}
    z=np.array([.2,4.,1.,9.,.5,2.,5.,.3])
    cost=weighted_huber(z,pack([sample])[-1])[0].sum()
    repeated=weighted_huber(np.tile(z,3),pack([many])[-1])[0].sum()
    assert cost==pytest.approx(repeated)


def test_mixed_groups_have_equal_total_weight():
    sample={'T_W_E':np.eye(4),'points_M':np.zeros((16,3)),'uv':np.zeros((16,2)),
            'point_weights':[.5/4]*4+[.5/12]*12}
    weights=pack([sample])[-1]
    assert weights[:8].sum()==pytest.approx(weights[8:].sum())
    sample['point_weights'][0]=-1
    with pytest.raises(ValueError,match='weights'):pack([sample])
