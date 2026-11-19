import numpy as np
import pytest
from camera_calibration.geometry import CameraModel,pose_matrix,matrix_pose,se3_exp,inverse


@pytest.mark.parametrize('model',['none','brown_conrady','modified_brown_conrady'])
def test_projection_matches_sdk_and_inverts(model):
    rs=pytest.importorskip('pyrealsense2')
    intr=rs.intrinsics(); intr.width=1920; intr.height=1080; intr.fx=1400; intr.fy=1390; intr.ppx=950; intr.ppy=530
    intr.model=getattr(rs.distortion,model); intr.coeffs=[.1,-.05,.002,-.003,.01]
    camera=CameraModel({'K':[[1400,0,950],[0,1390,530],[0,0,1]],'coeffs':list(intr.coeffs),'model':model})
    rng=np.random.default_rng(2); points=rng.uniform([-.45,-.3,.6],[.45,.3,1.],(100,3))
    expected=np.array([rs.rs2_project_point_to_pixel(intr,p.tolist()) for p in points])
    uv=camera.project(points)
    np.testing.assert_allclose(uv,expected,atol=.0004,rtol=0)
    np.testing.assert_allclose(camera.unproject(uv),points[:,:2]/points[:,2,None],atol=1e-10)


def test_actual_d415_zero_inverse_brown():
    rs=pytest.importorskip('pyrealsense2')
    intr=rs.intrinsics(); intr.width=1920; intr.height=1080; intr.fx=1365.7306; intr.fy=1364.2703; intr.ppx=944.3655; intr.ppy=531.21265
    intr.model=rs.distortion.inverse_brown_conrady; intr.coeffs=[0]*5
    spec={'K':[[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]],'coeffs':[0]*5,'model':'inverse_brown_conrady'}
    camera=CameraModel(spec); uv=np.array([[0.,0.],[1919.,1079.],[960.,540.]])
    points=np.array([rs.rs2_deproject_pixel_to_point(intr,p.tolist(),.8) for p in uv])
    np.testing.assert_allclose(camera.project(points),uv,atol=.0002)
    spec['coeffs'][0]=.01
    with pytest.raises(ValueError,match='Nonzero'): CameraModel(spec)


def test_quaternion_order_and_inverse():
    T=se3_exp([.3,-.2,.8,.1,.2,.3])
    np.testing.assert_allclose(pose_matrix(matrix_pose(T)),T,atol=1e-14)
    np.testing.assert_allclose(inverse(T)@T,np.eye(4),atol=1e-14)
    with pytest.raises(ValueError): pose_matrix([0]*7)
