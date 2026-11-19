"""Known image geometry tests for experimental image-only refinements."""
import cv2
import numpy as np
import pytest
from investigate_residuals import edge_corners,internal_corners


@pytest.mark.parametrize('refiner',[edge_corners,internal_corners])
def test_recovers_known_projected_marker_boundary(refiner):
    marker=cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000),0,256)
    truth=np.array([[160,90],[415,115],[460,365],[140,345]],np.float32)
    src=np.array([[-.5,-.5],[255.5,-.5],[255.5,255.5],[-.5,255.5]],np.float32)
    H=cv2.getPerspectiveTransform(src,truth)
    image=cv2.warpPerspective(marker,H,(640,480),borderValue=255)
    image=cv2.GaussianBlur(image,(0,0),.7)
    image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    initial=truth+np.array([.8,-.6],np.float32)
    measured,info=refiner(image,initial)
    assert info['accepted']
    assert np.sqrt(np.mean(np.sum((measured-truth)**2,axis=1)))<.5


def test_edge_refinement_rejects_uninformative_image():
    image=np.full((400,400,3),127,np.uint8);uv=np.array([[50,50],[350,50],[350,350],[50,350]])
    measured,info=edge_corners(image,uv)
    assert not info['accepted']
    np.testing.assert_array_equal(measured,uv)
