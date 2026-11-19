"""Regression against the original r3kit corner pipeline and saved real failures."""
from pathlib import Path
import cv2
import numpy as np
import pytest
from camera_calibration.detection import MarkerDetector
from camera_calibration.dataset import read_json


@pytest.mark.parametrize('name',['0027_001','0028_000','0037_000'])
def test_original_r3kit_corners_on_real_frame(cfg,name):
    fixture=Path(__file__).parent/'fixtures'/'r3kit_corners'/f'{name}.png'
    image=cv2.imread(str(fixture)); assert image is not None
    marker={**cfg['marker'],'refinement':'R3KIT'}
    actual,reason=MarkerDetector(marker).detect(image)
    assert reason is None
    references=read_json(fixture.parent/'provenance.json')['images']
    expected=np.asarray(next(r['expected_uv_r3kit'] for r in references if r['file']==fixture.name))
    np.testing.assert_allclose(actual['uv'],expected,atol=1e-6,rtol=0)
    # With these actual failures the replaced SUBPIX-only pipeline chose a
    # different contour/corner; this test must exercise more than a clean tag.
    old,reason=MarkerDetector({**marker,'refinement':'SUBPIX'}).detect(image)
    assert reason is None
    assert np.max(np.linalg.norm(np.asarray(old['uv'])-expected,axis=1))>5
