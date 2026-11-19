"""Render a synthetic ArUco dataset and run the complete offline image-to-report pipeline."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
from conftest import synthetic_observations
from camera_calibration.models import load_config
from camera_calibration.dataset import create_session,write_json,Journal
from camera_calibration.capture import save_image
from camera_calibration.detection import MarkerDetector,aggregate
from camera_calibration.geometry import distance
from camera_calibration.report import calibrate_run


def main():
    cfg=load_config(); cfg['camera']['serial']='SYNTHETIC'; cfg['robot']['serial']='SYNTHETIC-NO-HARDWARE'
    cfg['capture']['burst_frames']=6; cfg['capture']['min_valid_frames']=6
    cfg['solver']['subset_count']=4; cfg['solver']['max_initializations']=4
    session=create_session(cfg,'synthetic'); captured=session/'captures/synthetic'; captured.mkdir(parents=True)
    source,spec,X,Y=synthetic_observations(count=30)
    write_json(session/'truth.json',{'T_W_C':X,'T_E_M':Y,'synthetic_only':True})
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    tag=cv2.aruco.generateImageMarker(dictionary,0,800)
    source_corners=np.array([[0,0],[799,0],[799,799],[0,799]],np.float32)
    detector=MarkerDetector(cfg['marker']); obs=[]
    for i,o in enumerate(source):
        H=cv2.getPerspectiveTransform(source_corners,np.array(o['uv'],np.float32))
        image=cv2.warpPerspective(tag,H,(1920,1080),borderValue=255)
        image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        folder=captured/'samples'/f'{i:04d}'; frames=[]
        with Journal(folder/'frames.jsonl') as log:
            for j in range(6):
                rel=f'images/{j:03d}.png'; save_image(folder/rel,image)
                det,reason=detector.detect(image)
                row={'image':rel,'detection':det,'rejection':reason,'actual':{'T_W_E':o['T_W_E']}}
                frames.append(row); log.append(row)
        item=aggregate(frames,spec,cfg,str(i)); item['split']=o['split']; obs.append(item)
        write_json(folder/'observation.json',item)
    write_json(captured/'observations.json',obs)
    write_json(captured/'manifest.json',{'status':'captured','config':cfg,'camera':spec,
                'robot':{'serial':'SYNTHETIC-NO-HARDWARE'},'synthetic_only':True})
    comparisons=[]
    for method,multi in [('baseline',False),('joint',False),('joint',True)]:
        out,result,validation=calibrate_run(captured,method,multi)
        dt,dr=distance(X,np.array(result['T_W_C']))
        comparisons.append({'method':method,'multi_initialization':multi,'result':str(out),
                            'synthetic_translation_error_mm':dt*1000,'synthetic_rotation_error_deg':float(np.rad2deg(dr)),
                            'holdout_rms_px':validation['rms_px'],'status':result['status_solve'],
                            'internal_validation':validation['status_internal_validation']})
    write_json(session/'comparison.json',comparisons)
    print('SYNTHETIC_OUTPUT='+str(session))
    for row in comparisons: print(row)
    # The reference baseline is measured, not required to meet the new solver's gate.
    if any(row['status']!='ok' or row['holdout_rms_px']>1 for row in comparisons if row['method']=='joint'):
        raise RuntimeError('Synthetic pipeline did not pass')


if __name__=='__main__': main()
