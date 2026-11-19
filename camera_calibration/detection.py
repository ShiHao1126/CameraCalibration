"""ArUco measurements and IPPE alternatives; robot poses are not fitted here."""
import cv2
import numpy as np
from collections import Counter
from pathlib import Path
from scipy.spatial.transform import Rotation
from .geometry import CameraModel, marker_points, mean_transform
from .dataset import read_json, read_jsonl, write_json, update_manifest


class MarkerDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg['dictionary']))
        params = cv2.aruco.DetectorParameters()
        self.r3kit = cfg['refinement']=='R3KIT'
        params.cornerRefinementMethod = (cv2.aruco.CORNER_REFINE_APRILTAG if self.r3kit else
                                        getattr(cv2.aruco, 'CORNER_REFINE_'+cfg['refinement']))
        if self.r3kit: params.cornerRefinementWinSize=5
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)

    def detect(self, image):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if self.r3kit and image.ndim==3 else image
        corners, ids, _ = self.detector.detectMarkers(gray if self.r3kit else image)
        if ids is None: return None, 'marker_not_found'
        matches = np.flatnonzero(ids.ravel()==self.cfg['id'])
        if len(matches)!=1: return None, 'target_id_missing_or_duplicate'
        raw=np.asarray(corners[matches[0]][0],np.float32).copy()
        refined=raw.copy()
        if self.r3kit:
            refined=cv2.cornerSubPix(gray,refined,(5,5),(-1,-1),
                                    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,.001))
        uv = np.asarray(refined, np.float64)
        margin = self.cfg['border_px']; height,width = image.shape[:2]
        if np.any(uv[:,0]<margin) or np.any(uv[:,0]>=width-margin) or np.any(uv[:,1]<margin) or np.any(uv[:,1]>=height-margin):
            return None, 'marker_near_image_border'
        shortest = float(np.linalg.norm(uv-np.roll(uv,1,axis=0),axis=1).min())
        if shortest < self.cfg['min_side_px']: return None, 'marker_too_small'
        return {'uv':uv.tolist(),'detector_uv':raw.tolist(),'shortest_side_px':shortest,
                'marker_id':self.cfg['id'],'refinement':self.cfg['refinement']}, None


def pnp_candidates(uv, camera, side=0.08):
    uv = np.asarray(uv, float)
    normalized = camera.unproject(uv)
    ok, rotations, translations, _ = cv2.solvePnPGeneric(marker_points(side), normalized,
                                                        np.eye(3), np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    results=[]
    if ok:
        for r,t in zip(rotations,translations):
            T=np.eye(4); T[:3,:3]=Rotation.from_rotvec(r.ravel()).as_matrix(); T[:3,3]=t.ravel()
            p=marker_points(side)@T[:3,:3].T+T[:3,3]
            if np.all(p[:,2]>1e-6):
                error=float(np.sqrt(np.mean(np.sum((camera.project(p)-uv)**2,axis=1))))
                results.append({'T_C_M':T.tolist(),'rms_px':error})
    return sorted(results,key=lambda x:x['rms_px'])


def aggregate(frames, camera_spec, cfg, sample_id):
    valid=[f for f in frames if f.get('detection') is not None]
    if len(valid)<cfg['capture']['min_valid_frames']:
        raise ValueError(f'Only {len(valid)} valid frames in burst')
    pixels=np.asarray([f['detection']['uv'] for f in valid])
    uv=np.median(pixels,axis=0)
    floor=cfg['capture']['corner_noise_floor_px']**2
    cov=np.array([np.cov(pixels[:,j,:],rowvar=False)+np.eye(2)*floor for j in range(4)])
    # Do not divide by burst length: frame noise may be correlated/systematic.
    transforms=[f['actual']['T_W_E'] for f in valid]
    T=mean_transform(transforms)
    candidates=pnp_candidates(uv,CameraModel(camera_spec),cfg['marker']['side_m'])
    if not candidates: raise ValueError('No valid positive-depth IPPE candidate')
    return {'sample_id':sample_id,'uv':uv.tolist(),'covariance_px2':cov.tolist(),
            'T_W_E':T.tolist(),'valid_frames':len(valid),'candidates':candidates}


def filter_capture(run):
    """Offline filtering of saved raw bursts. No hardware access or raw-file edits.

    Detection describes the encoded marker's four corners, not its mounting frame.
    Missing/cropped, too-small and near-border detections are rejected per frame.
    Training/test membership remains fixed by the acquisition plan.
    """
    run=Path(run); manifest=read_json(run/'manifest.json')
    if manifest['status']!='captured':
        raise ValueError('Filtering requires a completed capture run')
    cfg=manifest['config']; camera_spec=manifest['camera']
    CameraModel(camera_spec)
    detector=MarkerDetector(cfg['marker'])
    planned=manifest.get('plan',{}).get('samples')
    folders=([run/'samples'/sample['id'] for sample in planned] if planned is not None
             else sorted(p for p in (run/'samples').iterdir() if p.is_dir()))
    observations=[]; sample_reports=[]
    update_manifest(run,filter_status='running')
    try:
        for folder in folders:
            target=read_json(folder/'target.json')
            raw_frames=read_jsonl(folder/'frames.jsonl')
            frames=[]; detections=[]; reasons=Counter()
            for raw in raw_frames:
                image_path=(folder/raw['image']).resolve()
                if not image_path.is_relative_to(folder.resolve()):
                    raise ValueError('Image path escapes the sample directory')
                image=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
                if image is None:
                    detection,reason=None,'image_unreadable'
                elif image.shape[:2]!=(camera_spec['height'],camera_spec['width']):
                    detection,reason=None,'image_size_mismatch'
                else:
                    detection,reason=detector.detect(image)
                frames.append({**raw,'detection':detection,'rejection':reason})
                detections.append({'image':raw['image'],'detection':detection,'rejection':reason})
                if reason: reasons[reason]+=1
            write_json(folder/'detections.json',detections)
            valid_frames=sum(f['detection'] is not None for f in frames)
            row={'sample_id':target['id'],'split':target['split'],'total_frames':len(frames),
                 'valid_frames':valid_frames,'required_valid_frames':cfg['capture']['min_valid_frames'],
                 'frame_rejection_counts':dict(reasons)}
            try:
                observation=aggregate(frames,camera_spec,cfg,target['id'])
            except ValueError as e:
                row.update(status='rejected',reason='insufficient_valid_frames' if valid_frames<cfg['capture']['min_valid_frames'] else 'invalid_pose_observation',detail=str(e))
                write_json(folder/'rejected.json',{**row,'target':target})
                (folder/'observation.json').unlink(missing_ok=True)
            else:
                observation['split']=target['split']; observations.append(observation)
                row['status']='valid'
                write_json(folder/'observation.json',observation)
                (folder/'rejected.json').unlink(missing_ok=True)
            sample_reports.append(row)
        train=sum(o['split']=='train' for o in observations)
        test=sum(o['split']=='test' for o in observations)
        report={'schema_version':1,'stage':'after_capture','total_samples':len(folders),
                'valid_samples':len(observations),'rejected_samples':len(folders)-len(observations),
                'valid_train_samples':train,'valid_test_samples':test,
                'minimum_train_samples':cfg['solver']['min_train_poses'],'minimum_test_samples':3,
                'enough_samples_for_calibration':train>=cfg['solver']['min_train_poses'] and test>=3,
                'note':'Sample count is only a prerequisite; pose diversity and solve quality are checked during calibration.',
                'marker_config':cfg['marker'],'samples':sample_reports}
        write_json(run/'observations.json',observations)
        write_json(run/'filter_report.json',report)
        update_manifest(run,filter_status='completed',valid_samples=len(observations),rejected_samples=report['rejected_samples'])
        return report
    except BaseException as e:
        update_manifest(run,filter_status='failed',filter_error=str(e))
        raise
