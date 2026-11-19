"""Reprocess saved images with the original r3kit corners, preserving prior results."""
from _common import parser,run
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import cv2
from camera_calibration.dataset import read_json,read_jsonl,write_json,digest
from camera_calibration.detection import MarkerDetector,aggregate
from camera_calibration.solver import solve
from camera_calibration.validation import evaluate
from camera_calibration.report import render_report


def reprocess(source):
    source=Path(source).resolve(); manifest=read_json(source/'manifest.json')
    if manifest['status']!='captured': raise ValueError('Requires a completed capture')
    cfg=deepcopy(manifest['config']); cfg['marker']['refinement']='R3KIT'
    stamp=datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    out=source/'calibration'/f'{stamp}_r3kit_corners_joint'; out.mkdir(parents=True)
    write_json(out/'provenance.json',{'source_capture':str(source),'source_manifest_sha256':digest(source/'manifest.json'),
               'config':cfg,'changed_fields':['marker.refinement'],
               'reference':'https://github.com/dadadadawjb/r3kit/blob/95ce4caf1cb2a457b65264ba57981dc54ac6daf4/r3kit/algos/calib/aruco.py',
               'original_data_preserved':True,'split_policy':'unchanged planned train/test membership',
               'evaluation_note':'These held-out images have already been inspected during debugging; this is a diagnostic re-evaluation, not a fresh independent acceptance test.'})
    detector=MarkerDetector(cfg['marker']); observations=[]; reports=[]
    try:
        for target in manifest['plan']['samples']:
            folder=source/'samples'/target['id']; frames=[]; detections=[]
            for raw in read_jsonl(folder/'frames.jsonl'):
                path=(folder/raw['image']).resolve()
                if not path.is_relative_to(folder): raise ValueError('Image path escapes sample folder')
                image=cv2.imread(str(path))
                if image is None: detection,reason=None,'image_unreadable'
                elif image.shape[:2]!=(manifest['camera']['height'],manifest['camera']['width']):
                    detection,reason=None,'image_size_mismatch'
                else: detection,reason=detector.detect(image)
                frames.append({**raw,'detection':detection,'rejection':reason})
                detections.append({'image':str(path.relative_to(source)),'detection':detection,'rejection':reason})
            write_json(out/'detections'/f'{target["id"]}.json',detections)
            row={'sample_id':target['id'],'split':target['split'],'valid_frames':sum(f['detection'] is not None for f in frames)}
            try:
                obs=aggregate(frames,manifest['camera'],cfg,target['id'])
                obs['split']=target['split']; observations.append(obs); row['status']='valid'
            except ValueError as exc: row.update(status='rejected',reason=str(exc))
            reports.append(row)
            print(f'Processed {len(reports)}/{len(manifest["plan"]["samples"])}',flush=True)
        write_json(out/'observations.json',observations); write_json(out/'filter_report.json',reports)
        train=[o for o in observations if o['split']=='train']; test=[o for o in observations if o['split']=='test']
        if len(train)<cfg['solver']['min_train_poses'] or len(test)<3: raise ValueError('Insufficient valid training/test poses')
        result=solve(train,manifest['camera'],cfg)
        validation=evaluate(result,test,cfg)
        result.update(config=cfg,capture_run=str(source),robot_serial=manifest['robot']['serial'],
                      status_internal_validation=validation['status_internal_validation'])
        write_json(out/'result.json',result); write_json(out/'validation.json',validation)
        render_report(out,result,validation)
    except BaseException as exc:
        write_json(out/'failure.json',{'error':str(exc)}); raise
    print(f'RESULT={out}\nTrain RMS={result["train_rms_px"]:.4f} px; '
          f'test RMS={validation["rms_px"]:.4f} px; validation={validation["status_internal_validation"]}; physical=unverified')
    return out


def main():
    p=parser(__doc__); p.add_argument('--run',required=True,type=Path); a=p.parse_args()
    if a.config: p.error('Only marker refinement is changed; the capture configuration is otherwise retained')
    reprocess(a.run)


if __name__=='__main__': run(main)
