"""Offline production pipeline: raw bursts -> hybrid solve -> frozen validation.

No device imports. Every invocation creates independent derived artifacts and
retains the capture's planned train/test membership, images and state logs.
"""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from .dataset import read_json,read_jsonl,write_json,digest
from .detection import MarkerDetector,aggregate
from .features import internal_corners,edge_corners
from .geometry import inverse,marker_points
from .multipoint import aggregate_internal,fit,evaluate
from .solver import solve


def extract(run,out,manifest,cfg):
    """Extract both point families from raw PNGs without residual-based selection."""
    detector=MarkerDetector(cfg['marker']);spec=manifest['camera']
    original=[];outer=[];internal=[];reports=[];hashes={}
    planned=manifest.get('plan',{}).get('samples')
    if planned is None:
        planned=[read_json(p/'target.json') for p in sorted((run/'samples').iterdir()) if p.is_dir()]
    for target in planned:
        sid=target['id'];folder=run/'samples'/sid;frames=[];edges=[];details=[];edge_details=[];inner_details=[]
        hashes[str((folder/'frames.jsonl').relative_to(run))]=digest(folder/'frames.jsonl')
        for raw in read_jsonl(folder/'frames.jsonl'):
            path=(folder/raw['image']).resolve()
            if not path.is_relative_to(folder.resolve()):raise ValueError('Image path escapes sample folder')
            hashes[str(path.relative_to(run))]=digest(path)
            image=cv2.imread(str(path))
            if image is None:detection,reason=None,'image_unreadable'
            elif image.shape[:2]!=(spec['height'],spec['width']):detection,reason=None,'image_size_mismatch'
            else:detection,reason=detector.detect(image)
            edge=deepcopy(detection);inner=deepcopy(detection);ei=ii=None
            if detection:
                points,ei=edge_corners(image,detection['uv']);edge['uv']=points.tolist()
                virtual,ii=internal_corners(image,detection['uv']);inner['uv']=virtual.tolist()
            frames.append({**raw,'detection':detection})
            edges.append({**raw,'detection':edge})
            details.append({'image':str(path.relative_to(run)),'detection':detection,'rejection':reason})
            edge_details.append({'image':raw['image'],'detection':edge,'edge_fit':ei})
            inner_details.append({'image':raw['image'],'detection':inner,'internal_fit':ii})
        write_json(out/'detections'/f'{sid}.json',details)
        write_json(out/'edge_detections'/f'{sid}.json',edge_details)
        write_json(out/'internal_detections'/f'{sid}.json',inner_details)
        row={'sample_id':sid,'split':target['split'],'total_frames':len(frames),
             'valid_frames':sum(f['detection'] is not None for f in frames),
             'edge_fallback_frames':sum(d['edge_fit'] is not None and not d['edge_fit']['accepted'] for d in edge_details)}
        try:
            base=aggregate(frames,spec,cfg,sid);base['split']=target['split']
            a=aggregate(edges,spec,cfg,sid);a.update(split=target['split'],points_M=marker_points(.08).tolist())
            b=aggregate_internal(base,inner_details,min_frames=max(6,cfg['capture']['min_valid_frames']))
        except ValueError as exc:row.update(status='rejected',reason=str(exc))
        else:
            original.append(base);outer.append(a);internal.append(b)
            row.update(status='valid',internal_point_count=len(b['uv']))
        reports.append(row)
        print(f'Hybrid images: {len(reports)}/{len(planned)} poses, {len(original)} valid',flush=True)
    for name,obs in [('observations',original),('outer_observations',outer),('internal_observations',internal)]:
        write_json(out/(name+'.json'),obs)
    write_json(out/'input_hashes.json',hashes)
    write_json(out/'filter_report.json',{'samples':reports,'total_samples':len(planned),
        'valid_samples':len(original),'valid_train_samples':sum(o['split']=='train' for o in original),
        'valid_test_samples':sum(o['split']=='test' for o in original),
        'selection':'Image quality only; common membership for hybrid and same-burst r3kit; planned split unchanged.',
        'edge_fallback':'Use original r3kit corners if image-only edge fit is rejected; flag each fallback frame.'})
    return original,outer,internal


def mix(outer,internal):
    if [(o['sample_id'],o['split']) for o in outer]!=[(o['sample_id'],o['split']) for o in internal]:
        raise ValueError('Internal/outer membership mismatch')
    return [{**a,'points_M':a['points_M']+b['points_M'],'uv':a['uv']+b['uv'],
             'point_weights':[.5/len(a['uv'])]*len(a['uv'])+[.5/len(b['uv'])]*len(b['uv'])}
            for a,b in zip(outer,internal,strict=True)]


def validation(result,outer,internal,cfg):
    if set(result['train_ids']) & {o['sample_id'] for o in outer+internal}:
        raise ValueError('Training/test overlap')
    outer_metrics=evaluate(result,outer);inner_metrics=evaluate(result,internal)
    limit=cfg['solver']['test_rms_limit_px']
    passed=len(outer)>=3 and outer_metrics['rms_px']<=limit and result['status_solve']=='ok'
    return {'status_internal_validation':'passed' if passed else 'failed',
        'status_physical_accuracy':'unverified','rms_px':outer_metrics['rms_px'],
        'threshold_px':limit,'acceptance_metric':'common outer-edge 2D RMS, same threshold as before',
        'outer':outer_metrics,'internal':inner_metrics,'test_ids':[o['sample_id'] for o in outer],
        'test_count':len(outer),'scope':'Frozen model predicts held-out points using actual flange poses. No per-image pose fitting. Reused/inspected captures are diagnostic, not fresh independent acceptance.'}


def original_r3kit(run,out,base,outer,internal,spec):
    """Call pinned upstream image detection and run(); keep native and common metrics separate."""
    from .reference_r3kit.algos.calib.handeye import HandEyeCalibor
    def new():return HandEyeCalibor(marker_type='aruco',ext_calib_params={'dict_type':'6X6_1000','marker_length':80})
    train=[o for o in base if o['split']=='train']
    held_outer=[o for o in outer if o['split']=='test'];held_inner=[o for o in internal if o['split']=='test']
    N_M=np.diag([1.,-1.,-1.,1.]);N_M[:3,3]=[.04,.04,0]
    rows=[]
    def finish(cal,method,ids):
        raw=cal.run(intrinsics=np.array(spec['K'],float),opt_intrinsics=False,opt_distortion=False)
        if raw is None:raise ValueError('Upstream r3kit produced no solution')
        result={'method':method,'T_W_C':inverse(raw['g2c']),'T_E_M':inverse(raw['b2w'])@N_M,
                'camera':deepcopy(spec),'train_ids':ids,'upstream_original_result':raw,
                'upstream_error_meaning':['independent_training_image_pose_fit_px','mean_closure_translation_m','mean_closure_rotation_rad'],
                'status_physical_accuracy':'unverified'}
        result['T_C_W']=inverse(result['T_W_C'])
        result['test']={'outer':evaluate(result,held_outer),'internal':evaluate(result,held_inner)}
        rows.append(result);write_json(out/(method+'.json'),result)
    # Same-burst backend comparison: exactly the same pose IDs and measured
    # r3kit corner medians. The upstream run() itself is unchanged.
    cal=new();cal.ext_calibor.image_size=(spec['height'],spec['width'])
    for o in train:
        cal.ext_calibor.obj_points.append(cal.ext_calibor.objp_marker.copy())
        cal.ext_calibor.img_points.append(np.asarray(o['uv'],np.float32))
        cal.b2g.append(np.asarray(o['T_W_E']))
    finish(cal,'r3kit_same_burst',[o['sample_id'] for o in train])
    # Complete original image -> corners -> calibrateCamera -> Shah route.
    # First image fixed in advance. Only common training poses enter the solve;
    # any first-frame detection rejection is recorded, never substituted.
    cal=new();accepted=[];rejected=[]
    for o in train:
        folder=run/'samples'/o['sample_id'];raw=read_jsonl(folder/'frames.jsonl')[0]
        image=cv2.imread(str(folder/raw['image']))
        if cal.add_image_pose(image,np.asarray(raw['actual']['T_W_E']),vis=False):accepted.append(o['sample_id'])
        else:rejected.append(o['sample_id'])
    if len(accepted)<10:raise ValueError('Too few first-frame images for full r3kit comparison')
    finish(cal,'r3kit_full_first_frame',accepted)
    write_json(out/'r3kit_protocol.json',{'commit':'95ce4caf1cb2a457b65264ba57981dc54ac6daf4',
        'opt_intrinsics':False,'opt_distortion':False,'first_frame_rejected_ids':rejected,
        'source':'camera_calibration/reference_r3kit/SOURCE.json',
        'comparison':'Both evaluated on the same frozen outer/internal holdout points. First-frame and burst measurements differ; train IDs recorded per result.'})
    return rows


def calibrate_hybrid(run,multi=True):
    from .report import render_report
    run=Path(run).resolve();manifest=read_json(run/'manifest.json')
    if manifest['status']!='captured':raise ValueError('Calibration requires a completed capture run')
    cfg=deepcopy(manifest['config']);cfg['marker']['refinement']='R3KIT';spec=manifest['camera']
    stamp=datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    out=run/'calibration'/f'{stamp}_hybrid';out.mkdir(parents=True)
    try:
        if cfg['marker']['dictionary']!='DICT_6X6_1000' or cfg['marker']['id']!=0 or not np.isclose(cfg['marker']['side_m'],.08,rtol=0,atol=1e-12):
            raise ValueError('Hybrid junction geometry requires DICT_6X6_1000 ID 0, 80 mm')
        if np.any(np.asarray(spec['coeffs'])!=0):raise ValueError('Hybrid radial fit requires the recorded zero-distortion camera baseline')
        base,outer,internal=extract(run,out,manifest,cfg);mixed=mix(outer,internal)
        train=lambda obs:[o for o in obs if o['split']=='train']
        test=lambda obs:[o for o in obs if o['split']=='test']
        if len(train(base))<cfg['solver']['min_train_poses'] or len(test(base))<3:
            raise ValueError(f'Insufficient valid poses after filtering: train={len(train(base))}, test={len(test(base))}; inspect {out / "filter_report.json"}')
        seed=solve(train(base),spec,cfg,multi=multi)
        result=fit(train(mixed),seed,spec)
        # A successful local optimization cannot override an unobservable or
        # ambiguous initialization diagnosed by the multi-seed geometry solver.
        if seed['status_solve']!='ok':result['status_solve']=seed['status_solve']
        result.update(schema_version=2,method='hybrid',T_C_W=inverse(np.asarray(result['T_W_C'])),
            frames={'W':'Flexiv world','C':'native RGB optical','E':'flange','M':'marker center'},
            units={'translation':'m','rotation':'matrix'},marker=cfg['marker'],config=cfg,
            camera_factory=spec,camera_parameter_origin={'K':'factory, fixed','coeffs':'jointly fitted forward Brown k1/k2; p1=p2=k3=0'},
            capture_run=str(run),observations_directory=str(out),robot_serial=manifest['robot']['serial'],
            initialization_status=seed['status_solve'],loss='1px Huber; equal total weight per pose; internal/outer 50/50')
        if cfg['robot'].get('T_base_world') is not None:
            result['T_B_C']=np.asarray(cfg['robot']['T_base_world'])@np.asarray(result['T_W_C'])
            result['T_C_B']=inverse(result['T_B_C'])
        held=validation(result,test(outer),test(internal),cfg)
        result['status_internal_validation']=held['status_internal_validation']
        write_json(out/'result.json',result);write_json(out/'validation.json',held)
        write_json(out/'initialization.json',seed)
        protocol={'method':'hybrid','source_manifest_sha256':digest(run/'manifest.json'),
            'train_ids':result['train_ids'],'test_ids':held['test_ids'],
            'effective_config':cfg,'capture_config_preserved':True,'split_policy':'unchanged plan membership',
            'minimum_frames_per_internal_point':max(6,cfg['capture']['min_valid_frames']),
            'minimum_internal_points':12,'minimum_points_per_pose_total':16,
            'internal_measurements':'Actual subpixel junctions; homography only used for image quality gating.',
            'camera_model':'Fixed factory K; shared forward Brown k1/k2 fitted with X/Y; factory model saved separately.',
            'comparison_note':'Original first-frame and same-burst r3kit retained. The already inspected dataset does not provide fresh independent acceptance.'}
        write_json(out/'protocol.json',protocol)
        # Fixed five contiguous training groups; no test observations used in
        # initialization, fitting or fold construction.
        folds=[];ids=result['train_ids']
        for indices in np.array_split(np.arange(len(ids)),5):
            held_ids=[ids[i] for i in indices];rest=[o for o in train(base) if o['sample_id'] not in held_ids]
            if len(rest)<cfg['solver']['min_train_poses']:
                folds.append({'held_ids':held_ids,'status':'insufficient_training_poses'});continue
            fold_seed=solve(rest,spec,cfg,multi=multi)
            fitted=fit([o for o in train(mixed) if o['sample_id'] not in held_ids],fold_seed,spec)
            if fold_seed['status_solve']!='ok':fitted['status_solve']=fold_seed['status_solve']
            metrics={name:evaluate(fitted,[o for o in obs if o['sample_id'] in held_ids])
                     for name,obs in [('outer',outer),('internal',internal)]}
            folds.append({'held_ids':held_ids,'status':fitted['status_solve'],'result':fitted,'metrics':metrics})
            print(f'Hybrid grouped validation: {len(folds)}/5',flush=True)
        cv={'folds':folds,'all_folds_ok':all(f['status']=='ok' for f in folds)}
        if all('metrics' in f for f in folds):
            cv['rms_px']={name:float(np.sqrt(np.mean([e*e for f in folds for e in f['metrics'][name]['per_pose_rms_px'].values()]))) for name in ['outer','internal']}
        write_json(out/'cross_validation.json',cv)
        try:
            references=original_r3kit(run,out,base,outer,internal,spec)
            comparison={'status':'completed','hybrid_test':{'outer':held['outer'],'internal':held['internal']},'r3kit':references}
        except Exception as exc:
            comparison={'status':'failed','error':str(exc),'hybrid_result_preserved':True}
        write_json(out/'comparison.json',comparison)
        result['comparison_status']=comparison['status']
        result['cross_validation_all_folds_ok']=cv['all_folds_ok']
        write_json(out/'result.json',result)
        render_report(out,result,{**held,'cross_validation':cv.get('rms_px'),'comparison_status':comparison['status']})
    except BaseException as exc:
        write_json(out/'failure.json',{'status_solve':'failed','error':str(exc)});raise
    return out,result,held
