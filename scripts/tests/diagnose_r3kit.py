"""Offline upstream r3kit comparison and training-only camera-model experiments."""
from pathlib import Path
import argparse
import copy
import sys
import importlib.metadata
import cv2
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from camera_calibration.dataset import read_json,read_jsonl,write_json,digest
from camera_calibration.geometry import inverse,se3_exp,CameraModel,distance
from camera_calibration.detection import pnp_candidates
from camera_calibration.solver import solve,unpack,raw_residual


def metrics(X,Y,spec,observations):
    G,uv,_=unpack(observations)
    r=raw_residual(X,Y,G,uv,CameraModel(spec),.08)
    per=np.sqrt(np.mean(np.sum(r*r,axis=-1),axis=1))
    return {'rms_px':float(np.sqrt(np.mean(per**2))),
            'per_pose':{o['sample_id']:float(e) for o,e in zip(observations,per)},
            'residuals_px':{o['sample_id']:v.tolist() for o,v in zip(observations,r)}}


def compare(run,processed):
    manifest=read_json(run/'manifest.json'); cfg=read_json(processed/'result.json')['config']
    spec=manifest['camera']; current=read_json(processed/'result.json')
    observations=read_json(processed/'observations.json')
    train=[o for o in observations if o['split']=='train']; test=[o for o in observations if o['split']=='test']
    reference=run/'diagnostics/r3kit_reference'
    sys.path.insert(0,str(reference))
    from r3kit.algos.calib.handeye import HandEyeCalibor
    out=run/'diagnostics/r3kit_full_comparison'; out.mkdir(parents=True,exist_ok=True)
    write_json(out/'provenance.json',{'reference_commit':'95ce4caf1cb2a457b65264ba57981dc54ac6daf4',
               'reference_sources':{str(p.relative_to(reference)):digest(p) for p in reference.rglob('*.py')},
               'source_capture':str(run),'processed_corners':str(processed),
               'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','opencv-python']},
               'opt_intrinsics':False,'opt_distortion':False,'marker_mm':80,
               'pose_input':'actual flange T_W_E; never target pose',
               'test_note':'Original membership retained; these images have been inspected during debugging and are not a fresh independent acceptance set.'})
    def new(): return HandEyeCalibor(marker_type='aruco',ext_calib_params={'dict_type':'6X6_1000','marker_length':80})
    # Marker-center frame M -> upstream top-left frame N, both in metres.
    N_M=np.diag([1.,-1.,-1.,1.]); N_M[:3,3]=[.04,.04,0]
    comparisons=[]
    def summarize(name,result,training,testing,camera=spec,extra=None):
        X=np.asarray(result['T_W_C']); Y=np.asarray(result['T_E_M'])
        row={'method':name,'T_W_C':X,'T_E_M':Y,'camera':camera,
             'train_count':len(training),'test_count':len(testing),
             'train':metrics(X,Y,camera,training),'test':metrics(X,Y,camera,testing)}
        if extra: row.update(extra)
        comparisons.append(row)
        print(name,'train',row['train']['rms_px'],'test',row['test']['rms_px'],flush=True)
    def finish_upstream(name,cal,training,testing):
        original=cal.run(intrinsics=np.array(spec['K'],float),opt_intrinsics=False,opt_distortion=False)
        result={'T_W_C':inverse(original['g2c']),'T_E_M':inverse(original['b2w'])@N_M}
        summarize(name,result,training,testing,extra={'upstream_original_result':original,
                  'upstream_error_meaning':['independent_per_image_pose_fit_px','mean_handeye_closure_translation_m','mean_handeye_closure_rotation_rad']})
    # Full original image -> corner -> per-image pose -> Shah pipeline. One
    # preselected first image per planned pose; held-out images never enter run().
    cal=new(); first=[]; rejected=[]; images=[]
    common={o['sample_id'] for o in observations}
    for target in manifest['plan']['samples']:
        folder=run/'samples'/target['id']; raw=read_jsonl(folder/'frames.jsonl')[0]
        image=cv2.imread(str(folder/raw['image'])); one=new()
        accepted=one.add_image_pose(image,np.asarray(raw['actual']['T_W_E']),vis=False)
        item={'sample_id':target['id'],'image':str((folder/raw['image']).relative_to(run)),'accepted':bool(accepted)}
        if accepted:
            uv=one.ext_calibor.img_points[0]; item['uv']=uv.tolist()
            o={'sample_id':target['id'],'split':target['split'],'uv':uv.tolist(),
               'T_W_E':raw['actual']['T_W_E'],'covariance_px2':np.tile(np.eye(2)*.15**2,(4,1,1)).tolist(),
               'candidates':pnp_candidates(uv,CameraModel(spec))}
            first.append(o)
            if target['split']=='train':
                cal.ext_calibor.obj_points.extend(one.ext_calibor.obj_points)
                cal.ext_calibor.img_points.extend(one.ext_calibor.img_points)
                cal.ext_calibor.image_size=one.ext_calibor.image_size; cal.b2g.extend(one.b2g)
        else: rejected.append(target['id'])
        images.append(item)
    write_json(out/'upstream_first_frame_detections.json',images)
    a=[o for o in first if o['split']=='train']; b=[o for o in first if o['split']=='test']
    finish_upstream('r3kit_full_first_frame',cal,a,b)
    summarize('joint_same_first_frames',solve(a,spec,cfg),a,b)
    # Controlled backend-only comparison: identical median corners and actual
    # pose means as current pipeline, original unmodified run() implementation.
    cal=new(); ext=cal.ext_calibor; ext.image_size=(spec['height'],spec['width'])
    for o in train:
        ext.obj_points.append(ext.objp_marker.copy()); ext.img_points.append(np.asarray(o['uv'],np.float32))
        cal.b2g.append(np.asarray(o['T_W_E']))
    finish_upstream('r3kit_solver_same_burst_observations',cal,train,test)
    summarize('joint_current_burst',current,train,test)
    write_json(out/'comparison.json',comparisons)
    # All camera experiments use identical training-only robust raw-pixel loss,
    # identical initialization, unchanged marker dimensions and fixed splits.
    G,uv,_=unpack(train); X0=np.asarray(current['T_W_C']); Y0=np.asarray(current['T_E_M'])
    K0=np.asarray(spec['K']); experiments=[]
    variants={'fixed':[], 'focal':['fx','fy'], 'intrinsics':['fx','fy','cx','cy'],
              'radial':['k1','k2'], 'intrinsics_radial':['fx','fy','cx','cy','k1','k2'],
              'intrinsics_brown':['fx','fy','cx','cy','k1','k2','p1','p2']}
    for name,keys in variants.items():
        def decode(z):
            camera=copy.deepcopy(spec); K=K0.copy(); D=np.zeros(5)
            for key,value in zip(keys,z[12:]):
                if key in ['fx','fy']:
                    i=0 if key=='fx' else 1; K[i,i]*=np.exp(.05*value)
                elif key in ['cx','cy']: K[0 if key=='cx' else 1,2]+=20*value
                else: D[{'k1':0,'k2':1,'p1':2,'p2':3}[key]]=value*.05
            camera.update(K=K.tolist(),coeffs=D.tolist(),model='brown_conrady')
            return X0@se3_exp(z[:6]*.02),Y0@se3_exp(z[6:12]*.02),camera
        def fun(z):
            X,Y,camera=decode(z)
            return raw_residual(X,Y,G,uv,CameraModel(camera),.08).ravel()
        bounds=np.r_[np.full(12,100.),np.full(len(keys),2.)]
        fit=least_squares(fun,np.zeros(12+len(keys)),bounds=(-bounds,bounds),loss='huber',f_scale=1.,max_nfev=500,
                          ftol=1e-10,xtol=1e-10,gtol=1e-9)
        X,Y,camera=decode(fit.x)
        row={'model':name,'free_camera_parameters':keys,'T_W_C':X,'T_E_M':Y,'camera':camera,
             'success':bool(fit.success),'nfev':fit.nfev,'cost':fit.cost,'parameters_at_bounds':bool(np.any(abs(fit.x)>bounds*.999)),
             'train':metrics(X,Y,camera,train),'test':metrics(X,Y,camera,test),
             'extrinsic_change_m_rad':distance(X0,X)}
        experiments.append(row)
        print('camera',name,'train',row['train']['rms_px'],'test',row['test']['rms_px'],flush=True)
    write_json(out/'camera_model_experiments.json',experiments)
    print('OUTPUT='+str(out))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--processed',type=Path,required=True); a=p.parse_args()
    compare(a.run.resolve(),a.processed.resolve())
