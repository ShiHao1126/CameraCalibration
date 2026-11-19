"""Offline image-edge refinement and blocked training CV of residual hypotheses."""
from pathlib import Path
import argparse
import copy
import sys
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from camera_calibration.dataset import read_json,read_jsonl,write_json
from camera_calibration.geometry import CameraModel,marker_points,inverse,se3_exp
from camera_calibration.solver import unpack
from camera_calibration.detection import aggregate


from camera_calibration.features import internal_corners,edge_corners


def measure_edges(source,processed,out,cfg,spec,kind='edge'):
    observations=[];counts={'accepted':0,'fallback':0}
    for old in read_json(processed/'observations.json'):
        sid=old['sample_id'];folder=source/'samples'/sid;raw=read_jsonl(folder/'frames.jsonl')
        detected=read_json(processed/'detections'/f'{sid}.json');frames=[];details=[]
        for f,d in zip(raw,detected,strict=True):
            detection=copy.deepcopy(d['detection']);diagnostic=None
            if detection:
                image=cv2.imread(str(folder/f['image']))
                points,diagnostic=(internal_corners if kind=='internal' else edge_corners)(image,detection['uv']);detection['uv']=points.tolist()
                counts['accepted' if diagnostic['accepted'] else 'fallback']+=1
            frames.append({**f,'detection':detection})
            details.append({'image':f['image'],'detection':detection,kind+'_fit':diagnostic})
        obs=aggregate(frames,spec,cfg,sid);obs['split']=old['split'];observations.append(obs)
        write_json(out/(kind+'_detections')/f'{sid}.json',details)
    write_json(out/(kind+'_observations.json'),observations);write_json(out/(kind+'_fit_counts.json'),counts)
    return observations


class Model:
    def __init__(self,result,mode,camera_keys=None):
        self.result=result;self.mode=mode;self.X=np.asarray(result['T_W_C']);self.Y=np.asarray(result['T_E_M'])
        self.camera_keys=['fx','fy','cx','cy','k1','k2'] if 'camera' in mode else []
        self.extra=6 if mode in ['camera_drift','mount_drift'] else (3 if 'gravity' in mode else 0)
        if mode=='camera_drift':self.camera_keys=[]
        if camera_keys is not None:self.camera_keys=list(camera_keys)
        self.n=12+len(self.camera_keys)+self.extra
        self.bounds=np.r_[np.full(12,100.),np.full(len(self.camera_keys),5.),np.full(self.extra,3.)]

    def predict(self,z,observations):
        G=np.array([o['T_W_E'] for o in observations]);X=self.X@se3_exp(z[:6]*.02);Y=self.Y@se3_exp(z[6:12]*.02)
        spec=copy.deepcopy(self.result['camera']);K=np.asarray(spec['K']);D=np.zeros(5)
        for key,v in zip(self.camera_keys,z[12:]):
            if key in ['fx','fy']:i=0 if key=='fx' else 1;K[i,i]*=np.exp(.05*v)
            elif key in ['cx','cy']:K[0 if key=='cx' else 1,2]+=20*v
            else:D[0 if key=='k1' else 1]=.1*v
        spec.update(K=K.tolist(),coeffs=D.tolist(),model='brown_conrady')
        if self.mode in ['camera_drift','mount_drift']:
            times=np.array([o['relative_time'] for o in observations]);drift=z[-6:]*.005
            perturb=np.array([se3_exp(drift*t) for t in times])
            if self.mode=='camera_drift':
                Cs=np.array([inverse(X@d) for d in perturb])@G@Y
            else:Cs=inverse(X)@G@(Y@perturb)
        elif 'gravity' in self.mode:
            gravity=np.einsum('nji,j->ni',G[:,:3,:3],np.array([0.,0.,-1.]))
            torque=np.cross(self.Y[:3,3],gravity)/np.linalg.norm(self.Y[:3,3])
            rotations=Rotation.from_rotvec(torque*z[-3:]*.01).as_matrix()
            perturb=np.tile(np.eye(4),(len(G),1,1));perturb[:,:3,:3]=rotations
            Cs=inverse(X)@G@perturb@Y
        else:Cs=inverse(X)@G@Y
        P=np.einsum('nij,kj->nki',Cs[:,:3,:3],marker_points())+Cs[:,None,:3,3]
        pixels=CameraModel(spec).project(P.reshape(-1,3)).reshape(-1,4,2)
        return pixels,{'T_W_C':X,'T_E_M':Y,'camera':spec,'parameters':z.tolist(),'mode':self.mode,
                       'extra_model_note':'Drift/compliance models are diagnostic hypotheses; fitted terms are not independently measured physical changes.'}

    def residual(self,z,observations):
        return self.predict(z,observations)[0]-np.asarray([o['uv'] for o in observations])

    def fit(self,observations):
        return least_squares(lambda z:self.residual(z,observations).ravel(),np.zeros(self.n),
             bounds=(-self.bounds,self.bounds),loss='huber',f_scale=1,max_nfev=300,ftol=1e-9,xtol=1e-9,gtol=1e-8)

    def evaluate(self,z,obs):
        residual=self.residual(z,obs);per=np.sqrt(np.mean(np.sum(residual**2,axis=-1),axis=1))
        center=residual.mean(axis=1);deformation=residual-center[:,None,:]
        return {'rms_px':float(np.sqrt(np.mean(per**2))),'per_pose_rms_px':dict(zip([o['sample_id'] for o in obs],per.tolist())),
                'center_rms_px':float(np.sqrt(np.mean(np.sum(center**2,axis=-1)))),
                'shape_rms_px':float(np.sqrt(np.mean(np.sum(deformation**2,axis=-1)))),
                'residuals_px':dict(zip([o['sample_id'] for o in obs],residual.tolist()))}


def investigate(source,processed,stage='initial'):
    cv2.setNumThreads(2)
    result=read_json(processed/'result.json');spec=result['camera'];cfg=result['config']
    out=source/'diagnostics/residual_investigation';out.mkdir(parents=True,exist_ok=True)
    old=read_json(processed/'observations.json')
    edge=read_json(out/'edge_observations.json') if (out/'edge_observations.json').is_file() else measure_edges(source,processed,out,cfg,spec)
    sets={'r3kit':old,'edge_lines':edge}
    if stage!='initial':
        internal=read_json(out/'internal_observations.json') if (out/'internal_observations.json').is_file() else measure_edges(source,processed,out,cfg,spec,'internal')
        sets['internal_grid']=internal
    if stage=='additional':
        background=copy.deepcopy(old)
        transforms={x['sample_id']:np.asarray(x['M_reference_to_image']) for x in read_json(out/'background_optical_flow.json')}
        for o in background:
            A=np.eye(3);A[:2]=transforms[o['sample_id']]
            o['uv']=(np.c_[o['uv'],np.ones(4)]@np.linalg.inv(A).T)[:,:2].tolist()
        sets['background_compensation']=background
        write_json(out/'background_corrected_observations.json',background)
    times={}
    for obs in old:
        raw=read_jsonl(source/'samples'/obs['sample_id']/'frames.jsonl')
        times[obs['sample_id']]=np.mean([f['actual']['robot_after']['host_ns'] for f in raw])*1e-9
    start=min(times.values());end=max(times.values())
    for observations in sets.values():
        for obs in observations:obs['relative_time']=(times[obs['sample_id']]-start)/(end-start)-.5
    rows=[]
    if stage=='initial':
        jobs=[('r3kit',['fixed','camera','mount_drift','camera_drift','gravity','camera_gravity']),
              ('edge_lines',['fixed','camera','gravity','camera_gravity'])]
        variants={};filename='experiments.json'
    elif stage=='additional':
        jobs=[('internal_grid',['fixed','camera']),('background_compensation',['fixed','camera'])]
        variants={};filename='additional_experiments.json'
    else:
        variants={'k1':['k1'],'k1k2':['k1','k2'],'focal':['fx','fy'],'focal_k1':['fx','fy','k1'],
                  'focal_radial':['fx','fy','k1','k2'],'intrinsics_k1':['fx','fy','cx','cy','k1'],
                  'intrinsics':['fx','fy','cx','cy']}
        jobs=[(corner,list(variants)) for corner in ['r3kit','internal_grid','edge_lines']]
        filename='camera_complexity_selection.json'
    for corner,modes in jobs:
        observations=sets[corner]
        train=[o for o in observations if o['split']=='train'];test=[o for o in observations if o['split']=='test']
        folds=np.array_split(np.arange(len(train)),5)
        for mode in modes:
            model=Model(result,'camera' if mode in variants else mode,variants.get(mode));fit=model.fit(train);fold_results=[]
            for held in folds:
                training=[o for i,o in enumerate(train) if i not in held];validation=[train[i] for i in held]
                f=model.fit(training);fold_results.append({'validation_ids':[o['sample_id'] for o in validation],
                     'success':bool(f.success),**model.evaluate(f.x,validation)})
            cv_rms=float(np.sqrt(np.mean([e*e for f in fold_results for e in f['per_pose_rms_px'].values()])))
            row={'corners':corner,'model':mode,'success':bool(fit.success),'parameters_at_bounds':bool(np.any(abs(fit.x)>model.bounds*.999)),
                 'train':model.evaluate(fit.x,train),'test':model.evaluate(fit.x,test),'blocked_training_cv_rms_px':cv_rms,
                 'folds':fold_results,'fit':model.predict(fit.x,train)[1]}
            rows.append(row);write_json(out/filename,rows)
            print(corner,mode,'train',row['train']['rms_px'],'test',row['test']['rms_px'],'CV',cv_rms,'bound',row['parameters_at_bounds'],flush=True)
    write_json(out/'protocol.json',{'same_sample_ids':True,'same_80mm_square':True,
          'model_selection':'Five contiguous blocks within original training poses; aggregate out-of-fold RMS. Original six test poses kept unchanged, already inspected in prior debugging.',
          'loss':'All hypotheses use same raw pixel Huber loss f_scale=1, without covariance weights.',
          'edge_method':'Local image gradients and robust straight-edge intersections only; no robot poses or reprojection residuals used to choose image corners.',
          'production_changed':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);p.add_argument('--processed',required=True,type=Path)
    p.add_argument('--stage',choices=['initial','additional','complexity'],default='initial')
    a=p.parse_args();investigate(a.run.resolve(),a.processed.resolve(),a.stage)
