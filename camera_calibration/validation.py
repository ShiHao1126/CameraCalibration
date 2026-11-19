"""Frozen-model holdout predictions and optional independent physical references."""
import numpy as np
from .geometry import CameraModel
from .solver import unpack,raw_residual


def evaluate(result,observations,cfg,references=None):
    overlap=set(result['train_ids']) & {o['sample_id'] for o in observations}
    if overlap: raise ValueError('Training/test overlap')
    report={'status_internal_validation':'not_run','status_physical_accuracy':'unverified',
            'test_count':len(observations),'test_ids':[o['sample_id'] for o in observations]}
    if observations:
        G,uv,_=unpack(observations)
        r=raw_residual(np.asarray(result['T_W_C']),np.asarray(result['T_E_M']),G,uv,CameraModel(result['camera']),cfg['marker']['side_m'])
        errors=np.linalg.norm(r,axis=-1)
        rms=float(np.sqrt(np.mean(errors**2)))
        report.update({'rms_px':rms,'p95_px':float(np.percentile(errors,95)),
                       'max_px':float(errors.max()),'per_pose_rms_px':np.sqrt(np.mean(errors**2,axis=1)).tolist(),
                       'status_internal_validation':'passed' if len(observations)>=3 and rms<=cfg['solver']['test_rms_limit_px'] and result['status_solve']=='ok' else 'failed',
                       'threshold_px':cfg['solver']['test_rms_limit_px']})
    if references is not None:
        C=np.asarray(references['points_C_m'],float); W=np.asarray(references['points_W_m'],float)
        if C.shape!=W.shape or C.ndim!=2 or C.shape[1]!=3 or len(C)<3 or not np.isfinite(C).all() or not np.isfinite(W).all():
            raise ValueError('Invalid independent reference points')
        uncertainty=float(references['reference_uncertainty_mm'])
        if not np.isfinite(uncertainty) or uncertainty<0: raise ValueError('Invalid reference uncertainty')
        T=np.asarray(result['T_W_C']); delta=(C@T[:3,:3].T+T[:3,3]-W)*1000
        error=np.linalg.norm(delta,axis=1)
        report['physical']={'rms_mm':float(np.sqrt(np.mean(error**2))),'max_mm':float(error.max()),
                            'p95_mm':float(np.percentile(error,95)),'xyz_errors_mm':delta.tolist(),
                            'reference_uncertainty_mm':uncertainty,'source':references['source'],
                            'scope':'Provided independent reference points only; not the entire workspace.'}
        report['status_physical_accuracy']='passed' if error.max()+uncertainty<=1 and result['status_solve']=='ok' else 'failed'
    return report
