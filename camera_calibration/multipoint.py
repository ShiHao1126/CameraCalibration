"""Shared robot/camera calibration using measured internal marker junctions.

The marker homography is used only upstream for image detection/quality checks.
Optimization uses actual subpixel observations and known 80 mm marker geometry;
there are no per-image pose variables or homography-generated measurements.
"""
import copy
import numpy as np
from scipy.optimize import least_squares
from .geometry import CameraModel,inverse,se3_exp


def grid_points(grid,side=.08):
    grid=np.asarray(grid,float)
    if grid.ndim!=2 or grid.shape[1]!=2 or not np.isfinite(grid).all():
        raise ValueError('Expected finite Nx2 marker-grid coordinates')
    if np.any(grid<0) or np.any(grid>8): raise ValueError('Marker grid must be in [0,8]')
    return np.c_[(grid[:,0]/8-.5)*side,(.5-grid[:,1]/8)*side,np.zeros(len(grid))]


def aggregate_internal(sample,frames,min_frames=6,min_points=12):
    """Median each identified junction across a burst; never infer outer corners."""
    found={}
    for frame_index,frame in enumerate(frames):
        fit=frame.get('internal_fit')
        if not fit or not fit['accepted']: continue
        seen=set()
        for grid,uv in zip(fit['grid_points'],fit['image_points'],strict=True):
            key=tuple(grid)
            if key in seen: raise ValueError('Duplicate grid identity within a frame')
            seen.add(key);found.setdefault(key,[]).append((frame_index,np.asarray(uv,float)))
    keys=sorted(k for k,rows in found.items() if len(rows)>=min_frames)
    if len(keys)<min_points:
        raise ValueError(f"Sample {sample['sample_id']} has only {len(keys)} supported internal points; comparison aborted without dropping this pose")
    pixels=[];stats=[]
    for key in keys:
        rows=found[key];uv=np.array([x[1] for x in rows]);pixels.append(np.median(uv,axis=0))
        stats.append({'grid':key,'frames':[x[0] for x in rows],
                      'covariance_px2':np.cov(uv,rowvar=False).tolist()})
    return {'sample_id':sample['sample_id'],'split':sample['split'],'T_W_E':sample['T_W_E'],
            'grid_points':keys,'points_M':grid_points(keys).tolist(),'uv':np.asarray(pixels).tolist(),
            'point_support':stats,'measurement':'actual_subpixel_junctions_median_over_burst'}


def pack(observations):
    if not observations: raise ValueError('Empty observations')
    counts=np.array([len(o['uv']) for o in observations])
    if np.any(counts<4): raise ValueError('Need at least four measured points per pose')
    G=np.array([o['T_W_E'] for o in observations],float)
    P=np.concatenate([np.asarray(o['points_M'],float) for o in observations])
    uv=np.concatenate([np.asarray(o['uv'],float) for o in observations])
    if G.shape!=(len(observations),4,4) or P.shape!=(sum(counts),3) or uv.shape!=(sum(counts),2):
        raise ValueError('Invalid multipoint observation shape')
    if not all(np.isfinite(x).all() for x in (G,P,uv)): raise ValueError('Non-finite multipoint observations')
    indices=np.repeat(np.arange(len(G)),counts)
    # Equal total contribution per pose. The normalization is applied to the
    # robust cost, not to pixels, preserving the same 1px Huber transition.
    per_point=[]
    for o,n in zip(observations,counts,strict=True):
        w=np.asarray(o.get('point_weights',np.ones(n)),float)
        if w.shape!=(n,) or not np.isfinite(w).all() or np.any(w<=0):
            raise ValueError('Point weights must be positive finite Nx1 values')
        per_point.extend(4*w/w.sum())
    weights=np.repeat(per_point,2)
    return G,P,uv,indices,counts,weights


def predict(result,packed):
    G,P,_,indices,_,_=packed
    C=inverse(np.asarray(result['T_W_C']))@G@np.asarray(result['T_E_M'])
    xyz=np.einsum('nij,nj->ni',C[indices,:3,:3],P)+C[indices,:3,3]
    return CameraModel(result['camera']).project(xyz)


def evaluate(result,observations):
    packed=pack(observations);res=predict(result,packed)-packed[2];counts=packed[4]
    pieces=np.split(res,np.cumsum(counts)[:-1])
    per=np.array([np.sqrt(np.mean(np.sum(r*r,axis=1))) for r in pieces])
    return {'rms_px':float(np.sqrt(np.mean(per**2))),
            'aggregation':'sqrt(mean over poses of mean squared 2D point distances))',
            'point_count':int(sum(counts)),'pose_count':len(observations),
            'per_pose_rms_px':dict(zip([o['sample_id'] for o in observations],per.tolist())),
            'residuals_px':dict(zip([o['sample_id'] for o in observations],[r.tolist() for r in pieces]))}


def weighted_huber(z,weights):
    z=np.asarray(z);large=z>1;root=np.sqrt(np.maximum(z,1e-300))
    rho=np.array([np.where(large,2*root-1,z),np.where(large,1/root,1.),
                  np.where(large,-.5/np.maximum(z,1e-100)**1.5,0.)])
    return rho*weights[None,:]


def fit(observations,seed,camera_spec,radial=True):
    packed=pack(observations);X0=np.asarray(seed['T_W_C']);Y0=np.asarray(seed['T_E_M'])
    if np.any(np.asarray(camera_spec['coeffs'])!=0):
        raise ValueError('This comparison assumes the recorded zero-distortion D415 baseline')
    def decode(z):
        camera=copy.deepcopy(camera_spec)
        if radial: camera.update(model='brown_conrady',coeffs=[.1*z[12],.1*z[13],0.,0.,0.])
        return {'T_W_C':X0@se3_exp(z[:6]*.02),'T_E_M':Y0@se3_exp(z[6:12]*.02),'camera':camera}
    def residual(z):return (predict(decode(z),packed)-packed[2]).ravel()
    bounds=np.r_[np.full(12,100.),np.full(2 if radial else 0,5.)]
    fitted=least_squares(residual,np.zeros(len(bounds)),bounds=(-bounds,bounds),
        loss=lambda z:weighted_huber(z,packed[5]),f_scale=1.,max_nfev=400,
        ftol=1e-10,xtol=1e-10,gtol=1e-9)
    result=decode(fitted.x)
    singular=np.linalg.svd(fitted.jac,compute_uv=False)
    rank=int(np.count_nonzero(singular>singular[0]*1e-6))
    result.update(converged=bool(fitted.success),message=fitted.message,nfev=fitted.nfev,
                  cost=float(fitted.cost),parameters_at_bounds=bool(np.any(abs(fitted.x)>bounds*.999)),
                  train_ids=[o['sample_id'] for o in observations],
                  free_camera_parameters=['k1','k2'] if radial else [],
                  train_metrics=evaluate(result,observations),status_physical_accuracy='unverified')
    result['information']={'rank':rank,'parameter_count':len(bounds),
                           'singular_values':singular.tolist(),'rank_rtol':1e-6,
                           'scope':'Local robust-weighted Jacobian; X/Y tangent scales .02, radial scales .1. Not a physical accuracy bound.'}
    result['status_solve']=('failed' if not result['converged'] or result['parameters_at_bounds']
                            else 'degenerate' if rank<len(bounds) else 'ok')
    return result
