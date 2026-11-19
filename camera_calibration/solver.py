"""Shah baseline, branch-aware initialization and shared X/Y raw-pixel refinement."""
import itertools
import cv2
import numpy as np
from scipy.optimize import least_squares
from .geometry import inverse, marker_points, se3_exp, distance, CameraModel


def unpack(observations):
    G=np.array([o['T_W_E'] for o in observations],float)
    uv=np.array([o['uv'] for o in observations],float)
    cov=np.array([o['covariance_px2'] for o in observations],float)
    if G.shape[1:]!=(4,4) or uv.shape[1:]!=(4,2) or cov.shape[1:]!=(4,2,2):
        raise ValueError('Observation shape mismatch')
    if not np.isfinite(G).all() or not np.isfinite(uv).all() or not np.isfinite(cov).all():
        raise ValueError('Non-finite observation')
    whitening=np.linalg.inv(np.linalg.cholesky(cov))
    return G,uv,whitening


def raw_residual(X,Y,G,uv,camera,side):
    C=np.matmul(inverse(X),np.matmul(G,Y))
    points=np.einsum('nij,kj->nki',C[:,:3,:3],marker_points(side))+C[:,None,:3,3]
    return camera.project(points.reshape(-1,3)).reshape(-1,4,2)-uv


def shah(G,C,method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH):
    Rz,tz,Rw,tw=cv2.calibrateRobotWorldHandEye(C[:,:3,:3],C[:,:3,3],G[:,:3,:3],G[:,:3,3],method=method)
    # OpenCV solves C_i Z = W G_i. Here Z=Y^-1 and W=X^-1.
    Z=np.eye(4); Z[:3,:3]=Rz; Z[:3,3]=tz.ravel()
    W=np.eye(4); W[:3,:3]=Rw; W[:3,3]=tw.ravel()
    return inverse(W),inverse(Z)


def camera_pose_baseline(uv,camera,cfg):
    """The r3kit fixed-intrinsics calibrateCamera stage, with known D retained.

    For this D415 all five D values are zero, so this is the original camera-pose
    stage on the same detected corners, apart from the equivalent marker frame/units.
    """
    if camera.model=='modified_brown_conrady' and np.any(camera.coeffs!=0):
        raise ValueError('OpenCV camera-pose baseline cannot represent this modified Brown model')
    flags=(cv2.CALIB_USE_INTRINSIC_GUESS|cv2.CALIB_FIX_FOCAL_LENGTH|cv2.CALIB_FIX_PRINCIPAL_POINT|
           cv2.CALIB_FIX_K1|cv2.CALIB_FIX_K2|cv2.CALIB_FIX_K3|cv2.CALIB_FIX_K4|cv2.CALIB_FIX_K5|
           cv2.CALIB_FIX_K6|cv2.CALIB_FIX_TANGENT_DIST)
    obj=[marker_points(cfg['marker']['side_m']).astype(np.float32) for _ in uv]
    dist=camera.coeffs.copy() if camera.model=='brown_conrady' else np.zeros(5)
    _,_,_,rv,tv=cv2.calibrateCamera(obj,[u.astype(np.float32) for u in uv],
        (camera.spec['width'],camera.spec['height']),camera.K.copy(),dist,flags=flags)
    C=[]
    for r,t in zip(rv,tv):
        T=np.eye(4); T[:3,:3]=cv2.Rodrigues(r)[0]; T[:3,3]=t.ravel(); C.append(T)
    return np.array(C)


def score_seed(X,Y,G,uv,W,camera,side,delta):
    try:
        raw=raw_residual(X,Y,G,uv,camera,side)
        white=np.einsum('nkij,nkj->nki',W,raw)
        norm=np.linalg.norm(white,axis=-1)
        return float(np.mean(np.where(norm<=delta,norm**2,2*delta*norm-delta**2)))
    except (ValueError,np.linalg.LinAlgError): return float('inf')


def initializations(observations,camera,cfg,multi=True):
    G,uv,W=unpack(observations); n=len(G); sc=cfg['solver']; side=cfg['marker']['side_m']
    candidates=[[np.asarray(h['T_C_M']) for h in o['candidates']] for o in observations]
    if any(not c for c in candidates): raise ValueError('Observation has no pose candidate')
    seeds=[]
    def add(indices,branches,method):
        try:
            X,Y=shah(G[indices],np.array([candidates[i][b] for i,b in zip(indices,branches)]),method)
            cost=score_seed(X,Y,G,uv,W,camera,side,sc['huber_delta'])
            if np.isfinite(cost): seeds.append({'X':X,'Y':Y,'cost':cost,'indices':list(map(int,indices)),
                                                'branches':list(map(int,branches)),'method':int(method)})
        except (cv2.error,ValueError,np.linalg.LinAlgError): pass
    try:
        X,Y=shah(G,camera_pose_baseline(uv,camera,cfg))
        cost=score_seed(X,Y,G,uv,W,camera,side,sc['huber_delta'])
        if np.isfinite(cost):
            seeds.append({'X':X,'Y':Y,'cost':cost,'indices':list(range(n)),
                          'branches':None,'method':'fixed_KD_calibrateCamera_then_Shah'})
    except (cv2.error,ValueError,np.linalg.LinAlgError): pass
    if multi or not seeds:
        add(np.arange(n),[0]*n,cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH)
    if not multi:
        if not seeds: raise ValueError('Baseline initialization failed; poses may be degenerate')
        return seeds
    rng=np.random.default_rng(sc['seed'])
    k=min(sc['subset_size'],n)
    for _ in range(sc['subset_count']):
        indices=np.sort(rng.choice(n,k,replace=False))
        for branch in itertools.product(*[range(len(candidates[i])) for i in indices]):
            for method in (cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI):
                add(indices,branch,method)
    seeds.sort(key=lambda x:x['cost'])
    distinct=[]
    for s in seeds:
        if all(any(a>b for a,b in zip(distance(s['X'],t['X']),(0.0005,np.deg2rad(.05)))) for t in distinct):
            distinct.append(s)
        if len(distinct)>=sc['max_initializations']: break
    if not distinct: raise ValueError('No valid initialization; inspect pose coverage/corner ordering')
    return distinct


def refine(seed,G,uv,W,camera,cfg):
    X,Y=seed['X'].copy(),seed['Y'].copy()
    sc=cfg['solver']; side=cfg['marker']['side_m']; scale=np.r_[[.02]*3,[.02]*3,[.02]*3,[.02]*3]
    success=False; messages=[]
    for _ in range(sc['irls_iterations']):
        raw=raw_residual(X,Y,G,uv,camera,side)
        white=np.einsum('nkij,nkj->nki',W,raw)
        norm=np.linalg.norm(white,axis=-1)
        weights=np.sqrt(np.minimum(1,sc['huber_delta']/np.maximum(norm,1e-12)))
        X0,Y0=X.copy(),Y.copy()
        def residual(z):
            try:
                r=raw_residual(X0@se3_exp(z[:6]*scale[:6]),Y0@se3_exp(z[6:]*scale[6:]),G,uv,camera,side)
                return (np.einsum('nkij,nkj->nki',W,r)*weights[...,None]).ravel()
            except (ValueError,np.linalg.LinAlgError): return np.full(uv.size,1e8)
        sol=least_squares(residual,np.zeros(12),loss='linear',max_nfev=sc['max_nfev'],
                          ftol=1e-10,xtol=1e-10,gtol=1e-8)
        X=X0@se3_exp(sol.x[:6]*scale[:6]); Y=Y0@se3_exp(sol.x[6:]*scale[6:])
        success=bool(sol.success); messages.append(sol.message)
        if np.linalg.norm(sol.x)<1e-7: break
    return {'X':X,'Y':Y,'cost':score_seed(X,Y,G,uv,W,camera,side,sc['huber_delta']),
            'converged':success,'messages':messages,'seed':seed}


def information(X,Y,G,uv,W,camera,cfg):
    # Each column is a 1 mrad or 1 mm perturbation. No prior rows are added.
    def f(z):
        r=raw_residual(X@se3_exp(z[:6]*.001),Y@se3_exp(z[6:]*.001),G,uv,camera,cfg['marker']['side_m'])
        return np.einsum('nkij,nkj->nki',W,r).ravel()
    h=1e-4; eye=np.eye(12)*h
    J=np.column_stack([(f(v)-f(-v))/(2*h) for v in eye])
    _,s,V=np.linalg.svd(J,full_matrices=False)
    rank=int(np.count_nonzero(s>s[0]*cfg['solver']['rank_rtol']))
    H=J.T@J; marginal=H[:6,:6]-H[:6,6:]@np.linalg.pinv(H[6:,6:])@H[6:,:6]
    return {'rank':rank,'singular_values':s.tolist(),'condition_number':float(s[0]/s[-1]) if rank==12 else None,
            'weakest_direction_rx_ry_rz_tx_ty_tz_X_then_Y':V[-1].tolist(),
            'camera_marginal_information':marginal.tolist(),'parameter_scales':{'rad':.001,'m':.001},
            'interpretation':'Local data identifiability only; not absolute accuracy.'}


def solve(observations,camera_spec,cfg,method='joint',multi=True):
    if len(observations)<cfg['solver']['min_train_poses']: raise ValueError('Not enough valid training poses')
    camera=CameraModel(camera_spec); G,uv,W=unpack(observations)
    seeds=initializations(observations,camera,cfg,multi=multi if method=='joint' else False)
    if method=='baseline':
        fits=[{**seeds[0],'converged':True,'messages':['Shah pose-level baseline']}]
    elif method=='joint': fits=[refine(s,G,uv,W,camera,cfg) for s in seeds]
    else: raise ValueError('method must be baseline or joint')
    fits=[f for f in fits if np.isfinite(f['cost'])]
    if not fits: raise ValueError('All optimizations failed')
    fits.sort(key=lambda f:f['cost']); best=fits[0]
    info=information(best['X'],best['Y'],G,uv,W,camera,cfg)
    status='ok' if best['converged'] else 'failed'
    if info['rank']<12: status='degenerate'
    alternatives=[]
    for f in fits[1:]:
        dt,dr=distance(best['X'],f['X'])
        if f['cost']<=best['cost']*(1+cfg['solver']['ambiguous_relative_cost'])+1e-8 and (
            dt>cfg['solver']['ambiguous_translation_m'] or dr>np.deg2rad(cfg['solver']['ambiguous_rotation_deg'])):
            alternatives.append({'T_W_C':f['X'].tolist(),'T_E_M':f['Y'].tolist(),'cost':f['cost']})
    if alternatives and status=='ok': status='ambiguous'
    raw=raw_residual(best['X'],best['Y'],G,uv,camera,cfg['marker']['side_m'])
    result={'schema_version':1,'method':method,'status_solve':status,'status_physical_accuracy':'unverified',
            'T_W_C':best['X'].tolist(),'T_C_W':inverse(best['X']).tolist(),'T_E_M':best['Y'].tolist(),
            'frames':{'W':'Flexiv world','C':'native RGB optical','E':'flange','M':'marker center'},
            'units':{'translation':'m','rotation':'matrix'},'camera':camera_spec,'marker':cfg['marker'],
            'train_ids':[o['sample_id'] for o in observations], 'train_rms_px':float(np.sqrt(np.mean(np.sum(raw**2,axis=-1)))),
            'cost':best['cost'],'information':info,'alternatives':alternatives,
            'initializations':[{'cost':f['cost'],'converged':f['converged'],'T_W_C':f['X'].tolist(),'T_E_M':f['Y'].tolist()} for f in fits]}
    if cfg['robot'].get('T_base_world') is not None:
        T=np.asarray(cfg['robot']['T_base_world'])@best['X']
        result['T_B_C']=T.tolist(); result['T_C_B']=inverse(T).tolist()
    return result
