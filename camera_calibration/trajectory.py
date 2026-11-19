"""Choose geometrically diverse samples; preserve the taught path during replay."""
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.spatial.transform import Rotation
from .dataset import read_json, read_jsonl, write_json, digest


def validate_recording(states, cfg):
    if len(states)<3: raise ValueError('Trajectory has too few states')
    q=np.asarray([s['q'] for s in states],float)
    ns=np.array([s['host_ns'] for s in states],np.int64)
    if q.ndim!=2 or not np.isfinite(q).all(): raise ValueError('Invalid recorded joint trajectory')
    if np.any(np.diff(ns)<=0): raise ValueError('Robot timestamps are not increasing')
    gaps=np.diff(ns)*1e-9
    if gaps.max()>cfg['motion']['max_record_gap_s']:
        raise ValueError(f'Trajectory contains a {gaps.max():.3f}s gap; record again')
    if np.max(np.abs(np.diff(q,axis=0)))>cfg['motion']['max_record_step_rad']:
        raise ValueError('Trajectory joint jump exceeds recording limit')
    if any(s['state_age_s']>cfg['record']['max_state_age_s'] for s in states):
        raise ValueError('Trajectory contains stale robot states')
    if any(s['fault'] for s in states): raise ValueError('Trajectory contains a robot fault')
    return q


def choose_indices(states, candidates, count, cfg):
    """Farthest-point selection in translation + SO(3) geodesic distance."""
    candidates=np.unique(np.asarray(candidates,dtype=int))
    if count<3 or len(candidates)<count:
        raise ValueError(f'Requested {count} points; only {len(candidates)} trajectory candidates. Record a longer sweep.')
    poses=np.asarray([states[i]['T_W_E'] for i in candidates])
    rotations=Rotation.from_matrix(poses[:,:3,:3]).as_quat()
    xyz=poses[:,:3,3]
    sc=cfg['selection']
    def distances(k):
        dt=np.linalg.norm(xyz-xyz[k],axis=1)
        dr=2*np.arccos(np.clip(np.abs(rotations@rotations[k]),0,1))
        return (dt/sc['translation_scale_m'])**2+(dr/np.deg2rad(sc['rotation_scale_deg']))**2,dt,dr
    # Start with the observation farthest from the translation centroid.
    selected=[int(np.argmax(np.linalg.norm(xyz-xyz.mean(axis=0),axis=1)))]
    minimum,dt,dr=distances(selected[0])
    minimum[(dt<sc['min_translation_m']) & (dr<np.deg2rad(sc['min_rotation_deg']))]=-np.inf
    while len(selected)<count:
        minimum[selected]=-np.inf
        k=int(np.argmax(minimum))
        if not np.isfinite(minimum[k]):
            raise ValueError(f'Only {len(selected)} sufficiently distinct poses; lower --samples or record more rotation/translation')
        candidate_dist,dt,dr=distances(k)
        selected.append(k); minimum=np.minimum(minimum,candidate_dist)
        minimum[(dt<sc['min_translation_m']) & (dr<np.deg2rad(sc['min_rotation_deg']))]=-np.inf
    return sorted(int(candidates[k]) for k in selected)


def path_indices(q, required, cfg):
    """Joint-space RDP with recorded indices, mandatory samples, and bounded segment size."""
    q=np.asarray(q)
    keep=set([0,len(q)-1,*required])
    anchors=sorted(keep)
    pending=list(zip(anchors[:-1],anchors[1:]))
    eps=cfg['motion']['path_simplification_rad']
    max_step=cfg['motion']['max_waypoint_step_rad']
    while pending:
        a,b=pending.pop()
        if b-a<=1: continue
        segment=q[a:b+1]; delta=q[b]-q[a]
        denom=float(delta@delta)
        alpha=np.clip((segment-q[a])@delta/denom,0,1) if denom>1e-20 else np.zeros(len(segment))
        error=np.max(np.abs(segment-(q[a]+alpha[:,None]*delta)),axis=1)
        k=int(np.argmax(error))
        if error[k]>eps:
            mid=a+k
        elif np.max(np.abs(delta))>max_step:
            mid=(a+b)//2
        else: continue
        keep.add(mid); pending.extend([(a,mid),(mid,b)])
    indices=sorted(keep)
    if np.max(np.abs(np.diff(q[indices],axis=0)))>max_step:
        raise ValueError('Recorded step too large for conservative replay; record slower or at a higher rate')
    return indices


def make_plan(session, count, cfg):
    session=Path(session)
    meta=read_json(session/'manifest.json')
    if meta['status']!='trajectory_recorded': raise ValueError('Use a finalized trajectory session')
    if meta['robot_serial']!=cfg['robot']['serial'] or meta['camera_serial']!=cfg['camera']['serial']:
        raise ValueError('Session device identity differs from config')
    states=read_jsonl(session/'trajectory/states.jsonl')
    q=validate_recording(states,cfg)
    # Visibility is deliberately evaluated only after stationary acquisition.
    # Teaching frames (including missing/cropped markers) never veto a pose.
    candidates=range(len(states))
    chosen=choose_indices(states,candidates,count,cfg)
    route=path_indices(q,chosen,cfg)
    rng=np.random.default_rng(cfg['solver']['seed'])
    ntest=max(3,round(count*cfg['solver']['test_fraction']))
    if count-ntest<cfg['solver']['min_train_poses']:
        raise ValueError('Not enough samples for separate training/test groups; use at least 13 (recommended 30–50)')
    test=set(rng.choice(chosen,ntest,replace=False).tolist())
    plan={'schema_version':1,'session':str(session.resolve()),'robot_serial':meta['robot_serial'],
          'camera_serial':meta['camera_serial'],'trajectory_sha256':digest(session/'trajectory/states.jsonl'),
          'config':cfg,'sample_count':count,'route_indices':route,
          'selection_basis':'robot_pose_only','visibility_filter_stage':'after_capture',
          'samples':[{'id':f'{j:04d}','trajectory_index':i,'split':'test' if i in test else 'train',
                      'target_q':states[i]['q'],'target_T_W_E':states[i]['T_W_E']} for j,i in enumerate(chosen)],
          'recorded_state_count':len(states),'home_q':states[0]['q'],'end_q':states[-1]['q'],
          'route_note':'Joint path approximates recorded curve within configured tolerance; no endpoint shortcut.'}
    stamp=datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    path=session/'plans'/f"{stamp}_cam-{meta['camera_serial']}"/'plan.json'
    write_json(path,plan)
    return path,plan,states


def replay_direction(current_q, states, tolerance):
    q=np.asarray(current_q)
    distances=[np.max(np.abs(q-np.asarray(states[i]['q']))) for i in (0,-1)]
    if min(distances)>tolerance:
        raise ValueError('Robot is not at the recorded home/end. Manually return to one endpoint before replay.')
    return 1 if distances[0]<=distances[1] else -1
