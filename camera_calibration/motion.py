"""Bounded C2 interpolation of a taught joint path between acquisition stops."""
import time
import numpy as np
from scipy.interpolate import BPoly,PchipInterpolator


def replay_limits(cfg):
    """Time compression by s multiplies velocity/acceleration/jerk by s/s²/s³."""
    m=cfg['motion']; speed=float(m.get('replay_speed_scale',1.0))
    if not np.isfinite(speed) or speed<=0: raise ValueError('replay_speed_scale must be positive and finite')
    return {'replay_speed_scale':speed,
            'max_velocity_rad_s':m['max_velocity_rad_s']*speed,
            'max_acceleration_rad_s2':m['max_acceleration_rad_s2']*speed**2,
            'max_jerk_rad_s3':m.get('max_jerk_rad_s3',1.0)*speed**3}


def smooth_path(points,cfg):
    speed=replay_limits(cfg)['replay_speed_scale']
    q=np.asarray(points,float)
    if q.ndim!=2 or len(q)<2 or not np.isfinite(q).all(): raise ValueError('Invalid motion path')
    q=q[np.r_[True,np.max(np.abs(np.diff(q,axis=0)),axis=1)>1e-10]]
    if len(q)<2: return None
    m=cfg['motion']; vmax=m['max_velocity_rad_s']; amax=m['max_acceleration_rad_s2']
    times=np.r_[0,np.cumsum(np.maximum(np.max(np.abs(np.diff(q,axis=0)),axis=1)/vmax,.04))]
    v=PchipInterpolator(times,q).derivative()(times); v[0]=0; v[-1]=0
    def curve(t,velocity):
        return BPoly.from_derivatives(t,[[p,w,np.zeros(q.shape[1])] for p,w in zip(q,velocity)])
    # The Bezier control hull lies within a convex tube around each recorded
    # segment. Bounding all controls therefore bounds the complete curve.
    tolerance=m.get('smooth_path_tolerance_rad',0.001)
    for _ in range(32):
        spline=curve(times,v); worst=0.
        for i in range(len(q)-1):
            d=q[i+1]-q[i]; controls=spline.c[:,i,:]
            alpha=np.clip((controls-q[i])@d/(d@d),0,1)
            worst=max(worst,float(np.max(np.abs(controls-(q[i]+alpha[:,None]*d)))))
        if worst<=tolerance: break
        v*=.5
    else: raise ValueError('Cannot bound smoothed path deviation')
    bounds=[float(np.max(np.abs(spline.derivative(n).c))) for n in (1,2,3)]
    scale=max(1,bounds[0]/vmax,np.sqrt(bounds[1]/amax),
              np.cbrt(bounds[2]/m.get('max_jerk_rad_s3',1.0)))*(1+1e-6)
    return curve(times*scale/speed,v/scale*speed)


def stream_leg(robot,recorder,points,cfg,commands,indices):
    limits=replay_limits(cfg)
    spline=smooth_path(points,cfg)
    if spline is None:
        robot.send_joint(points[-1],limits['max_velocity_rad_s'],limits['max_acceleration_rad_s2'])
        return
    m=cfg['motion']; duration=float(spline.x[-1]); period=1/m['poll_hz']
    velocity_curve=spline.derivative()
    started=time.monotonic(); last=started
    while True:
        now=time.monotonic()
        if now-last>max(.15,5*period): raise RuntimeError('Motion command loop delayed; replay stopped')
        s=recorder.latest()
        if s['fault'] or not s['operational']: raise RuntimeError('Robot is no longer operational')
        current_t=min(now-started,duration)
        expected=spline(current_t)
        if np.max(np.abs(np.asarray(s['q'])-expected))>m['tracking_tolerance_rad']:
            raise RuntimeError('Robot tracking error exceeded the taught trajectory limit')
        target_t=min(current_t+period,duration)
        q=spline(target_t); dq=velocity_curve(target_t)
        if target_t>=duration: dq=np.zeros_like(dq)
        commands.append({'host_ns':time.monotonic_ns(),'phase':'continuous_path',
                         'trajectory_interval':[int(indices[0]),int(indices[-1])],
                         'target_q':q,'target_dq':dq})
        robot.send_joint(q,limits['max_velocity_rad_s'],limits['max_acceleration_rad_s2'],target_velocity=dq)
        last=now
        if current_t>=duration: break
        time.sleep(max(0,period-(time.monotonic()-now)))
