"""Manual recording, conservative taught-path replay, and stationary image bursts."""
from datetime import datetime
from pathlib import Path
import threading
import time
import cv2
import numpy as np
from .dataset import Journal, write_json, update_manifest, read_json, read_jsonl, digest
from .geometry import distance, mean_transform, CameraModel
from .trajectory import validate_recording, replay_direction, path_indices
from .motion import stream_leg, replay_limits


def save_image(path,image):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),image): raise IOError(f'Cannot save {path}')


class StateRecorder:
    def __init__(self, robot, path, cfg):
        self.robot=robot; self.cfg=cfg; self.log=Journal(path)
        self.states=[]; self.error=None; self.lock=threading.Lock(); self.stop_event=threading.Event()
        self.thread=threading.Thread(target=self._loop,daemon=True)

    def _loop(self):
        period=1/self.cfg['record']['robot_hz']
        due=time.monotonic()
        try:
            while not self.stop_event.is_set():
                s=self.robot.read()
                if s['state_age_s']>self.cfg['record']['max_state_age_s']: raise RuntimeError('Robot feedback stopped updating')
                if s['fault']: raise RuntimeError('Robot fault during logging')
                with self.lock:
                    s['index']=len(self.states); self.log.append(s); self.states.append(s)
                due+=period
                self.stop_event.wait(max(0,due-time.monotonic()))
                if due<time.monotonic()-period: due=time.monotonic()
        except BaseException as e: self.error=e; self.stop_event.set()

    def start(self):
        self.thread.start()
        deadline=time.monotonic()+3
        while not self.states:
            self.check()
            if time.monotonic()>deadline: raise RuntimeError('No robot feedback')
            time.sleep(0.01)
        return self

    def check(self):
        if self.error: raise RuntimeError('Robot recording failed') from self.error

    def latest(self):
        self.check()
        with self.lock: s=self.states[-1].copy()
        if (time.monotonic_ns()-s['host_ns'])*1e-9>self.cfg['record']['max_state_age_s']:
            raise RuntimeError('Stale host robot sample')
        return s

    def window(self, since_ns):
        self.check()
        with self.lock:
            return [s for s in self.states if s['host_ns']>=since_ns]

    def close(self):
        self.stop_event.set(); self.thread.join(timeout=5)
        if self.thread.is_alive(): raise RuntimeError('Robot logging thread did not stop')
        self.log.close()


def record_trajectory(session, robot, cfg, duration=None):
    """Record only robot feedback. Camera identity is a config label until capture."""
    session=Path(session); folder=session/'trajectory'
    write_json(folder/'robot.json',robot.metadata())
    recorder=StateRecorder(robot,folder/'states.jsonl',cfg).start()
    update_manifest(session,status='recording',recording_mode='robot_only')
    started=time.monotonic(); last_print=-np.inf; failure=None
    try:
        print('Recording robot trajectory only. Ctrl+C saves and stops.',flush=True)
        while duration is None or time.monotonic()-started<duration:
            state=recorder.latest()
            if time.monotonic()-last_print>=.5:
                print(f'\r{time.monotonic()-started:6.1f}s | robot states {state["index"]+1}',end='',flush=True)
                last_print=time.monotonic()
            time.sleep(.02)
    except KeyboardInterrupt:
        pass
    except BaseException as e:
        failure=e
    finally:
        recorder.close()
        print()
    if failure is None:
        try: recorder.check(); validate_recording(recorder.states,cfg)
        except Exception as e: failure=e
    if failure:
        update_manifest(session,status='recording_failed',error=str(failure))
        raise failure
    write_json(folder/'summary.json',{'states':len(recorder.states),'recording_mode':'robot_only',
               'duration_s':time.monotonic()-started,'home':recorder.states[0],'end':recorder.states[-1],
               'home_definition':'First recorded actual joint state; no factory home command is executed.'})
    update_manifest(session,status='trajectory_recorded',trajectory_sha256=digest(folder/'states.jsonl'))


def stationary(states,cfg):
    if len(states)<2: return False
    anchor=np.asarray(states[0]['T_W_E'])
    c=cfg['capture']
    # RDK dq is a noisy differentiated measurement. Judge observed motion over
    # the window instead of requiring every instantaneous dq to be below a cap.
    q=np.asarray([s['q'] for s in states],float)
    elapsed=(states[-1]['host_ns']-states[0]['host_ns'])*1e-9
    if elapsed<=0 or not np.isfinite(q).all(): return False
    excursion=float(np.ptp(q,axis=0).max())
    if excursion>c.get('max_joint_drift_rad',0.0005): return False
    if excursion/elapsed>c['max_joint_speed_rad_s']: return False
    for s in states:
        dt,dr=distance(anchor,np.asarray(s['T_W_E']))
        if dt>c['max_translation_drift_m'] or dr>np.deg2rad(c['max_rotation_drift_deg']): return False
    return True


def wait_stable(recorder,cfg):
    c=cfg['capture']; deadline=time.monotonic()+c['settle_timeout_s']; since=time.monotonic_ns()
    while time.monotonic()<deadline:
        window=recorder.window(since)
        if len(window)<2:
            time.sleep(0.02)
            continue
        if not stationary(window,cfg): since=time.monotonic_ns()
        elif (window[-1]['host_ns']-window[0]['host_ns'])*1e-9>=c['settle_seconds']: return window[-1]
        time.sleep(0.02)
    raise RuntimeError('Robot did not settle within timeout')


def capture_sample(folder,target,recorder,camera,cfg):
    """Save the entire stationary burst, even if the marker is absent or cropped."""
    folder=Path(folder); write_json(folder/'target.json',target)
    stable=wait_stable(recorder,cfg)
    # Flush camera backlog using new frames only after the stationary gate.
    for _ in range(cfg['capture']['fresh_frames']): camera.read()
    since=time.monotonic_ns(); frames=[]
    with Journal(folder/'frames.jsonl') as log:
        for i in range(cfg['capture']['burst_frames']):
            before=recorder.latest(); image,meta=camera.read(); after=recorder.latest()
            if not stationary([stable,*recorder.window(since)],cfg): raise RuntimeError('Robot moved during image burst')
            rel=f'images/{i:03d}.png'; save_image(folder/rel,image)
            actual={'q':after['q'],'T_W_E':mean_transform([before['T_W_E'],after['T_W_E']]).tolist(),
                    'robot_before':before,'robot_after':after}
            row={'image':rel,'camera':meta,'actual':actual}
            log.append(row); frames.append(row)
    actual_T=mean_transform([f['actual']['T_W_E'] for f in frames])
    dt,dr=distance(np.asarray(target['target_T_W_E']),actual_T)
    write_json(folder/'actual.json',{'T_W_E':actual_T,'q_mean':np.mean([f['actual']['q'] for f in frames],axis=0),
               'target_translation_error_m':dt,'target_rotation_error_rad':dr})
    return {'sample_id':target['id'],'frame_count':len(frames),'status':'raw_captured'}


def _wait_waypoint(robot,recorder,q,previous,cfg):
    m=cfg['motion']; start=time.monotonic(); settled=None
    q=np.asarray(q); previous=np.asarray(previous); d=q-previous; denom=d@d
    while time.monotonic()-start<m['waypoint_timeout_s']:
        s=recorder.latest()
        if not s['operational'] or s['fault']: raise RuntimeError('Robot is no longer operational')
        actual=np.asarray(s['q'])
        alpha=np.clip((actual-previous)@d/denom,0,1) if denom>1e-15 else 0
        if np.max(np.abs(actual-(previous+alpha*d)))>m['tracking_tolerance_rad']:
            raise RuntimeError('Robot deviated from the current taught path segment')
        if np.max(np.abs(actual-q))<=m['waypoint_tolerance_rad'] and np.max(np.abs(s['dq']))<=0.01:
            if settled is None: settled=time.monotonic()
            if time.monotonic()-settled>=0.10: return s
        else: settled=None
        time.sleep(1/m['poll_hz'])
    raise RuntimeError('Waypoint timeout')


def replay_and_collect(plan_path,robot,camera,cfg):
    limits=replay_limits(cfg)
    plan_path=Path(plan_path); plan=read_json(plan_path); session=Path(plan['session'])
    CameraModel(camera.spec)  # Reject an unsupported projection model before any motion.
    if plan['robot_serial']!=cfg['robot']['serial'] or plan['camera_serial']!=camera.spec['serial']:
        raise ValueError('Plan/device identity mismatch')
    if digest(session/'trajectory/states.jsonl')!=plan['trajectory_sha256']: raise ValueError('Trajectory changed after planning')
    if plan['config']!=cfg: raise ValueError('Configuration changed after planning; generate a new plan')
    states=read_jsonl(session/'trajectory/states.jsonl'); q_recorded=validate_recording(states,cfg)
    required=[s['trajectory_index'] for s in plan['samples']]
    if any(not isinstance(i,int) or i<0 or i>=len(states) for i in required):
        raise ValueError('Invalid sample trajectory index')
    if len(set(required))!=len(required): raise ValueError('Duplicate sample index')
    expected_route=path_indices(q_recorded,required,cfg)
    if plan['route_indices']!=expected_route: raise ValueError('Replay route differs from the verified taught path')
    for sample in plan['samples']:
        s=states[sample['trajectory_index']]
        if not np.array_equal(sample['target_q'],s['q']) or not np.array_equal(sample['target_T_W_E'],s['T_W_E']):
            raise ValueError('Sample target differs from the recorded state')
    info=robot.metadata()
    if 'q_min' in info and (np.any(q_recorded<np.asarray(info['q_min'])) or np.any(q_recorded>np.asarray(info['q_max']))):
        raise ValueError('Recorded joints exceed robot joint limits')
    # New teaching sessions have no camera connection or intrinsics. Legacy
    # sessions may include them; retain their compatibility check when present.
    camera_file=session/'trajectory/camera.json'
    if camera_file.is_file():
        recorded_spec=read_json(camera_file)
        for k in ('serial','width','height','model','K','coeffs'):
            if camera.spec[k]!=recorded_spec[k]: raise ValueError(f'Camera {k} changed since teaching')
    for k in ('width','height'):
        if camera.spec[k]!=cfg['camera'][k]: raise ValueError(f'Camera {k} differs from the planned profile')
    direction=replay_direction(robot.read()['q'],states,cfg['motion']['start_tolerance_rad'])
    route=plan['route_indices'][::direction]; targets={s['trajectory_index']:s for s in plan['samples']}
    stamp=datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    run=session/'captures'/f"{stamp}_cam-{camera.spec['serial']}"; run.mkdir(parents=True)
    print(f'CAPTURE={run}',flush=True)
    write_json(run/'manifest.json',{'schema_version':1,'config':cfg,'camera':camera.spec,'robot':robot.metadata(),
               'plan':plan,'direction':direction,'status':'starting','reference_frame':'world',
               'filter_status':'pending','motion_profile':'continuous_C2_between_sample_stops',
               'effective_limits':{**limits,'smooth_path_tolerance_rad':cfg['motion'].get('smooth_path_tolerance_rad',.001),
                                   'max_joint_drift_rad':cfg['capture'].get('max_joint_drift_rad',.0005)}})
    recorder=None; raw_samples=[]
    try:
        # Check endpoint before enabling. Never execute PLAN-Home or another unrecorded motion.
        robot.begin_motion(cfg['motion']['operational_timeout_s'])
        # A physical mode change or brake release must not invalidate the start point.
        if replay_direction(robot.read()['q'],states,cfg['motion']['start_tolerance_rad'])!=direction:
            raise RuntimeError('Robot endpoint changed while enabling; replay aborted')
        recorder=StateRecorder(robot,run/'replay/states.jsonl',cfg).start()
        previous=np.asarray(recorder.latest()['q'])
        with Journal(run/'replay/commands.jsonl') as commands:
            stops=[0]+[i for i in range(1,len(route)) if route[i] in targets or i==len(route)-1]
            previous_stop=0
            for stop_number,position in enumerate(stops):
                index=route[position]
                target_q=np.asarray(states[index]['q'])
                if stop_number==0:
                    commands.append({'host_ns':time.monotonic_ns(),'phase':'initial_endpoint','trajectory_index':index,
                                     'target_q':target_q,'target_dq':np.zeros_like(target_q),
                                     'target_T_W_E':states[index]['T_W_E']})
                    robot.send_joint(target_q,cfg['motion']['max_velocity_rad_s'],cfg['motion']['max_acceleration_rad_s2'])
                else:
                    leg_indices=route[previous_stop:position+1]
                    points=[states[i]['q'] for i in leg_indices]
                    stream_leg(robot,recorder,points,cfg,commands,leg_indices)
                    previous=np.asarray(points[-2])
                _wait_waypoint(robot,recorder,target_q,previous,cfg)
                previous=target_q
                previous_stop=position
                print(f'\rReplay stop {stop_number+1}/{len(stops)}; captured {len(raw_samples)}/{len(targets)}',end='',flush=True)
                if index in targets:
                    target=targets[index]; folder=run/'samples'/target['id']
                    raw_samples.append(capture_sample(folder,target,recorder,camera,cfg))
        write_json(run/'raw_samples.json',raw_samples)
        update_manifest(run,status='captured',captured_samples=len(raw_samples))
    except BaseException as e:
        write_json(run/'raw_samples.json',raw_samples)
        update_manifest(run,status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e),captured_samples=len(raw_samples))
        raise
    finally:
        try: robot.stop()
        finally:
            if recorder: recorder.close()
        print()
    return run
