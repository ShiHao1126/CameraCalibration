"""Verified RDK 1.9.x adapter. Connecting/reading does not enable or move the robot."""
import importlib.metadata
import time
import numpy as np
from ..geometry import pose_matrix


class Flexiv:
    def __init__(self, config):
        import flexivrdk
        self.sdk = flexivrdk
        self.config = config
        self.version = importlib.metadata.version('flexivrdk')
        if not self.version.startswith('1.9.'):
            raise RuntimeError(f'This motion adapter is verified for RDK 1.9.x, found {self.version}')
        self.robot = flexivrdk.Robot(config['serial'], config.get('network_interfaces', []))
        self.motion_owned = False
        self.info = self.robot.info()
        self.dof = int(self.info.DoF)
        self.last_stamp = None
        self.last_changed = time.monotonic()

    def metadata(self):
        names = ['serial_num','robot_sn','model_name','DoF','q_min','q_max','dq_max','software_ver']
        out = {'serial':self.config['serial'],'rdk_version':self.version,'reference_frame':'world'}
        for key in names:
            v = getattr(self.info, key, None)
            if v is not None:
                out[key] = list(v) if isinstance(v, (list, tuple)) else v
        return out

    def read(self):
        before = time.monotonic_ns()
        if not self.robot.connected(): raise RuntimeError('Robot disconnected')
        s = self.robot.states()
        after = time.monotonic_ns()
        q, dq, pose = np.asarray(s.q), np.asarray(s.dq), np.asarray(s.flange_pose)
        if q.shape != (self.dof,) or not np.isfinite(q).all() or not np.isfinite(dq).all():
            raise RuntimeError('Invalid robot state')
        stamp = tuple(s.timestamp)
        if stamp != self.last_stamp:
            self.last_stamp, self.last_changed = stamp, time.monotonic()
        return {'host_before_ns':before,'host_after_ns':after,'host_ns':(before+after)//2,
                'device_timestamp':stamp,'state_age_s':time.monotonic()-self.last_changed,
                'q':q.tolist(),'dq':dq.tolist(),'flange_pose':pose.tolist(),
                'tcp_pose':list(s.tcp_pose),'T_W_E':pose_matrix(pose).tolist(),
                'mode':str(self.robot.mode()),'fault':bool(self.robot.fault()),
                'operational':bool(self.robot.operational())}

    def begin_motion(self, timeout_s=30):
        if self.robot.fault(): raise RuntimeError('Robot has a fault; no automatic fault clearing')
        if self.robot.mode() != self.sdk.Mode.IDLE:
            raise RuntimeError('Robot must be IDLE before replay; another program may own motion')
        self.motion_owned = True
        self.robot.Enable()
        deadline = time.monotonic()+timeout_s
        while not self.robot.operational():
            if self.robot.fault() or time.monotonic() > deadline:
                raise RuntimeError('Robot not operational: select AUTO in Elements and check enable/E-stop status')
            time.sleep(0.1)
        self.robot.SwitchMode(self.sdk.Mode.NRT_JOINT_POSITION)

    def send_joint(self, q, velocity, acceleration, target_velocity=None):
        if not self.motion_owned: raise RuntimeError('Motion has not been explicitly started')
        if self.robot.fault() or not self.robot.operational(): raise RuntimeError('Robot lost operational state')
        dq=[0.0]*self.dof if target_velocity is None else list(target_velocity)
        if len(dq)!=self.dof or not np.isfinite(dq).all() or np.max(np.abs(dq))>velocity*1.00001:
            raise ValueError('Invalid target joint velocity')
        self.robot.SendJointPosition(list(q),dq,[velocity]*self.dof,[acceleration]*self.dof)

    def stop(self):
        if self.motion_owned:
            try: self.robot.Stop()
            finally: self.motion_owned = False

    def close(self):
        self.stop()
        self.robot = None
