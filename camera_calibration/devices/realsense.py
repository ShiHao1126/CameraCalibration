"""Native RGB acquisition; no implicit resize, alignment or distortion removal."""
import importlib.metadata
import time
import numpy as np


def enumerate_cameras():
    import pyrealsense2 as rs
    ctx = rs.context()
    return [{'serial':d.get_info(rs.camera_info.serial_number),
             'name':d.get_info(rs.camera_info.name),
             'usb':d.get_info(rs.camera_info.usb_type_descriptor)} for d in ctx.query_devices()]


class RealSense:
    def __init__(self, config):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config(); cfg.enable_device(str(config['serial']))
        cfg.enable_stream(rs.stream.color, int(config['width']), int(config['height']), rs.format.bgr8, int(config['fps']))
        self.last_frame = None
        try:
            self.profile = self.pipeline.start(cfg)
            dev = self.profile.get_device()
            if dev.get_info(rs.camera_info.name) != config['model']:
                raise RuntimeError('Camera model does not match configuration')
            intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.spec = {'serial':dev.get_info(rs.camera_info.serial_number), 'name':dev.get_info(rs.camera_info.name),
                         'firmware':dev.get_info(rs.camera_info.firmware_version),
                         'usb':dev.get_info(rs.camera_info.usb_type_descriptor),
                         'sdk_version':importlib.metadata.version('pyrealsense2'),
                         'width':intr.width,'height':intr.height,'fps':config['fps'],'format':'bgr8',
                         'K':[[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]],
                         'model':str(intr.model).split('.')[-1], 'coeffs':list(intr.coeffs)}
            for _ in range(config.get('warmup_frames',16)): self.read()
        except BaseException:
            self.close(); raise

    def read(self):
        before = time.monotonic_ns()
        frame = self.pipeline.wait_for_frames(5000).get_color_frame()
        after = time.monotonic_ns()
        if not frame: raise RuntimeError('Missing RGB frame')
        number = int(frame.get_frame_number())
        if self.last_frame is not None and number <= self.last_frame:
            raise RuntimeError('Repeated or reordered camera frame')
        self.last_frame = number
        meta = {'host_before_ns':before,'host_after_ns':after,'frame_number':number,
                'device_timestamp_ms':frame.get_timestamp(),'timestamp_domain':str(frame.get_frame_timestamp_domain())}
        for name in ('actual_exposure','gain_level','frame_timestamp','sensor_timestamp'):
            key = getattr(self.rs.frame_metadata_value, name, None)
            if key is not None and frame.supports_frame_metadata(key): meta[name] = frame.get_frame_metadata(key)
        return np.asanyarray(frame.get_data()).copy(), meta

    def close(self):
        if self.pipeline is not None:
            try: self.pipeline.stop()
            except RuntimeError: pass
            self.pipeline = None
