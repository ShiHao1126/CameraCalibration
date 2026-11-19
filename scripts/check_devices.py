"""Read-only hardware check. Never enables the robot or issues a motion command."""
from _common import parser,run
import time
from camera_calibration.models import load_config,dumps
from camera_calibration.dataset import create_session,write_json,update_manifest
from camera_calibration.devices.flexiv import Flexiv
from camera_calibration.devices.realsense import RealSense,enumerate_cameras
from camera_calibration.detection import MarkerDetector
from camera_calibration.capture import save_image


def main():
    args=parser(__doc__).parse_args(); cfg=load_config(args.config)
    out=create_session(cfg,'device_check'); robot=camera=None
    try:
        cameras=enumerate_cameras(); robot=Flexiv(cfg['robot']); camera=RealSense(cfg['camera'])
        states=[]
        for _ in range(10): states.append(robot.read()); time.sleep(.05)
        image,frame=camera.read(); save_image(out/'rgb.png',image)
        detection,reason=MarkerDetector(cfg['marker']).detect(image)
        data={'cameras':cameras,'selected_camera':camera.spec,'robot':robot.metadata(),
              'robot_states':states,'frame':frame,'marker':detection,'marker_rejection':reason,
              'motion_commands_sent':False}
        write_json(out/'devices.json',data); update_manifest(out,status='checked')
        print(dumps({'output':out,'camera':camera.spec,'robot':robot.metadata(),
                     'mode':states[-1]['mode'],'operational':states[-1]['operational'],
                     'marker':'visible' if detection else reason,'motion_commands_sent':False}))
    except BaseException as e:
        update_manifest(out,status='failed',error=str(e)); raise
    finally:
        if camera: camera.close()
        if robot: robot.close()


if __name__=='__main__': run(main)
