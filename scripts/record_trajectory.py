"""Record a manual drag trajectory starting at the operator's home pose; Ctrl+C saves."""
from _common import parser,run
from camera_calibration.models import load_config
from camera_calibration.dataset import create_session
from camera_calibration.devices.flexiv import Flexiv
from camera_calibration.capture import record_trajectory


def main():
    p=parser(__doc__); p.add_argument('--duration',type=float,help='Optional recording duration in seconds; starts immediately')
    args=p.parse_args(); cfg=load_config(args.config)
    if args.duration is not None and args.duration<=0: p.error('--duration must be positive')
    robot=None
    try:
        robot=Flexiv(cfg['robot'])
        if args.duration is None:
            input('在 Elements 中切到手动拖拽模式，将机械臂放到起始 home 并停稳；按 Enter 开始记录。')
        out=create_session(cfg)
        print(f'SESSION={out}',flush=True)
        record_trajectory(out,robot,cfg,args.duration)
        print(f'轨迹已保存：{out}\n下一步：python scripts/auto_calibrate.py --session {out} --samples 40 --execute')
    finally:
        if robot: robot.close()


if __name__=='__main__': run(main)
