"""Select points, replay the taught route, capture stationary bursts, then calibrate."""
from _common import parser,run
from pathlib import Path
from camera_calibration.models import load_config
from camera_calibration.dataset import read_json
from camera_calibration.trajectory import make_plan
from camera_calibration.capture import replay_and_collect
from camera_calibration.report import calibrate_run
from camera_calibration.detection import filter_capture
from camera_calibration.motion import replay_limits


def main():
    p=parser(__doc__); source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--session',type=Path); source.add_argument('--plan',type=Path)
    p.add_argument('--samples',type=int,default=40,help='Number of poses to visit; valid pose count is determined after capture')
    p.add_argument('--execute',action='store_true',help='Enable automatic motion along the recorded route; without this flag only write the plan')
    p.add_argument('--method',choices=['hybrid','baseline','joint'],default='hybrid',help='Default: internal + outer points, with upstream r3kit comparison')
    p.add_argument('--capture-only',action='store_true',help='Capture and filter images, but skip calibration')
    a=p.parse_args()
    if a.plan:
        plan=read_json(a.plan); cfg=load_config(a.config) if a.config else plan['config']; plan_path=a.plan
    else:
        cfg=load_config(a.config) if a.config else read_json(a.session/'manifest.json')['config']
        plan_path,plan,_=make_plan(a.session,a.samples,cfg)
    print(f'PLAN={plan_path}\nSamples={plan["sample_count"]}, waypoints={len(plan["route_indices"])}',flush=True)
    limits=replay_limits(cfg)
    print(f'Replay speed: {limits["replay_speed_scale"]:g}x; '
          f'joint velocity limit {limits["max_velocity_rad_s"]:g} rad/s; '
          f'acceleration limit {limits["max_acceleration_rad_s2"]:g} rad/s²',flush=True)
    if not a.execute:
        print('Plan only. To replay, select AUTO in Elements and rerun with --execute.'); return
    from camera_calibration.devices.flexiv import Flexiv
    from camera_calibration.devices.realsense import RealSense
    robot=camera=None
    try:
        robot=Flexiv(cfg['robot']); camera=RealSense(cfg['camera'])
        captured=replay_and_collect(plan_path,robot,camera,cfg)
    finally:
        try:
            if robot: robot.close()
        finally:
            if camera: camera.close()
    print(f'CAPTURE={captured}',flush=True)
    if a.capture_only or a.method!='hybrid':
        filtered=filter_capture(captured)
        print(f'FILTER: {filtered["valid_samples"]}/{filtered["total_samples"]} valid poses; '
              f'train={filtered["valid_train_samples"]}, test={filtered["valid_test_samples"]}. '
              f'Report: {captured/"filter_report.json"}',flush=True)
    if not a.capture_only:
        out,result,validation=calibrate_run(captured,a.method)
        print(f'RESULT={out}\nSolve={result["status_solve"]}; holdout={validation["status_internal_validation"]}; physical={validation["status_physical_accuracy"]}')
        if result.get('comparison_status')=='failed':
            print(f'r3kit comparison failed; inspect {out/"comparison.json"}. Hybrid result has been saved.',flush=True)
        if result['status_solve']!='ok' or validation['status_internal_validation']!='passed':
            raise RuntimeError('Calibration result saved but validation did not pass; inspect the saved report')


if __name__=='__main__': run(main)
