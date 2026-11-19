"""Offline selection only: no hardware SDK or robot connection."""
from _common import parser,run
from camera_calibration.models import load_config
from camera_calibration.dataset import read_json
from camera_calibration.trajectory import make_plan


def main():
    p=parser(__doc__); p.add_argument('--session',required=True); p.add_argument('--samples',type=int,default=40)
    a=p.parse_args(); cfg=load_config(a.config) if a.config else read_json(__import__('pathlib').Path(a.session)/'manifest.json')['config']
    path,plan,_=make_plan(a.session,a.samples,cfg)
    print(f'PLAN={path}\nSamples: {plan["sample_count"]}; taught-path waypoints: {len(plan["route_indices"])}')


if __name__=='__main__': run(main)
