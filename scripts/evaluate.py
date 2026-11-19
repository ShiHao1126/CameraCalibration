"""Evaluate a frozen result on its held-out poses and optional independent references."""
from _common import parser,run
from pathlib import Path
from datetime import datetime
from camera_calibration.dataset import read_json,write_json
from camera_calibration.validation import evaluate
from camera_calibration.report import render_report


def main():
    p=parser(__doc__); p.add_argument('--result',type=Path,required=True); p.add_argument('--references',type=Path)
    a=p.parse_args()
    if a.config: p.error('Evaluation uses the frozen result configuration')
    result=read_json(a.result); capture=Path(result['capture_run'])
    if result['method']=='hybrid':
        from camera_calibration.hybrid import validation as hybrid_validation
        measurements=Path(result.get('observations_directory',a.result.parent))
        outer=[o for o in read_json(measurements/'outer_observations.json') if o['split']=='test']
        inner=[o for o in read_json(measurements/'internal_observations.json') if o['split']=='test']
        validation=hybrid_validation(result,outer,inner,result['config'])
        if a.references:
            physical=evaluate(result,outer,result['config'],read_json(a.references))
            validation['physical']=physical['physical']
            validation['status_physical_accuracy']=physical['status_physical_accuracy']
        result['observations_directory']=str(measurements.resolve())
    else:
        obs=[o for o in read_json(capture/'observations.json') if o['split']=='test']
        validation=evaluate(result,obs,result['config'],read_json(a.references) if a.references else None)
    result['status_internal_validation']=validation['status_internal_validation']
    result['status_physical_accuracy']=validation['status_physical_accuracy']
    out=a.result.parent/('evaluation_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    write_json(out/'result.json',result); write_json(out/'validation.json',validation); render_report(out,result,validation)
    print(f'EVALUATION={out}')


if __name__=='__main__': run(main)
