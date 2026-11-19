"""Recompute calibration from an existing capture, without hardware access."""
from _common import parser,run
from camera_calibration.report import calibrate_run


def main():
    p=parser(__doc__); p.add_argument('--run',required=True)
    p.add_argument('--method',choices=['hybrid','baseline','joint'],default='hybrid',help='Default: internal + outer points, with saved upstream r3kit comparison')
    p.add_argument('--single-initialization',action='store_true',help='For controlled comparison of the pixel objective')
    a=p.parse_args()
    if a.config: p.error('Offline calibration uses the immutable capture configuration')
    out,result,validation=calibrate_run(a.run,a.method,not a.single_initialization)
    print(f'RESULT={out}\nMethod={result["method"]}; solve={result["status_solve"]}; test RMS={validation.get("rms_px")}; holdout={validation["status_internal_validation"]}; physical={validation["status_physical_accuracy"]}; r3kit={result.get("comparison_status","not_run")}')


if __name__=='__main__': run(main)
