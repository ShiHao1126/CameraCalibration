"""Compare pose baseline / single-seed pixel refinement / multiseed refinement on one capture."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from datetime import datetime
from camera_calibration.report import calibrate_run
from camera_calibration.dataset import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True); a=p.parse_args()
    rows=[]
    for method,multi in [('baseline',False),('joint',False),('joint',True)]:
        out,result,validation=calibrate_run(a.run,method,multi)
        rows.append({'method':method,'multi_initialization':multi,'result':str(out),
                     'solve':result['status_solve'],'train_rms_px':result['train_rms_px'],
                     'holdout':validation})
    path=a.run/'comparisons'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')/'comparison.json'
    write_json(path,rows); print(path)


if __name__=='__main__': main()
