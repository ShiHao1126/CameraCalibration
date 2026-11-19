"""Filter a completed capture by marker visibility/quality, without hardware access."""
from _common import parser,run
from pathlib import Path
from camera_calibration.detection import filter_capture


def main():
    p=parser(__doc__); p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    if args.config: p.error('Filtering uses the configuration saved with the capture')
    report=filter_capture(args.run)
    print(f'FILTER={args.run/"filter_report.json"}\n'
          f'Valid={report["valid_samples"]}/{report["total_samples"]}; '
          f'train={report["valid_train_samples"]}, test={report["valid_test_samples"]}; '
          f'enough samples={report["enough_samples_for_calibration"]}')


if __name__=='__main__': run(main)
