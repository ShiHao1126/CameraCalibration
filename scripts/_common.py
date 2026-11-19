"""Script bootstrap; all reusable work lives in camera_calibration/."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def parser(description):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--config',type=Path,help='Defaults to configs/d415_rizon4r.yaml; sessions retain a snapshot')
    return p


def run(main):
    try: main()
    except KeyboardInterrupt:
        print('\nInterrupted; active replay has requested robot.Stop().',file=sys.stderr); raise SystemExit(130)
    except Exception as e:
        print(f'ERROR: {type(e).__name__}: {e}',file=sys.stderr); raise SystemExit(1)
