"""Configuration and JSON-safe data helpers. Units are metres, radians and pixels."""
from pathlib import Path
import copy
import json
import math
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path) if path else ROOT / 'configs/d415_rizon4r.yaml'
    cfg = yaml.safe_load(path.read_text())
    cfg = copy.deepcopy(cfg)
    if not math.isclose(float(cfg['marker']['side_m']), 0.08, abs_tol=1e-9):
        raise ValueError('This installation uses the user-confirmed 0.08 m marker')
    for section, keys in {
        'camera': ['width', 'height', 'fps'],
        'record': ['robot_hz', 'max_state_age_s'],
        'motion': ['max_velocity_rad_s', 'max_acceleration_rad_s2', 'max_record_gap_s',
                   'max_record_step_rad', 'path_simplification_rad', 'max_waypoint_step_rad',
                   'start_tolerance_rad', 'waypoint_tolerance_rad', 'tracking_tolerance_rad',
                   'waypoint_timeout_s', 'poll_hz'],
        'capture': ['settle_seconds', 'settle_timeout_s', 'burst_frames', 'min_valid_frames',
                    'corner_noise_floor_px'],
    }.items():
        for k in keys:
            v = float(cfg[section][k])
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f'{section}.{k} must be positive and finite')
    if cfg['capture']['min_valid_frames'] > cfg['capture']['burst_frames']:
        raise ValueError('min_valid_frames exceeds burst_frames')
    out = Path(cfg['outputs'])
    cfg['outputs'] = str(out if out.is_absolute() else ROOT / out)
    return cfg


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def dumps(value):
    return json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False)
