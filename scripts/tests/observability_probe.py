"""Reproduce a local observability check for ONE square marker.

This is a design experiment, not a hardware calibrator or accuracy benchmark.
Assumptions: ideal pinhole camera, exact robot poses, exact marker dimensions,
no measurement noise. Translation coordinates use metres and angles radians.
The 12 normalized perturbation coordinates each represent 0.001 rad or 0.001 m.
The perturbation is a local rotation-vector/translation chart; its derivative
at zero is the same tangent used by a right SE(3) exponential update.

Run: python observability_probe.py --output observability-results.json
Requires numpy and scipy. No camera/robot SDK, device access or network calls.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial.transform import Rotation


def transform(rotvec, translation):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    value[:3, 3] = translation
    return value


def inverse(value):
    result = np.eye(4)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ value[:3, 3]
    return result


def run():
    rng = np.random.default_rng(20261118)
    X = transform([0.5, -0.3, 0.15], [0.35, -0.4, 0.8])
    Y = transform([0.1, 0, 0.05], [0, 0.02, 0.12])
    front = Rotation.from_euler("x", 180, degrees=True).as_matrix()
    points = np.array([
        [-0.04, 0.04, 0], [0.04, 0.04, 0],
        [0.04, -0.04, 0], [-0.04, -0.04, 0],
    ])
    cases = []
    for mode in ("identical_pose", "translation_only", "single_axis_rotation", "diverse_rotation"):
        robot_poses = []
        for _ in range(40):
            C = np.eye(4)
            C[:3, 3] = ([0, 0, 0.8] if mode == "identical_pose" else
                         rng.uniform([-0.15, -0.12, 0.6], [0.15, 0.12, 1.0]))
            rotvec = np.zeros(3)
            if mode == "single_axis_rotation":
                rotvec[2] = rng.uniform(-0.7, 0.7)
            if mode == "diverse_rotation":
                rotvec = rng.uniform([-0.6, -0.6, -0.8], [0.6, 0.6, 0.8])
            C[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix() @ front
            robot_poses.append(X @ C @ inverse(Y))

        def prediction(delta):
            estimated_X = X @ transform(delta[:3] * 0.001, delta[3:6] * 0.001)
            estimated_Y = Y @ transform(delta[6:9] * 0.001, delta[9:] * 0.001)
            pixels = []
            for G in robot_poses:
                C = inverse(estimated_X) @ G @ estimated_Y
                camera_points = points @ C[:3, :3].T + C[:3, 3]
                if not np.all(camera_points[:, 2] > 0):
                    raise ValueError("Synthetic point is behind the camera")
                uv = 1400 * camera_points[:, :2] / camera_points[:, 2, None] + [960, 540]
                pixels.extend(uv.ravel())
            return np.array(pixels)

        step = 1e-4
        basis = np.eye(12) * step
        jacobian = np.column_stack([
            (prediction(direction) - prediction(-direction)) / (2 * step)
            for direction in basis
        ])
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        rank = int(np.count_nonzero(singular_values > singular_values[0] * 1e-7))
        cases.append({
            "case": mode, "pose_count": 40, "unknown_count": 12,
            "jacobian_rank": rank, "singular_values": singular_values.tolist(),
            "condition_number": (float(singular_values[0] / singular_values[-1])
                                 if rank == 12 else None),
        })

    expected = [6, 9, 11, 12]
    actual = [item["jacobian_rank"] for item in cases]
    if actual != expected:
        raise AssertionError(f"Reproduction mismatch: {actual}, expected {expected}")
    return {
        "experiment": "single_marker_local_observability",
        "seed": 20261118,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "assumptions": "synthetic; pinhole; exact robot poses; exact 80 mm square; no noise",
        "parameter_scales": {"rotation_rad": 0.001, "translation_m": 0.001},
        "relative_svd_threshold": 1e-7,
        "camera": {"width": 1920, "height": 1080, "fx": 1400, "fy": 1400, "cx": 960, "cy": 540},
        "limitation": "Full local rank does not prove global uniqueness, good conditioning or physical accuracy.",
        "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for case in result["cases"]:
        print(f"{case['case']}: rank {case['jacobian_rank']} / 12 from 40 poses")
