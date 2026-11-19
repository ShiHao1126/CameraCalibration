"""T_A_B maps B to A. Quaternion order at the RDK boundary is wxyz."""
import numpy as np
from scipy.spatial.transform import Rotation


def check_transform(T):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError('Expected finite 4x4 transform')
    R = T[:3, :3]
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-7) or not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or np.linalg.det(R) < 0.999:
        raise ValueError('Invalid rigid transform')
    return T


def inverse(T):
    T = check_transform(T)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def pose_matrix(pose):
    p = np.asarray(pose, float)
    if p.shape != (7,) or not np.isfinite(p).all() or abs(np.linalg.norm(p[3:])-1) > 0.01:
        raise ValueError('Invalid RDK pose [x,y,z,qw,qx,qy,qz]')
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(p[[4, 5, 6, 3]]).as_matrix()
    T[:3, 3] = p[:3]
    return T


def matrix_pose(T):
    T = check_transform(T)
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    return np.r_[T[:3, 3], q[[3, 0, 1, 2]]]


def mean_transform(transforms):
    arr = np.asarray(transforms)
    out = np.eye(4)
    out[:3, :3] = Rotation.from_matrix(arr[:, :3, :3]).mean().as_matrix()
    out[:3, 3] = arr[:, :3, 3].mean(axis=0)
    return out


def distance(A, B):
    delta = inverse(A) @ B
    return float(np.linalg.norm(delta[:3, 3])), float(Rotation.from_matrix(delta[:3, :3]).magnitude())


def se3_exp(xi):
    w, v = np.asarray(xi)[:3], np.asarray(xi)[3:]
    a = np.linalg.norm(w)
    W = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if a < 1e-7:
        V = np.eye(3) + W/2 + W@W/6
    else:
        V = np.eye(3) + (1-np.cos(a))/a**2*W + (a-np.sin(a))/a**3*(W@W)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(w).as_matrix()
    T[:3, 3] = V @ v
    return T


def marker_points(side=0.08):
    h = side/2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


class CameraModel:
    """Raw pixel projection matching librealsense forward models, with numeric inverse."""
    def __init__(self, spec):
        self.spec = spec
        self.K = np.asarray(spec['K'], dtype=float)
        self.coeffs = np.asarray(spec['coeffs'], dtype=float)
        self.model = spec['model'].split('.')[-1].lower()
        if self.model == 'inverse_brown_conrady' and (self.coeffs.shape != (5,) or np.any(self.coeffs != 0)):
            raise ValueError('Nonzero inverse Brown-Conrady requires a separately validated forward inverse model')
        if self.model not in ('none', 'brown_conrady', 'modified_brown_conrady', 'inverse_brown_conrady'):
            raise ValueError(f'Unsupported distortion model: {self.model}')
        if self.K.shape != (3, 3) or self.coeffs.shape != (5,) or not np.isfinite(self.K).all() or not np.isfinite(self.coeffs).all():
            raise ValueError('Invalid camera intrinsics')
        if min(self.K[0, 0], self.K[1, 1]) <= 0:
            raise ValueError('Invalid focal length')

    def distort(self, xy):
        x, y = np.asarray(xy, float).T
        if self.model in ('none', 'inverse_brown_conrady'):
            return np.c_[x, y]
        k1, k2, p1, p2, k3 = self.coeffs
        r2 = x*x+y*y
        f = 1+k1*r2+k2*r2*r2+k3*r2*r2*r2
        if self.model == 'modified_brown_conrady':
            x, y = x*f, y*f
            return np.c_[x+2*p1*x*y+p2*(r2+2*x*x), y+2*p2*x*y+p1*(r2+2*y*y)]
        return np.c_[x*f+2*p1*x*y+p2*(r2+2*x*x), y*f+2*p2*x*y+p1*(r2+2*y*y)]

    def project(self, points):
        p = np.asarray(points, float)
        if not np.isfinite(p).all() or np.any(p[:, 2] <= 1e-6):
            raise ValueError('Points behind camera or non-finite')
        d = self.distort(p[:, :2]/p[:, 2, None])
        return d*np.array([self.K[0, 0], self.K[1, 1]])+self.K[:2, 2]

    def unproject(self, uv):
        target = (np.asarray(uv)-self.K[:2, 2])/np.array([self.K[0, 0], self.K[1, 1]])
        xy = target.copy()
        for _ in range(25):
            err = self.distort(xy)-target
            if np.max(np.abs(err)) < 1e-12:
                return xy
            eps = 1e-6
            dx = (self.distort(xy+[eps, 0])-self.distort(xy-[eps, 0]))/(2*eps)
            dy = (self.distort(xy+[0, eps])-self.distort(xy-[0, eps]))/(2*eps)
            J = np.stack([dx, dy], axis=-1)
            xy -= np.linalg.solve(J, err[..., None])[..., 0]
        raise ValueError('Camera inverse projection failed to converge')
