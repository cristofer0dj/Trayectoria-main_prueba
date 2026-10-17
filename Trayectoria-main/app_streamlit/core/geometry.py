"""Cinemática y geometría: rapidez, aceleración, curvatura, torsión y Frenet-Serret."""
from dataclasses import dataclass
import numpy as np
from . import trajectory as tr
from .errors import DegenerateGeometryError

EPS = 1e-9


@dataclass
class Kinematics:
    r: np.ndarray; v: np.ndarray; a: np.ndarray
    speed: np.ndarray; accel: np.ndarray
    curvature: np.ndarray; torsion: np.ndarray
    Tn: np.ndarray; N: np.ndarray; B: np.ndarray


def kinematics(p: tr.Params, t) -> Kinematics:
    t = np.asarray(t, float)
    r, v, a, j = tr.position(p, t), tr.velocity(p, t), tr.acceleration(p, t), tr.jerk(p, t)
    speed = np.linalg.norm(v, axis=1)
    if np.any(speed < EPS):
        i = int(np.argmin(speed))
        raise DegenerateGeometryError(
            f"El vector velocidad es nulo en t≈{t[i]:.3f} (a, b y c producen un punto fijo): "
            "la dirección tangente y la curvatura no están definidas.")
    cross = np.cross(v, a)
    cn = np.linalg.norm(cross, axis=1)
    if np.any(cn < EPS):
        i = int(np.argmin(cn))
        raise DegenerateGeometryError(
            f"En t≈{t[i]:.3f} el producto r'×r'' es nulo (la trayectoria degenera en una recta): "
            "el triedro de Frenet-Serret no está definido. Cambie a, b o c.")
    Tn = v / speed[:, None]
    B = cross / cn[:, None]
    N = np.cross(B, Tn)
    curvature = cn / speed**3
    torsion = np.einsum("ij,ij->i", cross, j) / cn**2
    return Kinematics(r, v, a, speed, np.linalg.norm(a, axis=1), curvature, torsion, Tn, N, B)
