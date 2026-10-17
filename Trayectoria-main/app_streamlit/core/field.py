"""Campo escalar T(x,y,z) = 20 + 4x - 3y + 2z + x² + y² + 2z² - xy."""
import numpy as np

HESSIAN = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [0.0, 0.0, 4.0]])  # constante


def T(x, y, z):
    return 20 + 4 * x - 3 * y + 2 * z + x**2 + y**2 + 2 * z**2 - x * y


def T_of(P):
    P = np.asarray(P, float)
    return T(P[..., 0], P[..., 1], P[..., 2])


def gradient(P):
    """∇T = (4+2x-y, -3+2y-x, 2+4z)"""
    P = np.asarray(P, float)
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    return np.stack([4 + 2 * x - y, -3 + 2 * y - x, 2 + 4 * z], axis=-1)


def dT_dt(P, v):
    """Regla de la cadena: dT/dt = ∇T · r'(t)"""
    return np.sum(gradient(P) * v, axis=-1)


def directional_derivative(P, u):
    """D_u T = ∇T · u  (u debe ser unitario)"""
    return np.sum(gradient(P) * np.asarray(u, float), axis=-1)
