"""Longitud de arco y distancia directa."""
import numpy as np


def cumulative_arc_length(t, speed):
    """s(t) = ∫0^t |r'(u)| du  (regla del trapecio sobre la rapidez analítica)."""
    dt = np.diff(t)
    return np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * dt)])


def straight_distance(r):
    return float(np.linalg.norm(r[-1] - r[0]))
