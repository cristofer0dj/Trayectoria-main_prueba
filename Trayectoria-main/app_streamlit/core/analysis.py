"""Orquestador: calcula TODO el análisis de una configuración (a, b, c)."""
from dataclasses import dataclass
import numpy as np
from . import trajectory as tr, field, geometry, metrics, constraints
from .constraints import Limits, Violation


@dataclass
class Analysis:
    params: tr.Params
    limits: Limits
    t: np.ndarray
    k: geometry.Kinematics
    temp: np.ndarray
    grad: np.ndarray
    dTdt: np.ndarray
    dirder: np.ndarray        # D_T̂ T = ∇T·T̂  (por unidad de distancia)
    s: np.ndarray             # longitud de arco acumulada
    arc: float
    straight: float
    violations: list

    @property
    def r(self): return self.k.r
    @property
    def ok(self): return len(self.violations) == 0
    @property
    def ratio(self): return self.arc / self.straight if self.straight > 1e-12 else float("inf")

    def at(self, t0: float) -> int:
        return int(np.argmin(np.abs(self.t - t0)))


def analyze(params: tr.Params, limits: Limits, n: int = 1500) -> Analysis:
    params.validate(); limits.validate()
    t = tr.time_grid(params, n)
    k = geometry.kinematics(params, t)
    temp = field.T_of(k.r)
    grad = field.gradient(k.r)
    dTdt = field.dT_dt(k.r, k.v)
    dirder = field.directional_derivative(k.r, k.Tn)
    s = metrics.cumulative_arc_length(t, k.speed)
    series = {
        "speed": ("Rapidez máxima", "u/t", k.speed, limits.speed_max),
        "accel": ("Aceleración máxima", "u/t²", k.accel, limits.accel_max),
        "temp": ("Temperatura máxima", "°", temp, limits.temp_max),
        "curv": ("Curvatura máxima", "1/u", k.curvature, limits.curv_max),
    }
    viol = constraints.find_violations(t, k.r, series, limits, params.t_end)
    return Analysis(params, limits, t, k, temp, grad, dTdt, dirder, s,
                    float(s[-1]), metrics.straight_distance(k.r), viol)
