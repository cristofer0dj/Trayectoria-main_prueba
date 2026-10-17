"""Modelo de la trayectoria  r(t) = ((a+bt)cos t, (a+bt)sen t, c t).
Derivadas analíticas (sin aproximación numérica)."""
from dataclasses import dataclass
import numpy as np
from .errors import InvalidParameterError

DEFAULT_T_END = 4 * np.pi


@dataclass(frozen=True)
class Params:
    a: float = 2.0
    b: float = 0.5
    c: float = 0.5
    t_end: float = DEFAULT_T_END

    def validate(self) -> "Params":
        for name in ("a", "b", "c", "t_end"):
            v = getattr(self, name)
            try:
                ok = np.isfinite(float(v))
            except (TypeError, ValueError):
                ok = False
            if not ok:
                raise InvalidParameterError(f"El parámetro «{name}» debe ser un número finito.")
        if self.t_end <= 0:
            raise InvalidParameterError("La duración del recorrido t_end debe ser mayor que 0.")
        if max(abs(self.a), abs(self.b), abs(self.c)) > 1e3:
            raise InvalidParameterError("Los parámetros a, b y c deben estar entre -1000 y 1000.")
        return self


def time_grid(p: Params, n: int = 1500) -> np.ndarray:
    return np.linspace(0.0, p.t_end, n)


def position(p: Params, t):
    t = np.asarray(t, float)
    rho = p.a + p.b * t
    return np.stack([rho * np.cos(t), rho * np.sin(t), p.c * t], axis=-1)


def velocity(p: Params, t):
    """r'(t)"""
    t = np.asarray(t, float)
    rho = p.a + p.b * t
    return np.stack([p.b * np.cos(t) - rho * np.sin(t),
                     p.b * np.sin(t) + rho * np.cos(t),
                     np.full_like(t, p.c)], axis=-1)


def acceleration(p: Params, t):
    """r''(t)"""
    t = np.asarray(t, float)
    rho = p.a + p.b * t
    return np.stack([-2 * p.b * np.sin(t) - rho * np.cos(t),
                     2 * p.b * np.cos(t) - rho * np.sin(t),
                     np.zeros_like(t)], axis=-1)


def jerk(p: Params, t):
    """r'''(t) (se usa para la torsión)"""
    t = np.asarray(t, float)
    rho = p.a + p.b * t
    return np.stack([-3 * p.b * np.cos(t) + rho * np.sin(t),
                     -3 * p.b * np.sin(t) - rho * np.cos(t),
                     np.zeros_like(t)], axis=-1)
