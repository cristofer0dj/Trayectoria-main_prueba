"""Condiciones de operación y detección de incumplimientos."""
from dataclasses import dataclass, field
import numpy as np
from .errors import InvalidParameterError


@dataclass(frozen=True)
class Limits:
    t_max: float = 4 * np.pi
    speed_max: float = 3.00
    accel_max: float = 2.50
    temp_max: float = 50.0
    curv_max: float = 0.50

    def validate(self) -> "Limits":
        names = {"t_max": "Tiempo máximo", "speed_max": "Rapidez máxima",
                 "accel_max": "Aceleración máxima", "temp_max": "Temperatura máxima",
                 "curv_max": "Curvatura máxima"}
        for k, label in names.items():
            v = getattr(self, k)
            try:
                ok = np.isfinite(float(v)) and float(v) > 0
            except (TypeError, ValueError):
                ok = False
            if not ok:
                raise InvalidParameterError(f"{label}: el límite debe ser un número positivo.")
        return self


@dataclass
class Violation:
    key: str
    condition: str
    unit: str
    t_first: float            # primer instante en que se incumple
    pos_first: np.ndarray
    t_peak: float             # instante del peor valor
    pos_peak: np.ndarray
    value: float              # valor registrado en el peor instante
    limit: float
    n_intervals: int = 1
    mask: np.ndarray = field(default=None, repr=False)

    @property
    def excess(self) -> float:
        return self.value - self.limit

    @property
    def excess_pct(self) -> float:
        return 100.0 * self.excess / self.limit


def _count_intervals(mask):
    m = mask.astype(int)
    return int(m[0] + np.sum(np.diff(m) == 1))


def find_violations(t, r, series: dict, limits: Limits, t_end: float):
    """series: key -> (nombre, unidad, arreglo, límite). Devuelve lista de Violation."""
    out = []
    if t_end > limits.t_max + 1e-9:
        end = r[-1]
        i0 = int(np.searchsorted(t, limits.t_max))
        i0 = min(i0, len(t) - 1)
        mask = t > limits.t_max
        out.append(Violation("time", "Tiempo máximo de recorrido", "", float(limits.t_max), r[i0],
                             float(t_end), end, float(t_end), float(limits.t_max), 1, mask))
    for key, (name, unit, arr, lim) in series.items():
        mask = arr > lim
        if not np.any(mask):
            continue
        i_first = int(np.argmax(mask))
        i_peak = int(np.argmax(arr))
        out.append(Violation(key, name, unit, float(t[i_first]), r[i_first], float(t[i_peak]),
                             r[i_peak], float(arr[i_peak]), float(lim),
                             _count_intervals(mask), mask))
    return out
