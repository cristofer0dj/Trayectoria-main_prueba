"""Análisis local: aproximación lineal (plano tangente) y cuadrática del campo T."""
import numpy as np
from . import field

DIRECTIONS = {"Tangente T̂ (avance)": "T", "Normal N̂ (giro)": "N", "Binormal B̂": "B",
              "Gradiente ∇T/|∇T|": "G"}


def linear_model(P0):
    """Devuelve (T0, ∇T0): L(P) = T0 + ∇T0·(P-P0)."""
    return float(field.T_of(P0)), field.gradient(P0)


def compare_along_direction(P0, u, hs):
    """Compara T real, lineal y cuadrática en P0 + h·u para cada h."""
    P0 = np.asarray(P0, float); u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    T0, g = linear_model(P0)
    rows = []
    for h in hs:
        P = P0 + h * u
        real = float(field.T_of(P))
        lin = T0 + float(g @ (h * u))
        quad = lin + 0.5 * float((h * u) @ field.HESSIAN @ (h * u))
        rows.append(dict(h=float(h), real=real, lineal=lin, cuadratica=quad,
                         error_lineal=abs(real - lin), error_rel_pct=100 * abs(real - lin) / abs(real),
                         error_cuadratica=abs(real - quad)))
    return rows


def compare_along_trajectory(p, t0, dts):
    """Predicción lineal de T(r(t0+Δt)) ≈ T(t0) + (dT/dt)·Δt frente al valor real."""
    from . import trajectory as tr
    t0a = np.array([t0])
    P0 = tr.position(p, t0a)[0]
    v0 = tr.velocity(p, t0a)[0]
    T0 = float(field.T_of(P0)); rate = float(field.dT_dt(P0, v0))
    rows = []
    for dt in dts:
        t1 = np.array([t0 + dt])
        real = float(field.T_of(tr.position(p, t1)[0]))
        lin = T0 + rate * dt
        rows.append(dict(dt=float(dt), real=real, lineal=lin, error=abs(real - lin)))
    return rows
