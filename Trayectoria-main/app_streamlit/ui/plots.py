"""Presentación: figuras Plotly (sin lógica matemática propia, solo dibujo)."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core import field

RED = "#e63946"


def _arrow(fig, p, d, color, name, scale):
    q = p + d * scale
    fig.add_trace(go.Scatter3d(x=[p[0], q[0]], y=[p[1], q[1]], z=[p[2], q[2]], mode="lines+markers",
                               line=dict(color=color, width=7), marker=dict(size=[0, 5], color=color),
                               name=name, hoverinfo="name"))


def fig_3d(an, i, field_on=True, vectors=True, frenet=True, plane=True, viol=True):
    r = an.r; p0 = r[i]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=r[:, 0], y=r[:, 1], z=r[:, 2], mode="lines", name="Trayectoria",
        line=dict(color=an.temp, colorscale="Viridis", width=6,
                  colorbar=dict(title="T", len=0.6, thickness=12)),
        customdata=np.stack([an.t, an.temp], axis=-1),
        hovertemplate="t=%{customdata[0]:.2f}<br>T=%{customdata[1]:.1f}<extra></extra>"))
    if viol:
        for v in an.violations:
            if v.mask is not None and np.any(v.mask):
                m = v.mask
                fig.add_trace(go.Scatter3d(x=r[m, 0], y=r[m, 1], z=r[m, 2], mode="markers",
                                           marker=dict(size=3, color=RED), name=f"⚠ {v.condition}"))
    fig.add_trace(go.Scatter3d(x=[p0[0]], y=[p0[1]], z=[p0[2]], mode="markers",
                               marker=dict(size=8, color="orange", symbol="diamond",
                                           line=dict(color="black", width=1)),
                               name=f"Punto t={an.t[i]:.2f}"))
    ext = float(np.ptp(r, axis=0).max()) or 1.0
    sc = ext / 6
    if vectors:
        _arrow(fig, p0, an.k.v[i] / an.k.speed[i], "#1d3557", "Velocidad (dir.)", sc)
        if an.k.accel[i] > 1e-9:
            _arrow(fig, p0, an.k.a[i] / an.k.accel[i], "#f77f00", "Aceleración (dir.)", sc)
        g = an.grad[i]; gn = np.linalg.norm(g)
        if gn > 1e-9:
            _arrow(fig, p0, g / gn, "#7b2cbf", "Gradiente ∇T", sc)
    if frenet:
        _arrow(fig, p0, an.k.Tn[i], "#2a9d8f", "T̂ (tangente)", sc * 0.8)
        _arrow(fig, p0, an.k.N[i], "#e9c46a", "N̂ (normal)", sc * 0.8)
        _arrow(fig, p0, an.k.B[i], "#264653", "B̂ (binormal)", sc * 0.8)
    if plane:
        g = an.grad[i]; gn = np.linalg.norm(g)
        if gn > 1e-9:
            n = g / gn
            u1 = np.cross(n, [0, 0, 1.0]);
            if np.linalg.norm(u1) < 1e-6: u1 = np.cross(n, [0, 1.0, 0])
            u1 /= np.linalg.norm(u1); u2 = np.cross(n, u1)
            s = sc * 1.2
            C = np.array([p0 + s * (a * u1 + b * u2) for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
            fig.add_trace(go.Mesh3d(x=C[:, 0], y=C[:, 1], z=C[:, 2], i=[0, 0], j=[1, 2], k=[2, 3],
                                    color="#7b2cbf", opacity=0.3, name="Plano tangente a T=cte",
                                    showlegend=True))
    if field_on:
        lo, hi = r.min(0) - 1.0, r.max(0) + 1.0
        g = np.linspace(0, 1, 18)
        X, Y, Z = np.meshgrid(*[lo[k] + (hi[k] - lo[k]) * g for k in range(3)], indexing="ij")
        V = field.T(X, Y, Z)
        for lvl, col, nm in [(an.limits.temp_max, RED, f"T = {an.limits.temp_max:g} (límite)"),
                             (float(an.temp[i]), "#4895ef", f"T = {an.temp[i]:.1f} (punto)")]:
            if V.min() < lvl < V.max():
                fig.add_trace(go.Isosurface(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=V.ravel(),
                                            isomin=lvl, isomax=lvl, surface_count=1, opacity=0.18,
                                            colorscale=[[0, col], [1, col]], showscale=False,
                                            caps=dict(x_show=False, y_show=False, z_show=False),
                                            name=nm, showlegend=True))
    fig.update_layout(height=640, margin=dict(l=0, r=0, t=10, b=0),
                      scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
                      legend=dict(orientation="h", y=-0.02))
    return fig


def fig_dashboard(an, t_sel):
    L = an.limits
    items = [("Rapidez |r'|", an.k.speed, L.speed_max), ("Aceleración |r''|", an.k.accel, L.accel_max),
             ("Curvatura κ", an.k.curvature, L.curv_max), ("Temperatura T(r(t))", an.temp, L.temp_max)]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[x[0] for x in items])
    for n, (name, y, lim) in enumerate(items):
        rr, cc = n // 2 + 1, n % 2 + 1
        fig.add_trace(go.Scatter(x=an.t, y=y, mode="lines", name=name, showlegend=False), row=rr, col=cc)
        bad = y > lim
        if bad.any():
            fig.add_trace(go.Scatter(x=an.t[bad], y=y[bad], mode="markers", marker=dict(color=RED, size=4),
                                     showlegend=False, name="Incumple"), row=rr, col=cc)
        fig.add_hline(y=lim, line=dict(color=RED, dash="dash"), row=rr, col=cc)
        fig.add_vline(x=t_sel, line=dict(color="orange", width=2), row=rr, col=cc)
        fig.update_xaxes(title_text="t", row=rr, col=cc)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def fig_env(an, t_sel):
    fig = make_subplots(rows=1, cols=3, subplot_titles=["T(r(t))", "dT/dt = ∇T·r'", "D_T̂ T = ∇T·T̂"])
    for c, y in enumerate([an.temp, an.dTdt, an.dirder], 1):
        fig.add_trace(go.Scatter(x=an.t, y=y, mode="lines", showlegend=False), row=1, col=c)
        fig.add_hline(y=0 if c > 1 else an.limits.temp_max, line=dict(color="gray" if c > 1 else RED, dash="dash"), row=1, col=c)
        fig.add_vline(x=t_sel, line=dict(color="orange"), row=1, col=c)
        fig.update_xaxes(title_text="t", row=1, col=c)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def fig_orientation(an, t_sel):
    ang = np.degrees(np.arccos(np.clip(an.k.Tn @ an.k.Tn[0], -1, 1)))
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Giro de la tangente respecto a t=0 (°)", "Curvatura κ y torsión τ"])
    fig.add_trace(go.Scatter(x=an.t, y=ang, showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=an.t, y=an.k.curvature, name="κ"), row=1, col=2)
    fig.add_trace(go.Scatter(x=an.t, y=an.k.torsion, name="τ"), row=1, col=2)
    fig.add_hline(y=an.limits.curv_max, line=dict(color=RED, dash="dash"), row=1, col=2)
    for c in (1, 2):
        fig.add_vline(x=t_sel, line=dict(color="orange"), row=1, col=c); fig.update_xaxes(title_text="t", row=1, col=c)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def fig_arc(an, t_sel):
    fig = go.Figure(go.Scatter(x=an.t, y=an.s, name="Longitud de arco s(t)"))
    fig.add_trace(go.Scatter(x=an.t, y=np.linalg.norm(an.r - an.r[0], axis=1), name="Distancia directa |r(t)-r(0)|"))
    fig.add_vline(x=t_sel, line=dict(color="orange"))
    fig.update_layout(height=340, xaxis_title="t", margin=dict(l=10, r=10, t=10, b=10))
    return fig


def fig_local(rows):
    h = [r["h"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h, y=[max(r["error_lineal"], 1e-12) for r in rows], mode="lines+markers", name="Error lineal"))
    fig.add_trace(go.Scatter(x=h, y=[max(r["error_cuadratica"], 1e-12) for r in rows], mode="lines+markers", name="Error cuadrático"))
    fig.update_layout(height=320, xaxis_title="h (desplazamiento)", yaxis_title="|T real − aprox.|",
                      xaxis_type="log", yaxis_type="log", margin=dict(l=10, r=10, t=10, b=10))
    return fig


def fig_compare_3d(analyses):
    cols = {"Original": "#1d3557", "Propuesta 1": "#2a9d8f", "Propuesta 2": "#f77f00"}
    fig = go.Figure()
    for name, an in analyses.items():
        if an is not None:
            fig.add_trace(go.Scatter3d(x=an.r[:, 0], y=an.r[:, 1], z=an.r[:, 2], mode="lines",
                                       line=dict(color=cols.get(name), width=5), name=name))
    fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0), scene=dict(aspectmode="data"))
    return fig


def fig_compare_bars(analyses):
    fig = go.Figure()
    for name, an in analyses.items():
        if an is None: continue
        L = an.limits
        vals = [an.k.speed.max() / L.speed_max, an.k.accel.max() / L.accel_max,
                an.k.curvature.max() / L.curv_max, an.temp.max() / L.temp_max]
        fig.add_trace(go.Bar(name=name, x=["Rapidez", "Aceleración", "Curvatura", "Temperatura"], y=vals))
    fig.add_hline(y=1, line=dict(color=RED, dash="dash"), annotation_text="Límite (100 %)")
    fig.update_layout(barmode="group", height=360, yaxis_title="máximo / límite", margin=dict(l=10, r=10, t=10, b=10))
    return fig
