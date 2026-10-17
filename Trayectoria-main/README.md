# 🛰️ Sistema de análisis y optimización de una trayectoria espacial

App web interactiva (Streamlit + Plotly) — Proyecto integrador, Matemática III.
Trayectoria `r(t) = ((a+bt)cos t, (a+bt)sen t, ct)`, campo `T(x,y,z) = 20+4x−3y+2z+x²+y²+2z²−xy`.

## Estructura
```
index.html        # versión estática para GitHub Pages (JS + Plotly.js)
app_streamlit/app.py  # UI (presentación): navegación, sliders, tablas, alertas/toasts
app_streamlit/ui/plots.py  # figuras Plotly (3D, dashboard 2D, comparación)
app_streamlit/core/    # lógica matemática pura (sin Streamlit, reutilizable en una API)
  trajectory.py   #   r, r', r'', r'''  (analíticas)
  field.py        #   T, ∇T, dT/dt, derivada direccional, Hessiana
  geometry.py     #   rapidez, aceleración, curvatura, torsión, Frenet-Serret
  metrics.py      #   longitud de arco vs distancia directa
  constraints.py  #   límites de operación y detección de incumplimientos
  local.py        #   aproximación lineal/cuadrática alrededor de un punto
  analysis.py     #   orquestador   ·   recommend.py  ranking y recomendación
  errors.py       #   errores de dominio (con código HTTP equivalente)
app_streamlit/tests/test_core.py
```

