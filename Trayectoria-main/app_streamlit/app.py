"""Sistema de análisis y optimización de una trayectoria espacial — interfaz Streamlit.
Toda la matemática vive en core/; aquí solo hay presentación."""
import numpy as np
import pandas as pd
import streamlit as st

from core.trajectory import Params, DEFAULT_T_END
from core.constraints import Limits
from core.analysis import analyze
from core.errors import TrayectoriaError
from core import local, recommend
from ui import plots

st.set_page_config(page_title="Trayectoria espacial", page_icon="🛰️", layout="wide")

DEFAULTS = {"Original": dict(a=2.0, b=0.5, c=0.5),
            "Propuesta 1": dict(a=2.2, b=0.0, c=0.1),
            "Propuesta 2": dict(a=2.1, b=0.02, c=0.1)}
SECTIONS = ["1 · Título e integrantes", "2 · Planteamiento y modelo", "3 · Visor 3D y parámetros",
            "4 · Análisis dinámico", "5 · Comparación de configuraciones", "6 · Resultados y recomendación"]

if "cfg" not in st.session_state:
    st.session_state.cfg = {k: dict(v) for k, v in DEFAULTS.items()}
    st.session_state.team = "\n".join(['Eduardo Antonio Morales Alfaro — MA202402', 'Ronald Oswaldo Hernandez Rivera — HR202302', 'Cristofer Cesar Rosales Garcia — RG202502', 'Brayan Alexander Orellana Garcia — OG202301'])
    st.session_state.notes = ""
cfg = st.session_state.cfg


@st.cache_data(show_spinner=False)
def run(a, b, c, t_end, lim: tuple):
    """Devuelve (Analysis, None) o (None, mensaje) — nunca lanza hacia la UI."""
    try:
        return analyze(Params(a, b, c, t_end), Limits(*lim)), None
    except TrayectoriaError as e:
        return None, str(e)
    except Exception as e:  # entrada inesperada
        return None, f"Entrada no válida: {e}"


# ---------------------------------------------------------------- sidebar
sb = st.sidebar
sb.title("🛰️ Trayectoria espacial")
section = sb.radio("Navegación", SECTIONS, label_visibility="collapsed")
sb.divider()
sb.subheader("Parámetros")
active = sb.selectbox("Configuración activa", list(DEFAULTS))
for k, (lo, hi) in {"a": (-10.0, 10.0), "b": (-2.0, 2.0), "c": (-2.0, 2.0)}.items():
    key = f"w_{active}_{k}"
    st.session_state[key] = float(np.clip(cfg[active][k], lo, hi))
    cfg[active][k] = sb.slider(k, lo, hi, step=0.01, key=key)
if sb.button("Restablecer esta configuración"):
    cfg[active] = dict(DEFAULTS[active])
    st.rerun()
t_end = sb.slider("Duración del recorrido t_end", 1.0, 6 * np.pi, float(DEFAULT_T_END), 0.01)
if st.session_state.get("t_sel", 0) > t_end:
    st.session_state["t_sel"] = t_end
t_sel = sb.slider("Instante seleccionado t", 0.0, float(t_end), key="t_sel", step=0.01)

with sb.expander("Condiciones de operación"):
    L = Limits(
        st.number_input("Tiempo máximo", min_value=0.0, value=float(4 * np.pi), step=0.1, key="l_t"),
        st.number_input("Rapidez máxima", min_value=0.0, value=3.0, step=0.1, key="l_v"),
        st.number_input("Aceleración máxima", min_value=0.0, value=2.5, step=0.1, key="l_a"),
        st.number_input("Temperatura máxima", min_value=0.0, value=50.0, step=1.0, key="l_T"),
        st.number_input("Curvatura máxima", min_value=0.0, value=0.5, step=0.01, key="l_k"))
lim = (L.t_max, L.speed_max, L.accel_max, L.temp_max, L.curv_max)

results = {n: run(c_["a"], c_["b"], c_["c"], float(t_end), lim) for n, c_ in cfg.items()}
analyses = {n: r[0] for n, r in results.items()}
an, err = results[active]

# ---------------------------------------------------------------- toasts (solo si cambia algo)
sig = (active, tuple(cfg[active].values()), float(t_end), lim)
if st.session_state.get("sig") != sig:
    st.session_state["sig"] = sig
    if an is not None:
        if an.ok:
            st.toast(f"{active}: cumple todas las condiciones", icon="✅")
        for v in an.violations[:5]:
            st.toast(f"{v.condition}: {v.value:.2f} > {v.limit:.2f} (t={v.t_first:.2f})", icon="⚠️")
    elif err:
        st.toast(err, icon="🚫")


def alerts(an):
    if an.ok:
        st.success(f"✅ **{active} cumple** todas las condiciones de operación.")
        return
    st.error(f"❌ **{active} incumple {len(an.violations)} condición(es).**")
    rows = [{"Condición": v.condition, "t (primer incumplimiento)": round(v.t_first, 3),
             "Posición (x, y, z)": "({:.2f}, {:.2f}, {:.2f})".format(*v.pos_first),
             "t del pico": round(v.t_peak, 3), "Valor registrado": round(v.value, 3),
             "Límite": round(v.limit, 3), "Exceso": round(v.excess, 3), "Exceso %": round(v.excess_pct, 1)}
            for v in an.violations]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def need_valid():
    if an is None:
        st.error(f"🚫 **Configuración «{active}» no válida.** {err}  \nCorrige los parámetros en el panel lateral; "
                 "el renderizado queda bloqueado hasta entonces.")
        st.stop()


# ================================================================ 1
if section == SECTIONS[0]:
    st.title("Sistema de análisis y optimización de una trayectoria espacial")
    st.caption("Proyecto integrador · Matemática III · Aplicación interactiva")
    st.session_state.team = st.text_area("Integrantes (uno por línea, editable)", st.session_state.team, height=130)
    st.markdown("**¿Cómo usar la app?** Elige una sección en el menú lateral, ajusta *a*, *b*, *c* y los límites, "
                "y todo (3D, gráficas, alertas, comparación y recomendación) se recalcula al instante.")
    st.info("Pregunta central: ¿la trayectoria propuesta es adecuada para la misión bajo las condiciones de operación?")

# ================================================================ 2
elif section == SECTIONS[1]:
    st.header("Planteamiento y modelo matemático")
    st.markdown("Un vehículo autónomo inspecciona una estructura industrial siguiendo una trayectoria 3D. "
                "Debe cumplir límites de tiempo, rapidez, aceleración, temperatura y curvatura.")
    st.latex(r"\mathbf r_{a,b,c}(t)=(a+bt)\cos t\,\mathbf i+(a+bt)\sin t\,\mathbf j+ct\,\mathbf k,\quad 0\le t\le 4\pi")
    st.latex(r"T(x,y,z)=20+4x-3y+2z+x^2+y^2+2z^2-xy")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cinemática y geometría**")
        st.latex(r"\mathbf v=\mathbf r',\ \ \mathbf a=\mathbf r'',\ \ \text{rapidez}=|\mathbf r'|")
        st.latex(r"\kappa=\frac{|\mathbf r'\times\mathbf r''|}{|\mathbf r'|^3},\quad \tau=\frac{(\mathbf r'\times\mathbf r'')\cdot\mathbf r'''}{|\mathbf r'\times\mathbf r''|^2}")
        st.latex(r"\mathbf T=\frac{\mathbf r'}{|\mathbf r'|},\ \mathbf B=\frac{\mathbf r'\times\mathbf r''}{|\mathbf r'\times\mathbf r''|},\ \mathbf N=\mathbf B\times\mathbf T")
        st.latex(r"s(t)=\int_0^t|\mathbf r'(u)|\,du")
    with c2:
        st.markdown("**Entorno y análisis local**")
        st.latex(r"\nabla T=(4+2x-y,\ -3+2y-x,\ 2+4z)")
        st.latex(r"\frac{dT}{dt}=\nabla T\cdot\mathbf r',\qquad D_{\mathbf T}T=\nabla T\cdot\mathbf T")
        st.latex(r"L(\mathbf p)=T(\mathbf p_0)+\nabla T(\mathbf p_0)\cdot(\mathbf p-\mathbf p_0)")
        st.latex(r"T(\mathbf p_0+\mathbf h)-L=\tfrac12\,\mathbf h^{\!\top}H\,\mathbf h,\quad H=\begin{pmatrix}2&-1&0\\-1&2&0\\0&0&4\end{pmatrix}")
    st.markdown("**Condiciones iniciales:** tiempo ≤ 4π · rapidez ≤ 3.00 · aceleración ≤ 2.50 · T ≤ 50 · κ ≤ 0.50 "
                "(modificables en el panel lateral). Como T es cuadrática, el error de la aproximación lineal es exactamente "
                "½ hᵀHh.")

# ================================================================ 3
elif section == SECTIONS[2]:
    st.header("Visor 3D y controles")
    need_valid()
    alerts(an)
    o = st.columns(5)
    opts = [o[0].checkbox("Campo (isosuperficies)", True), o[1].checkbox("Vectores v, a, ∇T", True),
            o[2].checkbox("Triedro de Frenet", True), o[3].checkbox("Plano tangente", True),
            o[4].checkbox("Zonas de incumplimiento", True)]
    st.plotly_chart(plots.fig_3d(an, an.at(t_sel), *opts), use_container_width=True)
    st.caption("Rota con el ratón, zoom con la rueda, pan con clic derecho. El color de la curva es la temperatura; "
               "la superficie roja es T = límite (el interior es la zona permitida).")

# ================================================================ 4
elif section == SECTIONS[3]:
    st.header("Panel de análisis dinámico")
    need_valid()
    alerts(an)
    i = an.at(t_sel)
    m = st.columns(5)
    m[0].metric("t", f"{an.t[i]:.2f}"); m[1].metric("Rapidez", f"{an.k.speed[i]:.3f}")
    m[2].metric("Aceleración", f"{an.k.accel[i]:.3f}"); m[3].metric("Curvatura", f"{an.k.curvature[i]:.3f}")
    m[4].metric("Temperatura", f"{an.temp[i]:.2f}")
    t1, t2, t3, t4, t5 = st.tabs(["Gráficas", "Entorno", "Geometría", "Distancia", "Análisis local"])
    with t1:
        st.plotly_chart(plots.fig_dashboard(an, t_sel), use_container_width=True)
    with t2:
        g = an.grad[i]
        st.write(f"**Posición:** ({an.r[i,0]:.3f}, {an.r[i,1]:.3f}, {an.r[i,2]:.3f}) · **∇T** = ({g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}) · "
                 f"|∇T| = {np.linalg.norm(g):.3f} (máxima tasa de cambio por unidad de distancia)")
        d = an.dirder[i]
        st.write(f"**Derivada direccional a lo largo del avance:** D_T T = {d:.3f}; **dT/dt** = {an.dTdt[i]:.3f} → "
                 + ("el vehículo va hacia zonas **más calientes**." if d > 0 else "el vehículo va hacia zonas **más frías**."))
        j = int(np.argmax(np.abs(an.dTdt)))
        st.write(f"T varía más rápido en t = {an.t[j]:.2f} (|dT/dt| = {abs(an.dTdt[j]):.2f}). "
                 f"T máx = {an.temp.max():.2f} en t = {an.t[np.argmax(an.temp)]:.2f}; T mín = {an.temp.min():.2f} en t = {an.t[np.argmin(an.temp)]:.2f}.")
        st.plotly_chart(plots.fig_env(an, t_sel), use_container_width=True)
    with t3:
        k = an.k
        st.dataframe(pd.DataFrame({"Vector": ["r", "v = r'", "a = r''", "T̂", "N̂", "B̂"],
                                   "x": [an.r[i,0], k.v[i,0], k.a[i,0], k.Tn[i,0], k.N[i,0], k.B[i,0]],
                                   "y": [an.r[i,1], k.v[i,1], k.a[i,1], k.Tn[i,1], k.N[i,1], k.B[i,1]],
                                   "z": [an.r[i,2], k.v[i,2], k.a[i,2], k.Tn[i,2], k.N[i,2], k.B[i,2]]}).round(4),
                     hide_index=True)
        st.write(f"κ = {k.curvature[i]:.4f} (radio de curvatura {1/k.curvature[i]:.2f}) · τ = {k.torsion[i]:.4f}")
        j = int(np.argmax(k.curvature))
        st.write(f"Mayor cambio de dirección: t = {an.t[j]:.2f}, κ = {k.curvature[j]:.4f} "
                 f"({100*k.curvature[j]/an.limits.curv_max:.0f} % del límite).")
        st.plotly_chart(plots.fig_orientation(an, t_sel), use_container_width=True)
    with t4:
        c = st.columns(4)
        c[0].metric("Longitud de arco total", f"{an.arc:.3f}"); c[1].metric("Distancia directa", f"{an.straight:.3f}")
        c[2].metric("Recorrido / directa", f"{an.ratio:.2f}×"); c[3].metric("Acumulado hasta t", f"{an.s[i]:.3f}")
        st.progress(float(min(an.s[i] / an.arc, 1.0)), text=f"{100*an.s[i]/an.arc:.1f} % del recorrido")
        st.plotly_chart(plots.fig_arc(an, t_sel), use_container_width=True)
    with t5:
        P0 = an.r[i]
        dname = st.selectbox("Dirección del pequeño desplazamiento", list(local.DIRECTIONS))
        u = {"T": an.k.Tn[i], "N": an.k.N[i], "B": an.k.B[i], "G": an.grad[i]}[local.DIRECTIONS[dname]]
        if np.linalg.norm(u) < 1e-9:
            st.warning("El gradiente es nulo en este punto: el campo es localmente plano, elige otra dirección.")
        else:
            hs = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
            rows = local.compare_along_direction(P0, u, hs)
            st.write(f"T(p₀) = {an.temp[i]:.3f}; modelo lineal: L(p) = T(p₀) + ∇T·(p − p₀)")
            st.dataframe(pd.DataFrame(rows).rename(columns={"h": "h", "real": "T real", "lineal": "T lineal",
                         "cuadratica": "T cuadrática", "error_lineal": "Error lineal", "error_rel_pct": "Error lineal %",
                         "error_cuadratica": "Error cuadrático"}).round(5), hide_index=True, use_container_width=True)
            st.plotly_chart(plots.fig_local(rows), use_container_width=True)
            st.caption("El error lineal crece como h² (½ hᵀHh); el cuadrático es ≈ 0 porque T es un polinomio de grado 2.")
            tr_rows = local.compare_along_trajectory(an.params, float(an.t[i]), [-0.5, -0.25, 0.25, 0.5, 1.0])
            st.write("**A lo largo de la trayectoria:** T(r(t₀+Δt)) ≈ T(t₀) + (dT/dt)·Δt")
            st.dataframe(pd.DataFrame(tr_rows).rename(columns={"dt": "Δt", "real": "T real", "lineal": "T lineal", "error": "Error"}).round(5),
                         hide_index=True)

# ================================================================ 5
elif section == SECTIONS[4]:
    st.header("Comparación de configuraciones")
    for n, (a_, e_) in results.items():
        if a_ is None:
            st.error(f"🚫 {n}: {e_}")
    ok_an = {n: a_ for n, a_ in analyses.items() if a_ is not None}
    if not ok_an:
        st.stop()

    def fmt(v, lim_, a_):
        return f"{v:.3f} {'✅' if v <= lim_ else '❌'} (≤ {lim_:g})"
    rows = {"a": {}, "b": {}, "c": {}, "Estado": {}, "Condiciones incumplidas": {}, "Duración t_end": {},
            "Rapidez máx.": {}, "Aceleración máx.": {}, "Curvatura máx.": {}, "Temperatura máx.": {},
            "Longitud de arco": {}, "Distancia directa": {}, "Recorrido / directa": {}, "T inicial → final": {}}
    for n, a_ in analyses.items():
        if a_ is None:
            for r_ in rows: rows[r_][n] = "—"
            rows["Estado"][n] = "🚫 Inválida"; continue
        p, Lm = a_.params, a_.limits
        rows["a"][n], rows["b"][n], rows["c"][n] = f"{p.a:g}", f"{p.b:g}", f"{p.c:g}"
        rows["Estado"][n] = "✅ CUMPLE" if a_.ok else f"❌ INCUMPLE ({len(a_.violations)})"
        rows["Condiciones incumplidas"][n] = ", ".join(v.condition.replace(" máxima", "").replace(" máximo de recorrido", "") for v in a_.violations) or "Ninguna"
        rows["Duración t_end"][n] = f"{p.t_end:.3f} {'✅' if p.t_end <= Lm.t_max + 1e-9 else '❌'} (≤ {Lm.t_max:.3f})"
        rows["Rapidez máx."][n] = fmt(a_.k.speed.max(), Lm.speed_max, a_)
        rows["Aceleración máx."][n] = fmt(a_.k.accel.max(), Lm.accel_max, a_)
        rows["Curvatura máx."][n] = fmt(a_.k.curvature.max(), Lm.curv_max, a_)
        rows["Temperatura máx."][n] = fmt(a_.temp.max(), Lm.temp_max, a_)
        rows["Longitud de arco"][n] = f"{a_.arc:.3f}"
        rows["Distancia directa"][n] = f"{a_.straight:.3f}"
        rows["Recorrido / directa"][n] = f"{a_.ratio:.2f}×"
        rows["T inicial → final"][n] = f"{a_.temp[0]:.2f} → {a_.temp[-1]:.2f}"
    st.dataframe(pd.DataFrame(rows).T[list(DEFAULTS)], use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(plots.fig_compare_bars(ok_an), use_container_width=True)
    c2.plotly_chart(plots.fig_compare_3d(ok_an), use_container_width=True)
    st.caption("Edita cada configuración eligiéndola en «Configuración activa» del panel lateral.")

# ================================================================ 6
else:
    st.header("Resultados, recomendación final y justificación")
    rec = recommend.recommend(analyses)
    if rec is None:
        st.error("No hay ninguna configuración válida que evaluar."); st.stop()
    best = rec["best"]
    st.subheader("Decisión")
    st.markdown(rec["text"])
    b = analyses[best]
    st.subheader("Evidencia matemática")
    ev = []
    for n in rec["ranking"]:
        a_ = analyses[n]
        ev.append({"Configuración": n, "(a, b, c)": f"({a_.params.a:g}, {a_.params.b:g}, {a_.params.c:g})",
                   "Cumple": "Sí" if a_.ok else "No", "v máx": round(a_.k.speed.max(), 3),
                   "a máx": round(a_.k.accel.max(), 3), "κ máx": round(a_.k.curvature.max(), 3),
                   "T máx": round(a_.temp.max(), 2), "Longitud": round(a_.arc, 2)})
    st.dataframe(pd.DataFrame(ev), hide_index=True, use_container_width=True)
    st.markdown(f"**Justificación:** en una hélice/espiral suave la rapidez es |r'|=√((a+bt)²+b²+c²) y la aceleración "
                f"|r''|=√((a+bt)²+4b²); ambas crecen con el radio a+bt, mientras que la curvatura ≈ 1/radio disminuye. "
                f"Por eso el radio debe ser lo bastante grande para κ ≤ {L.curv_max:g} y lo bastante pequeño para v ≤ {L.speed_max:g} "
                f"y |a| ≤ {L.accel_max:g}; un c pequeño evita subir demasiado en z y limita T (término 2z+2z²). "
                f"{best} equilibra estas condiciones: ocupa {100*b.k.speed.max()/L.speed_max:.0f} % del límite de rapidez, "
                f"{100*b.k.accel.max()/L.accel_max:.0f} % de aceleración, {100*b.k.curvature.max()/L.curv_max:.0f} % de curvatura "
                f"y {100*b.temp.max()/L.temp_max:.0f} % de temperatura.")
    st.session_state.notes = st.text_area("Conclusiones y comentarios propios del grupo (editable)", st.session_state.notes, height=140)
