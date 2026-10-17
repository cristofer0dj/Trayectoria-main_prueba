"""Ranking de configuraciones y texto de recomendación automática."""


def score(an):
    """Menor es mejor: (nº de condiciones incumplidas, exceso relativo total, longitud)."""
    rel = sum(max(v.excess, 0) / v.limit for v in an.violations)
    return (len(an.violations), rel, an.arc)


def recommend(analyses: dict):
    valid = {k: v for k, v in analyses.items() if v is not None}
    if not valid:
        return None
    ranking = sorted(valid, key=lambda k: score(valid[k]))
    best = ranking[0]
    orig = valid.get("Original")
    lines = []
    if orig is not None and orig.ok:
        lines.append("La trayectoria **original cumple** todas las condiciones: se recomienda mantenerla.")
    else:
        if orig is not None:
            names = ", ".join(v.condition.lower() for v in orig.violations)
            lines.append(f"La trayectoria **original no es viable**: incumple {names}.")
        if best != "Original" and valid[best].ok:
            b = valid[best]; p = b.params
            lines.append(f"Se recomienda **{best}** (a={p.a:g}, b={p.b:g}, c={p.c:g}): cumple todas las "
                         f"condiciones, con longitud de arco {b.arc:.2f} y distancia directa {b.straight:.2f}.")
        elif best != "Original":
            lines.append(f"Ninguna configuración cumple todo; la menos mala es **{best}** "
                         f"({len(valid[best].violations)} condición(es) incumplida(s)). Se sugiere seguir ajustando a, b y c.")
        else:
            lines.append("Ninguna propuesta mejora a la original; se sugiere seguir ajustando a, b y c.")
    return dict(best=best, ranking=ranking, text="\n\n".join(lines))
