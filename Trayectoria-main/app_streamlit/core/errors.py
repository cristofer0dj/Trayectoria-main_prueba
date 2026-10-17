"""Errores de dominio. Cada uno lleva un código HTTP equivalente por si el
núcleo se expone luego mediante una API (p. ej. FastAPI)."""


class TrayectoriaError(Exception):
    http_status = 400


class InvalidParameterError(TrayectoriaError):
    """Parámetro no numérico, infinito, fuera de rango o límite no positivo."""
    http_status = 422


class DegenerateGeometryError(TrayectoriaError):
    """Vector nulo / división por cero: velocidad = 0 o r'×r'' = 0."""
    http_status = 422
