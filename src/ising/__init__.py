"""Fábrica de modelos Ising 2D y 3D según la geometría."""

from __future__ import annotations

from typing import Type

import numpy as np

from .base import IsingBase
from .cubica import IsingCubica
from .cuadrada import IsingCuadrada
from .triangular import IsingTriangular

REDES = ("cuadrada", "triangular", "cubica")

# Alias histórico.
Ising2D = IsingCuadrada

_CLASES: dict[str, Type[IsingBase]] = {
    "cuadrada": IsingCuadrada,
    "triangular": IsingTriangular,
    "cubica": IsingCubica,
}


def clase_ising(geometria: str) -> Type[IsingBase]:
    if geometria not in _CLASES:
        raise ValueError(f"geometria debe ser una de {REDES}, recibido: {geometria!r}")
    return _CLASES[geometria]


def crear_ising(geometria: str, **kwargs) -> IsingBase:
    return clase_ising(geometria)(**kwargs)


def tc_teorica(geometria: str, J: float = 1.0, kB: float = 1.0) -> float:
    """Tc de referencia del Ising ferromagnético con h=0.

    Las expresiones 2D son exactas. Para la red cúbica simple se usa el
    valor numérico aceptado porque el modelo 3D no tiene solución exacta.
    """
    if geometria == "cuadrada":
        # Onsager: Tc = 2J / ln(1+√2)
        return 2.0 * J / (kB * np.log(1.0 + np.sqrt(2.0)))
    if geometria == "triangular":
        # Triangular: Tc = 4J / ln(3)
        return 4.0 * J / (kB * np.log(3.0))
    if geometria == "cubica":
        # Estimación numérica de alta precisión: kB Tc / J ≈ 4.511524.
        return 4.511524 * J / kB
    raise ValueError(f"geometria desconocida: {geometria!r}")


def etiqueta_red(geometria: str) -> str:
    return {
        "cuadrada": "Red cuadrada (2D, 4 vecinos)",
        "triangular": "Red triangular (2D, 6 vecinos)",
        "cubica": "Red cúbica simple (3D, 6 vecinos)",
    }.get(geometria, geometria)


__all__ = [
    "IsingBase",
    "IsingCuadrada",
    "IsingTriangular",
    "IsingCubica",
    "Ising2D",
    "REDES",
    "clase_ising",
    "crear_ising",
    "tc_teorica",
    "etiqueta_red",
]
