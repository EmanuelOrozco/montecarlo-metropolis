"""Fábrica de modelos Ising según la geometría de la red."""

from __future__ import annotations

from typing import Type

import numpy as np

from .base import IsingBase
from .cuadrada import IsingCuadrada
from .triangular import IsingTriangular

REDES = ("cuadrada", "triangular")

# Alias histórico.
Ising2D = IsingCuadrada

_CLASES: dict[str, Type[IsingBase]] = {
    "cuadrada": IsingCuadrada,
    "triangular": IsingTriangular,
}


def clase_ising(geometria: str) -> Type[IsingBase]:
    if geometria not in _CLASES:
        raise ValueError(f"geometria debe ser una de {REDES}, recibido: {geometria!r}")
    return _CLASES[geometria]


def crear_ising(geometria: str, **kwargs) -> IsingBase:
    return clase_ising(geometria)(**kwargs)


def tc_teorica(geometria: str, J: float = 1.0, kB: float = 1.0) -> float:
    """Temperatura crítica exacta (ferromagnético, h = 0)."""
    if geometria == "cuadrada":
        # Onsager: Tc = 2J / ln(1+√2)
        return 2.0 * J / (kB * np.log(1.0 + np.sqrt(2.0)))
    if geometria == "triangular":
        # Triangular: Tc = 4J / ln(3)
        return 4.0 * J / (kB * np.log(3.0))
    raise ValueError(f"geometria desconocida: {geometria!r}")


def etiqueta_red(geometria: str) -> str:
    return {
        "cuadrada": "Red cuadrada (4 vecinos)",
        "triangular": "Red triangular (6 vecinos)",
    }.get(geometria, geometria)


__all__ = [
    "IsingBase",
    "IsingCuadrada",
    "IsingTriangular",
    "Ising2D",
    "REDES",
    "clase_ising",
    "crear_ising",
    "tc_teorica",
    "etiqueta_red",
]
