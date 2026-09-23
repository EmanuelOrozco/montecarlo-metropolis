"""Fábrica de modelos Ising 2D y 3D según la geometría."""

from __future__ import annotations

from typing import Type

import numpy as np

from .base import IsingBase
from .cubica import IsingCubica
from .cubicas_centradas import IsingBCC, IsingFCC
from .cuadrada import IsingCuadrada
from .material import IsingCoMP102, IsingFeMP13, IsingNiMP23
from .topologia_cubica import especificacion_cubica
from .topologia_material import especificacion_material
from .triangular import IsingTriangular

REDES = ("cuadrada", "triangular", "cubica", "bcc", "fcc", "fe_mp13", "ni_mp23", "co_mp102")
MATERIALES = ("fe_mp13", "ni_mp23", "co_mp102")

# Alias histórico.
Ising2D = IsingCuadrada

_CLASES: dict[str, Type[IsingBase]] = {
    "cuadrada": IsingCuadrada,
    "triangular": IsingTriangular,
    "cubica": IsingCubica,
    "bcc": IsingBCC,
    "fcc": IsingFCC,
    "fe_mp13": IsingFeMP13,
    "ni_mp23": IsingNiMP23,
    "co_mp102": IsingCoMP102,
}


def clase_ising(geometria: str) -> Type[IsingBase]:
    if geometria not in _CLASES:
        raise ValueError(f"geometria debe ser una de {REDES}, recibido: {geometria!r}")
    return _CLASES[geometria]


def crear_ising(geometria: str, **kwargs) -> IsingBase:
    return clase_ising(geometria)(**kwargs)


def dimension_red(geometria: str) -> int:
    """Dimensión espacial declarada por la geometría."""
    return clase_ising(geometria).dimension


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
    if geometria in {"cubica", "bcc", "fcc"}:
        # En 3D se usan estimaciones numéricas aceptadas, no soluciones exactas.
        return especificacion_cubica(geometria).tc_referencia * J / kB
    if geometria in MATERIALES:
        return especificacion_material(geometria).tc_referencia * J / kB
    raise ValueError(f"geometria desconocida: {geometria!r}")


def etiqueta_red(geometria: str) -> str:
    if geometria in MATERIALES:
        return especificacion_material(geometria).nombre
    return {
        "cuadrada": "Red cuadrada (2D, 4 vecinos)",
        "triangular": "Red triangular (2D, 6 vecinos)",
        "cubica": "Red cúbica simple (3D, 6 vecinos)",
        "bcc": "Red cúbica centrada en el cuerpo (BCC, 3D, 8 vecinos)",
        "fcc": "Red cúbica centrada en las caras (FCC, 3D, 12 vecinos)",
    }.get(geometria, geometria)


__all__ = [
    "IsingBase",
    "IsingCuadrada",
    "IsingTriangular",
    "IsingCubica",
    "IsingBCC",
    "IsingFCC",
    "IsingFeMP13",
    "IsingNiMP23",
    "IsingCoMP102",
    "Ising2D",
    "REDES",
    "MATERIALES",
    "clase_ising",
    "crear_ising",
    "dimension_red",
    "tc_teorica",
    "etiqueta_red",
]
