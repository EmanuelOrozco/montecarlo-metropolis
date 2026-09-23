"""Integración con Materials Project."""

from .mp_client import (
    MATERIALES_MP,
    asegurar_estructura,
    material_id_de_geometria,
)

__all__ = [
    "MATERIALES_MP",
    "asegurar_estructura",
    "material_id_de_geometria",
]
