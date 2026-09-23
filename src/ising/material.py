"""Modelo de Ising para estructuras descargadas de Materials Project."""

from __future__ import annotations

import numpy as np

from .cubicas_centradas import IsingCubicaConBase
from .topologia_material import especificacion_material, tabla_vecinos_material


class IsingMaterialMP(IsingCubicaConBase):
    """Red 3D con base cristalográfica tomada de Materials Project."""

    geometria = "material"
    material_id: str = ""

    @property
    def _tabla_vecinos(self) -> np.ndarray:
        tabla = getattr(self, "_vecinos_cache", None)
        if tabla is None:
            tabla = tabla_vecinos_material(self.geometria, self.L)
            self._vecinos_cache = tabla
        return tabla


class IsingFeMP13(IsingMaterialMP):
    """Hierro α (mp-13): celda convencional BCC Im-3m, z=8."""

    geometria = "fe_mp13"
    sitios_por_celda = especificacion_material("fe_mp13").sitios_por_celda
    z_vecinos = especificacion_material("fe_mp13").coordinacion
    material_id = "mp-13"

    def _crear_espines(self, modo: str) -> np.ndarray:
        if modo != "antiferromagnetico":
            return super()._crear_espines(modo)
        # BCC bipartita: esquinas vs centros de celda.
        spins = np.ones(self.forma, dtype=int)
        spins[:, :, :, 1] = -1
        return spins


class IsingNiMP23(IsingMaterialMP):
    """Níquel (mp-23): celda convencional FCC Fm-3m, z=12."""

    geometria = "ni_mp23"
    sitios_por_celda = especificacion_material("ni_mp23").sitios_por_celda
    z_vecinos = especificacion_material("ni_mp23").coordinacion
    material_id = "mp-23"


class IsingCoMP102(IsingMaterialMP):
    """Cobalto (mp-102): celda convencional FCC Fm-3m, z=12."""

    geometria = "co_mp102"
    sitios_por_celda = especificacion_material("co_mp102").sitios_por_celda
    z_vecinos = especificacion_material("co_mp102").coordinacion
    material_id = "mp-102"
