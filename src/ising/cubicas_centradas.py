"""Modelos Ising BCC y FCC con una base cristalográfica por celda."""

from __future__ import annotations

import numpy as np

from .base import IsingBase
from .topologia_cubica import especificacion_cubica, tabla_vecinos_cubica


class IsingCubicaConBase(IsingBase):
    """Implementación común para redes cúbicas con varios sitios por celda."""

    dimension = 3

    @property
    def _tabla_vecinos(self) -> np.ndarray:
        tabla = getattr(self, "_vecinos_cache", None)
        if tabla is None:
            tabla = tabla_vecinos_cubica(self.geometria, self.L)
            self._vecinos_cache = tabla
        return tabla

    def _energia_y_magnetizacion(self) -> tuple[float, float]:
        spins = self.spins.ravel()
        suma_vecinos = spins[self._tabla_vecinos].sum(axis=1)
        interaccion = -0.5 * self.J * float(np.dot(spins, suma_vecinos))
        current_m = float(spins.sum())
        return interaccion - self.h * current_m, current_m

    def _suma_vecinos(self, i: int, j: int, k: int, base: int) -> int:
        indice = (((i * self.L + j) * self.L + k) * self.sitios_por_celda) + base
        return int(self.spins.reshape(-1)[self._tabla_vecinos[indice]].sum())


class IsingBCC(IsingCubicaConBase):
    """Cúbica centrada en el cuerpo: 2 sitios/celda y z=8."""

    geometria = "bcc"
    sitios_por_celda = especificacion_cubica("bcc").sitios_por_celda
    z_vecinos = especificacion_cubica("bcc").coordinacion

    def _crear_espines(self, modo: str) -> np.ndarray:
        if modo != "antiferromagnetico":
            return super()._crear_espines(modo)
        # BCC es bipartita: esquinas y centros forman subredes opuestas.
        spins = np.ones(self.forma, dtype=int)
        spins[:, :, :, 1] = -1
        return spins


class IsingFCC(IsingCubicaConBase):
    """Cúbica centrada en las caras: 4 sitios/celda y z=12."""

    geometria = "fcc"
    sitios_por_celda = especificacion_cubica("fcc").sitios_por_celda
    z_vecinos = especificacion_cubica("fcc").coordinacion
