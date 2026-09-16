"""Modelo de Ising en red cúbica simple 3D.

Cada sitio (i, j, k) tiene seis primeros vecinos: ±x, ±y y ±z.
Se aplican condiciones de frontera periódicas en las tres direcciones.
"""

from __future__ import annotations

from .base import IsingBase


class IsingCubica(IsingBase):
    """Red L×L×L con coordinación z=6."""

    geometria = "cubica"
    dimension = 3
    z_vecinos = 6

    def _energia_y_magnetizacion(self) -> tuple[float, float]:
        """Cuenta una vez los enlaces salientes en +x, +y y +z."""
        spins = self.spins
        current_m = float(spins.sum())
        current_E = 0.0
        L = self.L
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    s = spins[i, j, k]
                    current_E += -self.J * s * spins[(i + 1) % L, j, k]
                    current_E += -self.J * s * spins[i, (j + 1) % L, k]
                    current_E += -self.J * s * spins[i, j, (k + 1) % L]
                    current_E += -self.h * s
        return current_E, current_m

    def _suma_vecinos(self, i: int, j: int, k: int) -> int:
        L = self.L
        spins = self.spins
        return int(
            spins[(i - 1) % L, j, k]
            + spins[(i + 1) % L, j, k]
            + spins[i, (j - 1) % L, k]
            + spins[i, (j + 1) % L, k]
            + spins[i, j, (k - 1) % L]
            + spins[i, j, (k + 1) % L]
        )
