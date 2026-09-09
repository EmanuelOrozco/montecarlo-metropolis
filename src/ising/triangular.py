"""Ising en red triangular: 6 vecinos (cuadrícula + dos diagonales).

Se representa sobre una malla L×L con PBC. Vecinos de (i, j):
  (i, j±1), (i±1, j), (i−1, j+1), (i+1, j−1).
"""

from __future__ import annotations

from .base import IsingBase


class IsingTriangular(IsingBase):
    geometria = "triangular"
    z_vecinos = 6

    def _energia_y_magnetizacion(self) -> tuple[float, float]:
        """Tres enlaces salientes por sitio → cada bond una vez (z = 6)."""
        spins = self.spins
        current_m = float(spins.sum())
        current_E = 0.0
        L = self.L
        for i in range(L):
            for j in range(L):
                s = spins[i, j]
                current_E += -self.J * s * spins[i, (j + 1) % L]
                current_E += -self.J * s * spins[(i + 1) % L, j]
                current_E += -self.J * s * spins[(i + 1) % L, (j - 1) % L]
                current_E += -self.h * s
        return current_E, current_m

    def _suma_vecinos(self, i: int, j: int) -> int:
        L = self.L
        spins = self.spins
        return int(
            spins[i, (j + 1) % L]
            + spins[i, (j - 1) % L]
            + spins[(i + 1) % L, j]
            + spins[(i - 1) % L, j]
            + spins[(i - 1) % L, (j + 1) % L]
            + spins[(i + 1) % L, (j - 1) % L]
        )
