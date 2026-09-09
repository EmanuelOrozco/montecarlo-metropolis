"""Ising en red cuadrada: 4 vecinos (arriba, abajo, izquierda, derecha)."""

from __future__ import annotations

from .base import IsingBase


class IsingCuadrada(IsingBase):
    geometria = "cuadrada"
    z_vecinos = 4

    def _energia_y_magnetizacion(self) -> tuple[float, float]:
        spins = self.spins
        current_m = float(spins.sum())
        current_E = 0.0
        L = self.L
        for i in range(L):
            for j in range(L):
                current_E += -self.J * spins[i, j] * spins[i, (j + 1) % L]
                current_E += -self.J * spins[i, j] * spins[(i + 1) % L, j]
                current_E += -self.h * spins[i, j]
        return current_E, current_m

    def _suma_vecinos(self, i: int, j: int) -> int:
        L = self.L
        spins = self.spins
        return int(
            spins[(i - 1) % L, j]
            + spins[(i + 1) % L, j]
            + spins[i, (j - 1) % L]
            + spins[i, (j + 1) % L]
        )
