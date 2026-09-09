"""Modelo de Ising 2D con Metropolis.

La lógica de aceptación, dE, actualización de E y m, y el bucle de MCS
es la misma que en avance.py. La única extensión es la red cuadrada:
cada espín tiene cuatro vecinos (arriba, abajo, izquierda, derecha)
con condiciones de frontera periódicas, en lugar de dos.
"""

from __future__ import annotations

import random

import numpy as np


INICIALIZACIONES = ("aleatorio", "ferromagnetico", "antiferromagnetico")


class Ising2D:
    def __init__(
        self,
        L: int,
        J: float,
        h: float,
        T: float,
        kB: float = 1.0,
        inicializacion: str = "aleatorio",
        seed: int | None = None,
    ) -> None:
        if inicializacion not in INICIALIZACIONES:
            raise ValueError(
                f"inicializacion debe ser una de {INICIALIZACIONES}, "
                f"recibido: {inicializacion!r}"
            )

        self.L = L
        self.Nspin = L * L
        self.J = J
        self.h = h
        self.T = T
        self.kB = kB
        self.beta = 1.0 / (self.kB * self.T)
        self.inicializacion = inicializacion

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.spins = self._crear_espines(inicializacion)
        self.E, self.m = self._energia_y_magnetizacion()

        self.energy_history: list[float] = []
        self.magnetization_history: list[float] = []
        self.grid_history: list[np.ndarray] = []

    def _crear_espines(self, modo: str) -> np.ndarray:
        """Tres soluciones de la configuración inicial (valores ±1)."""
        L = self.L
        if modo == "aleatorio":
            return np.random.choice([-1, 1], size=(L, L))

        if modo == "ferromagnetico":
            return np.ones((L, L), dtype=int)

        # Antiferromagnético: tablero de ajedrez, vecinos siempre opuestos.
        i, j = np.indices((L, L))
        return np.where((i + j) % 2 == 0, 1, -1).astype(int)

    def _energia_y_magnetizacion(self) -> tuple[float, float]:
        """E = -J Σ_<ij> s_i s_j - h Σ_i s_i, cada enlace una sola vez."""
        spins = self.spins
        current_m = float(np.sum(spins))
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

    def _intentar_voltear(self) -> None:
        """Un intento de flip Metropolis, igual que en avance.py."""
        idx = random.randint(0, self.Nspin - 1)
        i, j = divmod(idx, self.L)
        old_spin_value = int(self.spins[i, j])

        neighbor_sum = self._suma_vecinos(i, j)
        dE = 2 * (self.J * neighbor_sum + self.h) * old_spin_value

        # Evita overflow de exp a T muy baja: si dE <= 0 se acepta siempre.
        if dE <= 0 or random.random() < np.exp(-self.beta * dE):
            self.spins[i, j] = -old_spin_value
            self.E = self.E + dE
            self.m = self.m - 2 * old_spin_value

    def simular(self, MCS: int, guardar_red_cada: int = 1) -> None:
        """MCS pasos; en cada uno se intentan Nspin flips (como en avance.py)."""
        self.energy_history = [self.E]
        self.magnetization_history = [self.m]
        self.grid_history = [self.spins.copy()]

        for paso in range(MCS):
            for _ in range(self.Nspin):
                self._intentar_voltear()

            self.energy_history.append(self.E)
            self.magnetization_history.append(self.m)

            if (paso + 1) % guardar_red_cada == 0:
                self.grid_history.append(self.spins.copy())

    def promedios(self) -> tuple[float, float]:
        n = len(self.energy_history)
        if n == 0:
            return self.E, self.m
        return float(np.mean(self.energy_history)), float(np.mean(self.magnetization_history))
