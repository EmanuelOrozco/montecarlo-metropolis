"""Núcleo compartido del modelo de Ising en redes 2D y 3D."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

import numpy as np

INICIALIZACIONES = ("aleatorio", "ferromagnetico", "antiferromagnetico")


class IsingBase(ABC):
    """Red hipercúbica de espines ±1; la subclase define su geometría."""

    geometria: str = "base"
    dimension: int = 2
    sitios_por_celda: int = 1
    z_vecinos: int = 0

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

        if L < 2:
            raise ValueError("L debe ser mayor o igual que 2")
        if T <= 0:
            raise ValueError("T debe ser mayor que 0")

        self.L = L
        forma_celdas = (L,) * self.dimension
        self.forma = (
            forma_celdas
            if self.sitios_por_celda == 1
            else (*forma_celdas, self.sitios_por_celda)
        )
        self.Nspin = L**self.dimension * self.sitios_por_celda
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
        if modo == "aleatorio":
            return np.random.choice([-1, 1], size=self.forma)
        if modo == "ferromagnetico":
            return np.ones(self.forma, dtype=int)
        # Paridad alternada; en triangular el AFM está geométricamente frustrado.
        paridad = np.indices(self.forma).sum(axis=0)
        return np.where(paridad % 2 == 0, 1, -1).astype(int)

    @abstractmethod
    def _energia_y_magnetizacion(self) -> tuple[float, float]:
        """Energía total y magnetización; cada enlace se cuenta una sola vez."""

    @abstractmethod
    def _suma_vecinos(self, *coordenadas: int) -> int:
        """Suma de vecinos de un sitio con fronteras periódicas."""

    def _intentar_voltear(self) -> None:
        idx = random.randint(0, self.Nspin - 1)
        coordenadas = np.unravel_index(idx, self.forma)
        old_spin_value = int(self.spins[coordenadas])

        neighbor_sum = self._suma_vecinos(*coordenadas)
        dE = 2 * (self.J * neighbor_sum + self.h) * old_spin_value

        if dE <= 0 or random.random() < np.exp(-self.beta * dE):
            self.spins[coordenadas] = -old_spin_value
            self.E = self.E + dE
            self.m = self.m - 2 * old_spin_value

    def simular(self, MCS: int, guardar_red_cada: int = 1) -> None:
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
