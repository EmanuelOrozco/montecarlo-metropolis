"""Heisenberg clásico 3D con Monte Carlo Metropolis (bulk, PBC)."""

from __future__ import annotations

import numpy as np

from ..config import KB_EV, INICIALIZACIONES
from ..ising.topologia_cubica import tabla_vecinos_cubica
from ..ising.topologia_material import especificacion_material, tabla_vecinos_material
from ..materials.mp_client import MATERIALES_MP
from .espines import (
    espines_aleatorios,
    espines_antiferromagneticos_bcc,
    espines_antiferromagneticos_paridad,
    espines_ferromagneticos,
    vector_aleatorio_esfera,
)


class HeisenbergClasico:
    """Espines unitarios S_i ∈ S²; H = -J Σ_<ij> S_i·S_j - h·Σ S_i."""

    dimension = 3

    def __init__(
        self,
        geometria: str,
        L: int,
        J: float,
        T: float,
        h: float = 0.0,
        kB: float = KB_EV,
        inicializacion: str = "ferromagnetico",
        seed: int | None = None,
    ) -> None:
        if inicializacion not in INICIALIZACIONES:
            raise ValueError(
                f"inicializacion debe ser una de {INICIALIZACIONES}, "
                f"recibido: {inicializacion!r}"
            )
        if L < 2:
            raise ValueError("L debe ser >= 2")
        if T < 0.1:
            raise ValueError("La temperatura debe ser >= 0.1 (K en unidades físicas)")
        if kB <= 0:
            raise ValueError("kB debe ser positivo")

        self.geometria = geometria
        self.L = L
        self.J = float(J)
        self.h_campo = np.array([0.0, 0.0, float(h)], dtype=float)
        self.T = float(T)
        self.kB = float(kB)
        self.beta = 1.0 / (self.kB * self.T)
        self.inicializacion = inicializacion
        self.rng = np.random.default_rng(seed)

        self.sitios_por_celda, self.z_vecinos = self._meta_red(geometria)
        self.forma = (L, L, L, self.sitios_por_celda)
        self.Nspin = L**3 * self.sitios_por_celda
        self._tabla = self._construir_tabla()
        self.spins = self._crear_espines(inicializacion)
        self.E = self._energia_total()
        self.m_vec = self.spins.reshape(-1, 3).sum(axis=0)

        self.energy_history: list[float] = []
        self.magnetization_history: list[float] = []
        self.grid_history: list[np.ndarray] = []

    @staticmethod
    def _meta_red(geometria: str) -> tuple[int, int]:
        if geometria in MATERIALES_MP:
            spec = especificacion_material(geometria)
            return spec.sitios_por_celda, spec.coordinacion
        if geometria == "cubica":
            return 1, 6
        if geometria == "bcc":
            return 2, 8
        if geometria == "fcc":
            return 4, 12
        raise ValueError(f"Geometría Heisenberg no soportada: {geometria!r}")

    def _construir_tabla(self) -> np.ndarray:
        if self.geometria in MATERIALES_MP:
            return tabla_vecinos_material(self.geometria, self.L)
        if self.geometria in {"bcc", "fcc"}:
            return tabla_vecinos_cubica(self.geometria, self.L)
        # Cúbica simple: construir offsets ±x±y±z
        n = self.Nspin
        tabla = np.empty((n, 6), dtype=np.int32)
        forma = (self.L, self.L, self.L)
        offsets = (
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        )
        for i, j, k in np.ndindex(forma):
            idx = np.ravel_multi_index((i, j, k), forma)
            for p, (di, dj, dk) in enumerate(offsets):
                vec = ((i + di) % self.L, (j + dj) % self.L, (k + dk) % self.L)
                tabla[idx, p] = np.ravel_multi_index(vec, forma)
        return tabla

    def _crear_espines(self, modo: str) -> np.ndarray:
        forma_sitios = self.forma
        if modo == "ferromagnetico":
            return espines_ferromagneticos(forma_sitios)
        if modo == "aleatorio":
            return espines_aleatorios(forma_sitios, self.rng)
        # antiferromagnético
        if self.sitios_por_celda == 2:
            return espines_antiferromagneticos_bcc(forma_sitios)
        return espines_antiferromagneticos_paridad(forma_sitios)

    def _energia_total(self) -> float:
        flat = self.spins.reshape(-1, 3)
        suma_vec = flat[self._tabla].sum(axis=1)
        interaccion = -0.5 * self.J * float(np.einsum("ij,ij->", flat, suma_vec))
        campo = -float(np.dot(flat.sum(axis=0), self.h_campo))
        return interaccion + campo

    def magnetizacion_por_sitio(self) -> float:
        return float(np.linalg.norm(self.m_vec) / self.Nspin)

    def _intentar_voltear(self) -> None:
        idx = int(self.rng.integers(0, self.Nspin))
        flat = self.spins.reshape(-1, 3)
        sold = flat[idx].copy()
        snew = vector_aleatorio_esfera(self.rng)
        suma_vecinos = flat[self._tabla[idx]].sum(axis=0)
        dS = snew - sold
        dE = -self.J * float(np.dot(dS, suma_vecinos)) - float(
            np.dot(dS, self.h_campo)
        )
        if dE <= 0.0 or self.rng.random() < np.exp(-self.beta * dE):
            flat[idx] = snew
            self.E += dE
            self.m_vec += dS

    def simular(self, MCS: int, guardar_red_cada: int = 0) -> None:
        self.energy_history = [self.E]
        self.magnetization_history = [self.magnetizacion_por_sitio()]
        self.grid_history = [self.spins.copy()] if guardar_red_cada else []

        for paso in range(MCS):
            for _ in range(self.Nspin):
                self._intentar_voltear()
            self.energy_history.append(self.E)
            self.magnetization_history.append(self.magnetizacion_por_sitio())
            if guardar_red_cada and (paso + 1) % guardar_red_cada == 0:
                self.grid_history.append(self.spins.copy())

    def momentos_produccion(self, mcs_term: int) -> tuple[float, float, float, float]:
        """⟨E⟩, ⟨|m|⟩, ⟨m²⟩, ⟨E²⟩ desde mcs_term inclusive."""
        e = np.asarray(self.energy_history[mcs_term:], dtype=float)
        m = np.asarray(self.magnetization_history[mcs_term:], dtype=float)
        # |m| ya es por sitio; para χ usamos M = |m|*N
        M = m * self.Nspin
        return (
            float(e.mean()),
            float(m.mean()),
            float((M**2).mean()),
            float((e**2).mean()),
        )


def crear_heisenberg(geometria: str, **kwargs) -> HeisenbergClasico:
    return HeisenbergClasico(geometria=geometria, **kwargs)
