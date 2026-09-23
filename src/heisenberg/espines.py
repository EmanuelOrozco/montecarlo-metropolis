"""Utilidades de espines vectoriales unitarios (Heisenberg clásico)."""

from __future__ import annotations

import numpy as np


def vector_aleatorio_esfera(rng: np.random.Generator, size: tuple[int, ...] = ()) -> np.ndarray:
    """Vector(es) unitario(s) uniforme(s) en S² (método gaussiano, sin sesgo).

    Muestrear N(0,1)³ y normalizar es equivalente a la medida uniforme
    en la esfera (Marsaglia / Box-Muller en 3D).
    """
    forma = (*size, 3) if size else (3,)
    v = rng.normal(size=forma)
    normas = np.linalg.norm(v, axis=-1, keepdims=True)
    # Evita división por cero (probabilidad nula en la práctica).
    normas = np.maximum(normas, 1e-15)
    return v / normas


def espines_ferromagneticos(forma_sitios: tuple[int, ...]) -> np.ndarray:
    """Todos los espines apuntan a +z."""
    spins = np.zeros((*forma_sitios, 3), dtype=float)
    spins[..., 2] = 1.0
    return spins


def espines_antiferromagneticos_bcc(forma: tuple[int, int, int, int]) -> np.ndarray:
    """BCC bipartita: base 0 → +z, base 1 → −z."""
    spins = np.zeros((*forma, 3), dtype=float)
    spins[..., 0, 2] = 1.0
    spins[..., 1, 2] = -1.0
    return spins


def espines_antiferromagneticos_paridad(forma_sitios: tuple[int, ...]) -> np.ndarray:
    """Alternancia por paridad de índices (útil en SC; FCC queda frustrado)."""
    indices = np.indices(forma_sitios)
    paridad = sum(indices).astype(int) % 2
    spins = np.zeros((*forma_sitios, 3), dtype=float)
    spins[..., 2] = np.where(paridad == 0, 1.0, -1.0)
    return spins


def espines_aleatorios(forma_sitios: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    return vector_aleatorio_esfera(rng, size=forma_sitios)
