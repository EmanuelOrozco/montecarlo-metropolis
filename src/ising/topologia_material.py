"""Topología Ising construida desde estructuras de Materials Project."""

from __future__ import annotations

from functools import lru_cache
from itertools import product

import numpy as np

from ..materials.mp_client import (
    asegurar_estructura,
    material_id_de_geometria,
)
from .topologia_cubica import EspecificacionCubica

# Referencias numéricas del Ising 3D con la misma coordinación cristalina.
_TC_POR_COORDINACION = {
    6: 4.511524,  # cúbica simple
    8: 6.3558,  # BCC (Fe mp-13)
    12: 9.794,  # FCC
}


def _distancias_fraccionales(
    base: np.ndarray,
) -> tuple[float, int]:
    """Distancia² NN (en coordenadas de celda) y coordinación por sitio."""
    origen = base[0]
    candidatos: list[float] = []
    for destino in base:
        for desplazamiento in product((-1, 0, 1), repeat=3):
            if np.allclose(destino, origen) and desplazamiento == (0, 0, 0):
                continue
            vector = np.asarray(desplazamiento, dtype=float) + destino - origen
            candidatos.append(float(np.dot(vector, vector)))
    d2 = min(candidatos)
    coordinacion = sum(1 for v in candidatos if abs(v - d2) < 1e-10)
    return d2, coordinacion


@lru_cache(maxsize=None)
def especificacion_material(geometria: str) -> EspecificacionCubica:
    material_id = material_id_de_geometria(geometria)
    data = asegurar_estructura(material_id)
    base = tuple(tuple(map(float, coords)) for coords in data["frac_coords"])
    d2, z = _distancias_fraccionales(np.asarray(base, dtype=float))
    tc = _TC_POR_COORDINACION.get(z)
    if tc is None:
        raise RuntimeError(
            f"{geometria}: coordinación {z} sin Tc de referencia conocida"
        )
    nombre = (
        f"{data.get('formula_pretty') or data['formula']} "
        f"({material_id}, 3D, {z} vecinos)"
    )
    return EspecificacionCubica(
        clave=geometria,
        nombre=nombre,
        base=base,
        distancia_vecino2=d2,
        coordinacion=z,
        tc_referencia=tc,
    )


@lru_cache(maxsize=None)
def reglas_vecindad_material(
    geometria: str,
) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
    spec = especificacion_material(geometria)
    base = np.asarray(spec.base, dtype=float)
    reglas: list[tuple[tuple[int, int, int, int], ...]] = []
    for origen, posicion_origen in enumerate(base):
        vecinos: list[tuple[int, int, int, int]] = []
        for destino, posicion_destino in enumerate(base):
            for desplazamiento in product((-1, 0, 1), repeat=3):
                if origen == destino and desplazamiento == (0, 0, 0):
                    continue
                vector = (
                    np.asarray(desplazamiento, dtype=float)
                    + posicion_destino
                    - posicion_origen
                )
                if np.isclose(
                    float(np.dot(vector, vector)),
                    spec.distancia_vecino2,
                    atol=1e-10,
                ):
                    vecinos.append((*desplazamiento, destino))
        if len(vecinos) != spec.coordinacion:
            raise RuntimeError(
                f"{geometria}: sitio {origen} tiene {len(vecinos)} vecinos; "
                f"se esperaban {spec.coordinacion}"
            )
        reglas.append(tuple(vecinos))
    return tuple(reglas)


@lru_cache(maxsize=32)
def tabla_vecinos_material(geometria: str, L: int) -> np.ndarray:
    spec = especificacion_material(geometria)
    q = spec.sitios_por_celda
    forma = (L, L, L, q)
    n = L**3 * q
    tabla = np.empty((n, spec.coordinacion), dtype=np.int32)
    reglas = reglas_vecindad_material(geometria)
    for i, j, k, base_origen in np.ndindex(forma):
        indice = np.ravel_multi_index((i, j, k, base_origen), forma)
        for p, (di, dj, dk, base_destino) in enumerate(reglas[base_origen]):
            vecino = (
                (i + di) % L,
                (j + dj) % L,
                (k + dk) % L,
                base_destino,
            )
            tabla[indice, p] = np.ravel_multi_index(vecino, forma)
    return tabla


@lru_cache(maxsize=32)
def coordenadas_material(
    geometria: str, L: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordenadas en unidades de celda convencional (mismo orden que spins)."""
    spec = especificacion_material(geometria)
    base = np.asarray(spec.base, dtype=float)
    coordenadas = np.empty((L**3 * spec.sitios_por_celda, 3), dtype=float)
    p = 0
    for i, j, k in np.ndindex((L, L, L)):
        for posicion in base:
            coordenadas[p] = (i, j, k) + posicion
            p += 1
    return coordenadas[:, 0], coordenadas[:, 1], coordenadas[:, 2]


def metadatos_material(geometria: str) -> dict:
    return asegurar_estructura(material_id_de_geometria(geometria))
