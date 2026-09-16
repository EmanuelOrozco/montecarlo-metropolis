"""Topologías cristalográficas cúbicas con fronteras periódicas.

Las coordenadas se expresan en unidades de la constante de red de la celda
convencional. BCC usa dos sitios por celda y FCC cuatro.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np


@dataclass(frozen=True, slots=True)
class EspecificacionCubica:
    clave: str
    nombre: str
    base: tuple[tuple[float, float, float], ...]
    distancia_vecino2: float
    coordinacion: int
    tc_referencia: float

    @property
    def sitios_por_celda(self) -> int:
        return len(self.base)


ESPECIFICACIONES_CUBICAS = {
    "cubica": EspecificacionCubica(
        clave="cubica",
        nombre="Cúbica simple",
        base=((0.0, 0.0, 0.0),),
        distancia_vecino2=1.0,
        coordinacion=6,
        tc_referencia=4.511524,
    ),
    "bcc": EspecificacionCubica(
        clave="bcc",
        nombre="Cúbica centrada en el cuerpo (BCC)",
        base=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
        distancia_vecino2=0.75,
        coordinacion=8,
        tc_referencia=6.3558,
    ),
    "fcc": EspecificacionCubica(
        clave="fcc",
        nombre="Cúbica centrada en las caras (FCC)",
        base=(
            (0.0, 0.0, 0.0),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
        ),
        distancia_vecino2=0.5,
        coordinacion=12,
        tc_referencia=9.794,
    ),
}


def especificacion_cubica(geometria: str) -> EspecificacionCubica:
    try:
        return ESPECIFICACIONES_CUBICAS[geometria]
    except KeyError as exc:
        opciones = tuple(ESPECIFICACIONES_CUBICAS)
        raise ValueError(
            f"Geometría cúbica desconocida {geometria!r}; use una de {opciones}"
        ) from exc


@lru_cache(maxsize=None)
def reglas_vecindad(
    geometria: str,
) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
    """Reglas (dx, dy, dz, base_destino) para cada sitio de la base."""
    spec = especificacion_cubica(geometria)
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
                    atol=1e-12,
                ):
                    vecinos.append((*desplazamiento, destino))

        if len(vecinos) != spec.coordinacion:
            raise RuntimeError(
                f"{geometria}: base {origen} tiene {len(vecinos)} vecinos; "
                f"se esperaban {spec.coordinacion}"
            )
        reglas.append(tuple(vecinos))

    return tuple(reglas)


@lru_cache(maxsize=32)
def tabla_vecinos_cubica(geometria: str, L: int) -> np.ndarray:
    """Tabla de vecinos para L³ celdas convencionales."""
    spec = especificacion_cubica(geometria)
    q = spec.sitios_por_celda
    forma = (L, L, L, q)
    n = L**3 * q
    tabla = np.empty((n, spec.coordinacion), dtype=np.int32)
    reglas = reglas_vecindad(geometria)

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
def coordenadas_cubicas(
    geometria: str, L: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordenadas cartesianas en el mismo orden que `spins.ravel()`."""
    spec = especificacion_cubica(geometria)
    base = np.asarray(spec.base, dtype=float)
    coordenadas = np.empty((L**3 * spec.sitios_por_celda, 3), dtype=float)
    p = 0
    for i, j, k in np.ndindex((L, L, L)):
        for posicion in base:
            coordenadas[p] = (i, j, k) + posicion
            p += 1
    return (
        coordenadas[:, 0],
        coordenadas[:, 1],
        coordenadas[:, 2],
    )
