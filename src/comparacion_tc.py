"""Comparación de Tc estimada vs teórica al variar el tamaño de la celda.

Patrones de optimización:
- Flyweight / LRU: tabla de vecinos reutilizada por (geometría, L).
- Lookup de Metropolis: exp(-βΔE) precalculado para los ΔE discretos.
- Warm start: el barrido en T reutiliza la red ya termalizada.
- Coarse-to-fine: malla gruesa y luego refinamiento alrededor del pico de χ.
- Acumuladores: momentos de m sin guardar historiales ni copias de red.
- Early stopping: se detiene el crecimiento de L si el error relativo ≤ 2 %.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import Generator, default_rng

from .analisis_temperatura import estimar_tc
from .config import (
    ERROR_REL_MAX,
    H_T,
    J_FERRO,
    KB,
    L_MAX_COMP,
    L_MIN_COMP,
    L_MIN_PARADA,
    L_PASO_COMP,
    MCS_TERMALIZACION,
    N_TEMPS_FINA,
    N_TEMPS_GRUESA,
    SEED,
    preparar_salidas_tamano,
)
from .ising import etiqueta_red, tc_teorica

_VECINOS_CUADRADA = ((-1, 0), (1, 0), (0, -1), (0, 1))
_VECINOS_TRIANGULAR = (
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (-1, 1),
    (1, -1),
)


@dataclass(slots=True)
class ResultadoTamano:
    L: int
    N: int
    tc_estimada: float
    tc_teorica: float
    error_relativo: float
    chi_max: float
    n_temperaturas: int


@dataclass(slots=True)
class ComparacionTc:
    geometria: str
    tc_teorica: float
    filas: list[ResultadoTamano] = field(default_factory=list)
    convergida: bool = False
    L_convergencia: int | None = None


def error_relativo(tc_est: float, tc_teo: float) -> float:
    return abs(tc_est - tc_teo) / abs(tc_teo) if tc_teo != 0.0 else math.inf


def tamanos_celda(
    l_min: int = L_MIN_COMP,
    l_max: int = L_MAX_COMP,
    paso: int = L_PASO_COMP,
) -> tuple[int, ...]:
    return tuple(range(int(l_min), int(l_max) + 1, int(paso)))


@lru_cache(maxsize=32)
def tabla_vecinos(geometria: str, L: int) -> np.ndarray:
    """Flyweight: índices de vecinos con PBC, forma (N, z)."""
    offsets = _VECINOS_CUADRADA if geometria == "cuadrada" else _VECINOS_TRIANGULAR
    n = L * L
    z = len(offsets)
    nb = np.empty((n, z), dtype=np.int32)
    for i in range(L):
        for j in range(L):
            k = i * L + j
            for p, (di, dj) in enumerate(offsets):
                nb[k, p] = ((i + di) % L) * L + ((j + dj) % L)
    return nb


def _tabla_aceptacion(beta: float, J: float, h: float, z: int) -> dict[tuple[int, int], float]:
    """P(aceptar | s, suma de vecinos) para todos los ΔE discretos."""
    tabla: dict[tuple[int, int], float] = {}
    for s in (-1, 1):
        for nn in range(-z, z + 1, 2):
            dE = 2.0 * (J * nn + h) * s
            tabla[(s, nn)] = 1.0 if dE <= 0.0 else math.exp(-beta * dE)
    return tabla


def _mcs_para_L(L: int) -> tuple[int, int]:
    n_term = min(MCS_TERMALIZACION, max(40, 8 * L))
    n_prod = max(300, 28 * L)
    return n_term, n_term + n_prod


def _pico_parabolico(temperaturas: np.ndarray, chi: np.ndarray) -> tuple[float, int]:
    """Refina el argmax de χ con una parábola en tres puntos."""
    idx = int(np.argmax(chi))
    if idx == 0 or idx == len(temperaturas) - 1:
        return float(temperaturas[idx]), idx
    x1, x2, x3 = (float(temperaturas[idx + k]) for k in (-1, 0, 1))
    y1, y2, y3 = (float(chi[idx + k]) for k in (-1, 0, 1))
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if abs(denom) < 1e-18:
        return x2, idx
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    if a >= 0.0:
        return x2, idx
    x_star = -b / (2.0 * a)
    if x_star < min(x1, x3) or x_star > max(x1, x3):
        return x2, idx
    return float(x_star), idx


def _metropolis_momentos(
    spins: np.ndarray,
    vecinos: np.ndarray,
    J: float,
    h: float,
    T: float,
    mcs: int,
    n_term: int,
    rng: Generator,
) -> tuple[float, float, np.ndarray]:
    """Un barrido Metropolis; acumula ⟨|m|⟩ y χ sin historial."""
    s = spins
    n = s.size
    z = vecinos.shape[1]
    beta = 1.0 / max(T, 1e-12)
    aceptar = _tabla_aceptacion(beta, J, h, z)
    m = int(s.sum())
    acc_abs = 0.0
    acc_m = 0.0
    acc_m2 = 0.0
    n_prod = 0

    for paso in range(mcs):
        idxs = rng.integers(0, n, size=n, dtype=np.int32)
        us = rng.random(n)
        for k in range(n):
            i = int(idxs[k])
            old = int(s[i])
            nn = int(s[vecinos[i]].sum())
            if us[k] < aceptar[(old, nn)]:
                s[i] = -old
                m -= 2 * old
        if paso >= n_term:
            acc_abs += abs(m)
            acc_m += m
            acc_m2 += m * m
            n_prod += 1

    n_prod = max(n_prod, 1)
    mean_m = acc_m / n_prod
    chi = (acc_m2 / n_prod - mean_m * mean_m) / (T * n)
    mag = (acc_abs / n_prod) / n
    return mag, chi, s


def _barrido_chi(
    spins: np.ndarray,
    vecinos: np.ndarray,
    temps: np.ndarray,
    J: float,
    h: float,
    mcs: int,
    n_term: int,
    rng: Generator,
    reutilizar: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chis = np.empty(len(temps), dtype=float)
    mags = np.empty(len(temps), dtype=float)
    estado = spins
    for i, T in enumerate(temps):
        if not reutilizar:
            estado = np.ones_like(spins)
        mag, chi, estado = _metropolis_momentos(
            estado, vecinos, J, h, float(T), mcs, n_term, rng
        )
        mags[i] = mag
        chis[i] = chi
    return mags, chis, estado


def estimar_tc_para_L(
    L: int,
    geometria: str,
    J: float = J_FERRO,
    h: float = H_T,
    seed: int = SEED,
) -> ResultadoTamano:
    """Tc(L) con búsqueda gruesa y refinamiento alrededor del máximo de χ."""
    tc_teo = tc_teorica(geometria, J=J, kB=KB)
    vecinos = tabla_vecinos(geometria, L)
    n_term, mcs = _mcs_para_L(L)
    rng = default_rng(seed + 1000 * L)
    spins = np.ones(L * L, dtype=np.int8)

    t_lo = max(0.15, tc_teo * 0.62)
    t_hi = tc_teo * 1.55
    temps_g = np.linspace(t_lo, t_hi, N_TEMPS_GRUESA)
    _mags_g, chis_g, _spins = _barrido_chi(
        spins, vecinos, temps_g, J, h, mcs, n_term, rng, reutilizar=True
    )
    _tc_g, idx_g = estimar_tc(temps_g, chis_g)

    ancho = max(0.12, 0.55 / max(L, 2))
    t_c0 = float(temps_g[idx_g])
    temps_f = np.unique(
        np.clip(
            np.linspace(t_c0 - ancho, t_c0 + ancho, N_TEMPS_FINA),
            t_lo,
            t_hi,
        )
    )
    spins_fino = np.ones(L * L, dtype=np.int8)
    _mags_f, chis_f, _ = _barrido_chi(
        spins_fino, vecinos, temps_f, J, h, mcs, n_term, rng, reutilizar=True
    )

    fusion: dict[float, float] = {}
    for t, c in zip(temps_g, chis_g):
        fusion[round(float(t), 6)] = float(c)
    for t, c in zip(temps_f, chis_f):
        fusion[round(float(t), 6)] = float(c)
    claves = sorted(fusion)
    temps_u = np.array(claves, dtype=float)
    chis_u = np.array([fusion[k] for k in claves], dtype=float)

    tc_est, _idx = _pico_parabolico(temps_u, chis_u)
    chi_max = float(np.max(chis_u))
    return ResultadoTamano(
        L=L,
        N=L * L,
        tc_estimada=tc_est,
        tc_teorica=tc_teo,
        error_relativo=error_relativo(tc_est, tc_teo),
        chi_max=chi_max,
        n_temperaturas=int(len(temps_u)),
    )


class ComparadorTamano:
    """Orquesta L = 2, 4, … y corta si el error relativo ≤ 2 %."""

    def __init__(
        self,
        geometria: str,
        l_min: int = L_MIN_COMP,
        l_max: int = L_MAX_COMP,
        paso: int = L_PASO_COMP,
        error_max: float = ERROR_REL_MAX,
        l_min_parada: int = L_MIN_PARADA,
        seed: int = SEED,
    ) -> None:
        self.geometria = geometria
        self.l_min = l_min
        self.l_max = l_max
        self.paso = paso
        self.error_max = error_max
        self.l_min_parada = l_min_parada
        self.seed = seed

    def ejecutar(self) -> ComparacionTc:
        tc_teo = tc_teorica(self.geometria, J=J_FERRO, kB=KB)
        out = ComparacionTc(geometria=self.geometria, tc_teorica=tc_teo)
        print(
            f"Comparación Tc vs L  |  {etiqueta_red(self.geometria)}\n"
            f"  L = {self.l_min} … {self.l_max} (paso {self.paso})  |  "
            f"parada si error ≤ {100 * self.error_max:.1f} % y L ≥ {self.l_min_parada}\n"
            f"  Tc teórica = {tc_teo:.6f}"
        )
        for L in tamanos_celda(self.l_min, self.l_max, self.paso):
            fila = estimar_tc_para_L(L, self.geometria, seed=self.seed)
            out.filas.append(fila)
            marca = "  ← ≤ 2 %" if fila.error_relativo <= self.error_max else ""
            print(
                f"  L={L:2d} ({fila.N:4d} espines)  "
                f"Tc_est={fila.tc_estimada:.4f}  "
                f"error={100 * fila.error_relativo:6.2f} %  "
                f"χ_max={fila.chi_max:.3f}{marca}"
            )
            if L >= self.l_min_parada and fila.error_relativo <= self.error_max:
                out.convergida = True
                out.L_convergencia = L
                print(f"  Criterio del 2 % alcanzado en L={L}. Se detiene el barrido.")
                break
        if not out.convergida:
            print(
                f"  No se alcanzó error ≤ {100 * self.error_max:.1f} % "
                f"hasta L={self.l_max} (efecto de tamaño finito)."
            )
        return out


def guardar_tabla(comp: ComparacionTc, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["L", "N", "Tc_estimada", "Tc_teorica", "error_relativo", "error_porcentaje", "chi_max"]
        )
        for f in comp.filas:
            writer.writerow(
                [
                    f.L,
                    f.N,
                    f"{f.tc_estimada:.8f}",
                    f"{f.tc_teorica:.8f}",
                    f"{f.error_relativo:.8f}",
                    f"{100 * f.error_relativo:.4f}",
                    f"{f.chi_max:.8f}",
                ]
            )


def graficar_comparacion(comp: ComparacionTc, ruta: Path) -> None:
    Ls = np.array([f.L for f in comp.filas], dtype=float)
    tcs = np.array([f.tc_estimada for f in comp.filas], dtype=float)
    errs = 100.0 * np.array([f.error_relativo for f in comp.filas], dtype=float)
    teo = comp.tc_teorica

    fig, (ax_t, ax_e) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_t.plot(Ls, tcs, "o-", color="#8E44AD", lw=1.8, ms=7, label=r"$T_c(L)$ estimada")
    ax_t.axhline(teo, color="#27AE60", ls="--", lw=1.5, label=rf"$T_c$ teórica = {teo:.4f}")
    ax_t.set_ylabel(r"$T_c$")
    ax_t.set_title(f"Temperatura crítica vs tamaño de celda — {etiqueta_red(comp.geometria)}")
    ax_t.legend(loc="best", fontsize=9)
    ax_t.grid(True, alpha=0.35)

    ax_e.plot(Ls, errs, "s-", color="#C0392B", lw=1.8, ms=6, label=r"error relativo")
    ax_e.axhline(100 * ERROR_REL_MAX, color="#E67E22", ls="-.", lw=1.4, label="umbral 2 %")
    ax_e.set_xlabel(r"Lado de la celda  $L$  ($N=L\times L$)")
    ax_e.set_ylabel(r"Error relativo  $|T_c(L)-T_c|/T_c$  [%]")
    ax_e.legend(loc="best", fontsize=9)
    ax_e.grid(True, alpha=0.35)

    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def ejecutar_comparacion_tamano(geometria: str) -> ComparacionTc:
    rutas = preparar_salidas_tamano(geometria)
    print(f"Salida → {rutas['base']}\n")
    comparador = ComparadorTamano(geometria)
    resultado = comparador.ejecutar()

    ruta_csv = rutas["tablas"] / "tc_vs_L.csv"
    ruta_fig = rutas["graficas"] / "tc_vs_tamano.png"
    guardar_tabla(resultado, ruta_csv)
    graficar_comparacion(resultado, ruta_fig)
    print(f"Tabla: {ruta_csv}")
    print(f"Gráfica: {ruta_fig}")
    print(f"Listo → {rutas['base']}")
    return resultado
