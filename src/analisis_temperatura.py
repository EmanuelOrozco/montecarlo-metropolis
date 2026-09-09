"""Barrido de temperatura: magnetización |m|(T) y estimación de Tc."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .config import (
    CASOS,
    CASO_TEMPERATURA,
    H_T,
    KB,
    L,
    MCS_T,
    MCS_TERMALIZACION,
    N_TEMPERATURAS,
    SEED,
    T_MAX,
    T_MIN,
)
from .ising import crear_ising, tc_teorica

PasoCallback = Callable[
    [int, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    None,
]


@dataclass
class ResultadoTemperatura:
    temperaturas: np.ndarray
    magnetizacion: np.ndarray
    energia: np.ndarray
    susceptibilidad: np.ndarray
    calor_especifico: np.ndarray
    redes_finales: list[np.ndarray]
    tc_estimada: float
    tc_teorica: float
    indice_tc: int
    geometria: str
    mcs_termalizacion: int


def _caso_por_clave(clave: str) -> dict:
    for caso in CASOS:
        if caso["clave"] == clave:
            return caso
    raise KeyError(f"No hay caso con clave {clave!r} en CASOS")


def _malla_temperaturas(
    t_min: float,
    t_max: float,
    n: int,
    geometria: str,
    J: float = 1.0,
) -> np.ndarray:
    """Malla densificada al subir T y alrededor de la Tc de la geometría."""
    tc = tc_teorica(geometria, J=J)
    t_min = max(float(t_min), 1e-6)
    n = max(int(n), 12)

    u = np.linspace(0.0, 1.0, n)
    base = t_min + (t_max - t_min) * (u**0.52)

    n_crit = max(n // 3, 10)
    n_alto = max(n // 4, 8)
    criticos = np.linspace(tc * 0.7, min(t_max, tc * 1.45), n_crit)
    altos = np.linspace(tc * 1.05, t_max, n_alto)
    bajos = np.geomspace(t_min, max(t_min * 20, 0.2), 5)

    temps = np.unique(
        np.round(np.concatenate([bajos, base, criticos, altos]), decimals=6)
    )
    return temps[(temps >= t_min) & (temps <= t_max)]


def estimar_tc(temperaturas: np.ndarray, susceptibilidad: np.ndarray) -> tuple[float, int]:
    """Tc numérica: temperatura donde la susceptibilidad χ es máxima."""
    idx = int(np.argmax(susceptibilidad))
    return float(temperaturas[idx]), idx


def barrido_temperatura(
    geometria: str = "cuadrada",
    t_min: float = T_MIN,
    t_max: float = T_MAX,
    n_temps: int = N_TEMPERATURAS,
    mcs: int = MCS_T,
    mcs_term: int = MCS_TERMALIZACION,
    clave_caso: str = CASO_TEMPERATURA,
    L_red: int = L,
    h: float = H_T,
    seed: int = SEED,
    on_paso: PasoCallback | None = None,
) -> ResultadoTemperatura:
    caso = _caso_por_clave(clave_caso)
    temps = _malla_temperaturas(t_min, t_max, n_temps, geometria, J=caso["J"])
    n_term = int(np.clip(mcs_term, 1, max(mcs - 1, 1)))
    tc_teo = tc_teorica(geometria, J=caso["J"], kB=KB)

    mags: list[float] = []
    energias: list[float] = []
    chis: list[float] = []
    ces: list[float] = []
    redes: list[np.ndarray] = []

    print(
        f"Barrido T ∈ [{temps[0]:.4g}, {temps[-1]:.4g}]  "
        f"({len(temps)} puntos)  |  MCS={mcs}  |  "
        f"promedio MCS≥{n_term}  |  red={geometria}  |  caso={caso['titulo']}"
    )

    for i, T in enumerate(temps):
        modelo = crear_ising(
            geometria,
            L=L_red,
            J=caso["J"],
            h=h,
            T=float(T),
            kB=KB,
            inicializacion=caso["inicializacion"],
            seed=seed + i,
        )
        modelo.simular(mcs, guardar_red_cada=mcs + 1)

        m_hist = np.asarray(modelo.magnetization_history[n_term:], dtype=float)
        e_hist = np.asarray(modelo.energy_history[n_term:], dtype=float)
        nspin = modelo.Nspin

        # Promedios de producción: se descartan los primeros n_term MCS.
        m_media = float(np.mean(np.abs(m_hist)) / nspin)
        e_media = float(np.mean(e_hist) / nspin)
        chi = float((np.mean(m_hist**2) - np.mean(m_hist) ** 2) / (T * nspin))
        cv = float((np.mean(e_hist**2) - np.mean(e_hist) ** 2) / (T * T * nspin))
        red = modelo.spins.copy()

        mags.append(m_media)
        energias.append(e_media)
        chis.append(chi)
        ces.append(cv)
        redes.append(red)

        temps_acum = np.asarray(temps[: i + 1], dtype=float)
        mags_acum = np.asarray(mags, dtype=float)
        chis_acum = np.asarray(chis, dtype=float)

        marca = "  ← cerca de Tc" if abs(T - tc_teo) < 0.2 else ""
        print(
            f"  [{i + 1:02d}/{len(temps)}]  T={T:7.4f}  "
            f"|m|/N={m_media:7.4f}  χ={chi:8.4f}{marca}"
        )

        if on_paso is not None:
            on_paso(i, float(T), m_media, chi, red, temps_acum, mags_acum, chis_acum)

    temps_arr = np.asarray(temps, dtype=float)
    chi_arr = np.asarray(chis, dtype=float)
    tc_est, idx_tc = estimar_tc(temps_arr, chi_arr)

    return ResultadoTemperatura(
        temperaturas=temps_arr,
        magnetizacion=np.asarray(mags, dtype=float),
        energia=np.asarray(energias, dtype=float),
        susceptibilidad=chi_arr,
        calor_especifico=np.asarray(ces, dtype=float),
        redes_finales=redes,
        tc_estimada=tc_est,
        tc_teorica=tc_teo,
        indice_tc=idx_tc,
        geometria=geometria,
        mcs_termalizacion=n_term,
    )
