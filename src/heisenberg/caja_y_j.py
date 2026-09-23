"""Caja mínima (Tc vs L) y calibración de J para Heisenberg clásico."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..config import (
    DELTA_TC_CAJA_MAX,
    KB_EV,
    L_MAX_COMP_CENTRADAS,
    L_MIN_COMP,
    L_MIN_PARADA,
    L_PASO_COMP,
    MCS_CAJA_H,
    MCS_TERMALIZACION_H,
    N_TEMPS_FINA,
    N_TEMPS_GRUESA,
    SEED,
    l_max_comparacion,
    preparar_salidas_calibracion,
    tc_experimental_k,
)
from ..materials.estructura import etiqueta_estructura
from .base import HeisenbergClasico, crear_heisenberg


@dataclass(slots=True)
class ResultadoCaja:
    L: int
    N: int
    tc_estrella: float  # kB T / J  (adimensional)
    chi_max: float
    delta_rel_prev: float | None


@dataclass(slots=True)
class CalibracionHeisenberg:
    geometria: str
    tipo_etiqueta: str
    L_optimo: int
    N_optimo: int
    tc_estrella: float
    J_eV: float
    tc_experimental_k: float
    tc_simulada_k: float
    filas: list[ResultadoCaja]


def _malla_T(t_min: float, t_max: float, n: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, n)
    return t_min + (t_max - t_min) * (u**1.3)


def _estimar_tc_estrella(
    geometria: str,
    L: int,
    t_max_star: float,
    inicializacion: str,
    seed: int,
) -> tuple[float, float]:
    """Tc* = kB T / J con J=1, kB=1 (unidades reducidas)."""
    temps = np.unique(
        np.concatenate(
            [
                _malla_T(0.1, t_max_star, N_TEMPS_GRUESA),
                _malla_T(0.35 * t_max_star, t_max_star, N_TEMPS_FINA),
            ]
        )
    )
    chis: list[float] = []
    spins_warm = None
    for T in temps:
        modelo = crear_heisenberg(
            geometria,
            L=L,
            J=1.0,
            T=float(max(T, 0.1)),
            h=0.0,
            kB=1.0,
            inicializacion=inicializacion,
            seed=seed + int(T * 1000),
        )
        if spins_warm is not None:
            modelo.spins = spins_warm
            modelo.E = modelo._energia_total()
            modelo.m_vec = modelo.spins.reshape(-1, 3).sum(axis=0)
        modelo.simular(MCS_CAJA_H, guardar_red_cada=0)
        spins_warm = modelo.spins.copy()
        _e, m, m2, _e2 = modelo.momentos_produccion(
            min(MCS_TERMALIZACION_H, max(1, len(modelo.energy_history) // 2))
        )
        M = m * modelo.Nspin
        chi = (m2 - M**2) / (modelo.Nspin * float(T))
        chis.append(max(float(chi), 0.0))

    idx = int(np.argmax(chis))
    return float(temps[idx]), float(chis[idx])


def encontrar_caja_minima(
    geometria: str,
    *,
    inicializacion: str = "ferromagnetico",
    seed: int = SEED,
) -> tuple[list[ResultadoCaja], int]:
    """Barrido L hasta estabilizar Tc* (cambio relativo < DELTA_TC_CAJA_MAX)."""
    _q, z = HeisenbergClasico._meta_red(geometria)
    # Cota MFA clásica: Tc* ≈ 2z/3
    t_max_star = 1.15 * (2.0 * z / 3.0)
    L_max = min(l_max_comparacion(geometria), L_MAX_COMP_CENTRADAS)

    filas: list[ResultadoCaja] = []
    L_opt = L_max
    prev: float | None = None

    print(
        f"Caja mínima Heisenberg | {etiqueta_estructura(geometria)}\n"
        f"  L={L_MIN_COMP}…{L_max} | unidades reducidas (J=1, kB=1)\n"
        f"  parada si ΔTc/Tc ≤ {100*DELTA_TC_CAJA_MAX:.1f} % y L≥{L_MIN_PARADA}"
    )

    for L in range(L_MIN_COMP, L_max + 1, L_PASO_COMP):
        tc_s, chi = _estimar_tc_estrella(
            geometria, L, t_max_star, inicializacion, seed + L
        )
        q, _z = HeisenbergClasico._meta_red(geometria)
        N = L**3 * q
        delta = None if prev is None else abs(tc_s - prev) / max(prev, 1e-12)
        filas.append(
            ResultadoCaja(
                L=L, N=N, tc_estrella=tc_s, chi_max=chi, delta_rel_prev=delta
            )
        )
        marca = ""
        if delta is not None and delta <= DELTA_TC_CAJA_MAX and L >= L_MIN_PARADA:
            marca = f"  ← caja mínima (Δ={100*delta:.2f} %)"
            L_opt = L
            print(
                f"  L={L:2d} (N={N:5d})  Tc*={tc_s:.4f}  χ_max={chi:.3g}{marca}"
            )
            break
        print(
            f"  L={L:2d} (N={N:5d})  Tc*={tc_s:.4f}  χ_max={chi:.3g}"
            + (f"  Δ={100*delta:.2f} %" if delta is not None else "")
        )
        prev = tc_s
        L_opt = L

    return filas, L_opt


def calibrar_J(geometria: str, tc_estrella: float) -> tuple[float, float]:
    """J [eV] = kB[eV/K] * Tc_exp[K] / Tc*.

    Así la simulación física con ese J reproduce Tc_exp.
    """
    tc_exp = tc_experimental_k(geometria)
    if tc_exp is None:
        raise ValueError(f"No hay Tc experimental para {geometria}")
    J = KB_EV * tc_exp / tc_estrella
    tc_pred = (J / KB_EV) * tc_estrella
    return float(J), float(tc_pred)


def graficar_caja(filas: list[ResultadoCaja], ruta: Path, titulo: str) -> None:
    Ls = [f.L for f in filas]
    Tcs = [f.tc_estrella for f in filas]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(Ls, Tcs, "o-", color="#5B2C6F", label=r"$T_c^\star(L)=k_B T_c/J$")
    ax.set_xlabel(r"Lado de caja $L$ (bulk, PBC)")
    ax.set_ylabel(r"$T_c^\star$ (adimensional)")
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


def guardar_calibracion_csv(cal: CalibracionHeisenberg, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "geometria",
                "estructura",
                "L",
                "N",
                "Tc_estrella",
                "chi_max",
                "delta_rel_prev",
                "L_optimo",
                "J_eV",
                "Tc_exp_K",
                "kB_eV_per_K",
            ]
        )
        for f in cal.filas:
            w.writerow(
                [
                    cal.geometria,
                    cal.tipo_etiqueta,
                    f.L,
                    f.N,
                    f"{f.tc_estrella:.8f}",
                    f"{f.chi_max:.6g}",
                    "" if f.delta_rel_prev is None else f"{f.delta_rel_prev:.6g}",
                    cal.L_optimo,
                    f"{cal.J_eV:.8g}",
                    f"{cal.tc_experimental_k:.2f}",
                    f"{KB_EV:.12g}",
                ]
            )


def ejecutar_calibracion_heisenberg(
    geometria: str,
    *,
    inicializacion: str = "ferromagnetico",
) -> CalibracionHeisenberg:
    rutas = preparar_salidas_calibracion(geometria, "heisenberg")
    etiqueta = etiqueta_estructura(geometria)
    print(f"\n=== Calibración Heisenberg: {etiqueta} ===\n")

    # Verificar estructura MP si aplica
    from ..materials.mp_client import MATERIALES_MP
    from ..materials.estructura import analizar_estructura_mp

    if geometria in MATERIALES_MP:
        analisis = analizar_estructura_mp(geometria)
        estado = "OK" if analisis.valida else "REVISAR"
        print(f"Validación MP [{estado}]: {analisis.mensaje}")
        print(
            f"  tipo={analisis.tipo_red}  símbolo={analisis.simbolo_espacial}  "
            f"z={analisis.coordinacion}  a={analisis.a_angstrom:.4f} Å\n"
        )
        if not analisis.valida:
            raise RuntimeError(f"Estructura MP inválida: {analisis.mensaje}")

    filas, L_opt = encontrar_caja_minima(geometria, inicializacion=inicializacion)
    tc_star = next(f.tc_estrella for f in filas if f.L == L_opt)
    tc_exp = tc_experimental_k(geometria) or 0.0
    J, tc_pred = calibrar_J(geometria, tc_star)
    N_opt = next(f.N for f in filas if f.L == L_opt)

    cal = CalibracionHeisenberg(
        geometria=geometria,
        tipo_etiqueta=etiqueta,
        L_optimo=L_opt,
        N_optimo=N_opt,
        tc_estrella=tc_star,
        J_eV=J,
        tc_experimental_k=tc_exp,
        tc_simulada_k=tc_pred,
        filas=filas,
    )
    print(
        f"\nL óptimo = {L_opt} (N={N_opt})\n"
        f"Tc* (J=1,kB=1) = {tc_star:.4f}\n"
        f"Tc experimental = {tc_exp:.1f} K\n"
        f"kB = {KB_EV:.6e} eV/K\n"
        f"J calibrado = {J:.6g} eV\n"
        f"Tc predicha = {tc_pred:.1f} K\n"
    )
    graficar_caja(
        filas,
        rutas["graficas"] / "tc_vs_caja.png",
        f"Caja mínima — Heisenberg — {etiqueta}",
    )
    guardar_calibracion_csv(cal, rutas["tablas"] / "calibracion_j_caja.csv")
    (rutas["base"] / "resumen.txt").write_text(
        f"modelo: Heisenberg clásico (Metropolis, bulk PBC)\n"
        f"estructura: {etiqueta}\n"
        f"inicializacion: {inicializacion}\n"
        f"kB_eV_per_K: {KB_EV}\n"
        f"L_optimo: {L_opt}\n"
        f"N_optimo: {N_opt}\n"
        f"Tc_estrella: {tc_star}\n"
        f"J_eV: {J}\n"
        f"Tc_exp_K: {tc_exp}\n"
        f"Tc_sim_K: {tc_pred}\n",
        encoding="utf-8",
    )
    print(f"Listo → {rutas['base']}")
    return cal
