"""Barridos de temperatura y ejecución Heisenberg clásico."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from ..config import (
    H,
    KB_EV,
    MCS,
    MCS_T_H,
    MCS_TERMALIZACION_H,
    N_TEMPERATURAS_H,
    SEED,
    T,
    T_MIN_HEISENBERG_K,
    preparar_salidas_t_fija,
    preparar_salidas_vs_t,
    tc_experimental_k,
)
from ..materials.estructura import etiqueta_estructura
from .base import HeisenbergClasico, crear_heisenberg
from .caja_y_j import CalibracionHeisenberg, ejecutar_calibracion_heisenberg


@dataclass(slots=True)
class ResultadoTHeisenberg:
    temperaturas: np.ndarray
    magnetizacion: np.ndarray
    energia: np.ndarray
    susceptibilidad: np.ndarray
    tc_estimada: float
    tc_referencia: float


def _malla_T(t_min: float, t_max: float, n: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, n)
    return t_min + (t_max - t_min) * (u**1.25)


def barrido_temperatura_heisenberg(
    geometria: str,
    L: int,
    J: float,
    t_max_k: float,
    *,
    inicializacion: str = "ferromagnetico",
    seed: int = SEED,
) -> ResultadoTHeisenberg:
    temps = _malla_T(T_MIN_HEISENBERG_K, t_max_k, N_TEMPERATURAS_H)
    mags: list[float] = []
    enes: list[float] = []
    chis: list[float] = []
    etiqueta = etiqueta_estructura(geometria)
    print(
        f"Barrido Heisenberg T∈[{T_MIN_HEISENBERG_K}, {t_max_k:.1f}] K | "
        f"L={L} | J={J:.5g} eV | {etiqueta} | init={inicializacion}"
    )
    for i, T_paso in enumerate(temps, start=1):
        modelo = crear_heisenberg(
            geometria,
            L=L,
            J=J,
            T=float(T_paso),
            h=0.0,
            kB=KB_EV,
            inicializacion=inicializacion,
            seed=seed + i,
        )
        modelo.simular(MCS_T_H, guardar_red_cada=0)
        e, m, m2, _ = modelo.momentos_produccion(
            min(MCS_TERMALIZACION_H, max(1, len(modelo.energy_history) // 3))
        )
        M = m * modelo.Nspin
        chi = (m2 - M**2) / (modelo.Nspin * KB_EV * float(T_paso))
        mags.append(m)
        enes.append(e / modelo.Nspin)
        chis.append(max(chi, 0.0))
        marca = "  ← cerca Tc" if chi == max(chis) and i > 5 else ""
        print(
            f"  [{i:02d}/{len(temps)}] T={T_paso:8.2f} K  |m|={m:.4f}  "
            f"E/N={e/modelo.Nspin:.4f}  χ={chi:.4g}{marca}"
        )

    chi_arr = np.asarray(chis)
    idx = int(np.argmax(chi_arr))
    tc_est = float(temps[idx])
    tc_ref = float(tc_experimental_k(geometria) or tc_est)
    return ResultadoTHeisenberg(
        temperaturas=temps,
        magnetizacion=np.asarray(mags),
        energia=np.asarray(enes),
        susceptibilidad=chi_arr,
        tc_estimada=tc_est,
        tc_referencia=tc_ref,
    )


def _graficar_m_vs_t(res: ResultadoTHeisenberg, ruta, titulo: str) -> None:
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax0.plot(res.temperaturas, res.magnetizacion, "o-", color="#C0392B", label=r"$\langle|m|\rangle$")
    ax0.axvline(res.tc_referencia, color="#1E8449", ls="--", label=rf"$T_C$ exp ≈ {res.tc_referencia:.0f} K")
    ax0.axvline(res.tc_estimada, color="#6C3483", ls=":", label=rf"$T_c$ est ≈ {res.tc_estimada:.1f} K")
    ax0.set_ylabel(r"Magnetización $\langle|m|\rangle$")
    ax0.set_title(titulo)
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=8)

    ax1.plot(res.temperaturas, res.susceptibilidad, "s-", color="#2471A3", label=r"$\chi$")
    ax1.axvline(res.tc_referencia, color="#1E8449", ls="--")
    ax1.axvline(res.tc_estimada, color="#6C3483", ls=":")
    ax1.set_xlabel(r"Temperatura $T$ [K]")
    ax1.set_ylabel(r"Susceptibilidad $\chi$")
    ax1.set_title(r"Susceptibilidad (pico $\approx T_c$)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


def _graficar_e_vs_t(res: ResultadoTHeisenberg, ruta, titulo: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(res.temperaturas, res.energia, "o-", color="#1A5276")
    ax.axvline(res.tc_referencia, color="#1E8449", ls="--", label="Tc exp")
    ax.axvline(res.tc_estimada, color="#6C3483", ls=":", label="Tc est")
    ax.set_xlabel(r"$T$ [K]")
    ax.set_ylabel(r"$\langle E/N\rangle$ [eV]")
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


def _graficar_mcs(modelo: HeisenbergClasico, rutas, titulo: str) -> None:
    pasos = np.arange(len(modelo.energy_history))
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax0.plot(pasos, np.asarray(modelo.energy_history) / modelo.Nspin, color="#1A5276")
    ax0.set_ylabel(r"$E/N$ [eV]")
    ax0.set_title(titulo)
    ax0.grid(True, alpha=0.3)
    ax1.plot(pasos, modelo.magnetization_history, color="#C0392B")
    ax1.set_xlabel("MCS")
    ax1.set_ylabel(r"$|m|$")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(rutas["graficas"] / "energia_magnetizacion_vs_mcs.png", dpi=140)
    plt.close(fig)


def ejecutar_heisenberg_completo(
    geometria: str,
    *,
    inicializacion: str = "ferromagnetico",
    modo: str = "todo",
) -> CalibracionHeisenberg:
    """Calibración (caja+J) y, según modo, T fija y/o |m|(T)."""
    cal = ejecutar_calibracion_heisenberg(geometria, inicializacion=inicializacion)
    etiqueta = cal.tipo_etiqueta

    if modo in ("actual", "todo"):
        print("\n--- Heisenberg temperatura fija ---\n")
        rutas = preparar_salidas_t_fija(geometria, "heisenberg")
        T_fija = max(T_MIN_HEISENBERG_K, 0.3 * cal.tc_experimental_k)
        modelo = crear_heisenberg(
            geometria,
            L=cal.L_optimo,
            J=cal.J_eV,
            T=T_fija,
            h=H * cal.J_eV,
            kB=KB_EV,
            inicializacion=inicializacion,
            seed=SEED,
        )
        print(
            f"{etiqueta} | L={cal.L_optimo} | J={cal.J_eV:.5g} eV | "
            f"T={T_fija:.1f} K | init={inicializacion}"
        )
        modelo.simular(MCS, guardar_red_cada=0)
        _graficar_mcs(
            modelo,
            rutas,
            f"Heisenberg T fija — {etiqueta} — {inicializacion}",
        )
        (rutas["base"] / "resumen.txt").write_text(
            f"modelo: Heisenberg clásico\n"
            f"estructura: {etiqueta}\n"
            f"inicializacion: {inicializacion}\n"
            f"L: {cal.L_optimo}\n"
            f"J_eV: {cal.J_eV}\n"
            f"T_K: {T_fija}\n"
            f"kB_eV_per_K: {KB_EV}\n"
            f"E_final/N: {modelo.E/modelo.Nspin}\n"
            f"|m|_final: {modelo.magnetizacion_por_sitio()}\n",
            encoding="utf-8",
        )
        print(f"Listo → {rutas['base']}")

    if modo in ("temperatura", "todo", "tamano"):
        print("\n--- Heisenberg magnetización vs T ---\n")
        rutas = preparar_salidas_vs_t(geometria, "heisenberg")
        t_max = 1.4 * cal.tc_experimental_k
        res = barrido_temperatura_heisenberg(
            geometria,
            L=cal.L_optimo,
            J=cal.J_eV,
            t_max_k=t_max,
            inicializacion=inicializacion,
        )
        _graficar_m_vs_t(
            res,
            rutas["graficas"] / "magnetizacion_vs_temperatura.png",
            f"Heisenberg |m|(T) — {etiqueta} — {inicializacion}",
        )
        _graficar_e_vs_t(
            res,
            rutas["graficas"] / "energia_vs_temperatura.png",
            f"Heisenberg E(T) — {etiqueta} — {inicializacion}",
        )
        (rutas["base"] / "resumen.txt").write_text(
            f"modelo: Heisenberg clásico\n"
            f"estructura: {etiqueta}\n"
            f"inicializacion: {inicializacion}\n"
            f"L_optimo: {cal.L_optimo}\n"
            f"J_eV: {cal.J_eV}\n"
            f"Tc_estimada_K: {res.tc_estimada}\n"
            f"Tc_experimental_K: {res.tc_referencia}\n"
            f"T_min_K: {T_MIN_HEISENBERG_K}\n"
            f"kB_eV_per_K: {KB_EV}\n",
            encoding="utf-8",
        )
        print(
            f"Tc est={res.tc_estimada:.1f} K | Tc exp={res.tc_referencia:.1f} K\n"
            f"Listo → {rutas['base']}"
        )

    return cal
