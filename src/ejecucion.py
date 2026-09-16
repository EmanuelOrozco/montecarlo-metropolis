"""Pipelines de ejecución: temperatura fija y barrido vs T, por geometría."""

from __future__ import annotations

import numpy as np

from .analisis_temperatura import _malla_temperaturas, barrido_temperatura
from .comparacion_tc import ejecutar_comparacion_tamano
from .config import (
    CASOS,
    H,
    KB,
    MCS,
    MCS_TERMALIZACION,
    N_TEMPERATURAS,
    SEED,
    T,
    T_MIN,
    VIDEO_T_FPS,
    lado_geometria,
    preparar_salidas_t_fija,
    preparar_salidas_vs_t,
    t_max_geometria,
)
from .ising import crear_ising, dimension_red, etiqueta_red, tc_teorica
from .visualizacion import (
    GrabadorVideoTemperatura,
    crear_video,
    guardar_instantanea,
    graficar_energia,
    graficar_energia_vs_t,
    graficar_magnetizacion,
    graficar_magnetizacion_vs_t,
)


def ejecutar_temperatura_fija(geometria: str) -> None:
    """T fija: tres inicializaciones, gráficas E/m vs MCS y video."""
    rutas = preparar_salidas_t_fija(geometria)
    stride_video = max(1, MCS // 250)
    etiqueta = etiqueta_red(geometria)
    lado = lado_geometria(geometria)
    forma = "×".join([str(lado)] * dimension_red(geometria))

    print(f"{etiqueta}  |  {forma}  |  MCS={MCS}  |  T={T}  |  h={H}")
    print(f"Salida → {rutas['base']}")
    print(f"Guardando red cada {stride_video} paso(s) para el video.\n")

    modelos = []
    for i, caso in enumerate(CASOS):
        modelo = crear_ising(
            geometria,
            L=lado,
            J=caso["J"],
            h=H,
            T=T,
            kB=KB,
            inicializacion=caso["inicializacion"],
            seed=SEED + i,
        )
        guardar_instantanea(
            modelo.spins,
            f"{caso['titulo']} — inicial ({geometria})",
            rutas["redes"] / f"red_{caso['clave']}_inicial.png",
            geometria=geometria,
        )
        print(f"Simulando: {caso['titulo']}  (J={caso['J']:+.1f}) ...")
        modelo.simular(MCS, guardar_red_cada=stride_video)
        guardar_instantanea(
            modelo.spins,
            f"{caso['titulo']} — final MCS={MCS} ({geometria})",
            rutas["redes"] / f"red_{caso['clave']}_final.png",
            geometria=geometria,
        )
        n = modelo.Nspin
        n_term = min(MCS_TERMALIZACION, len(modelo.energy_history) - 1)
        e_prod = float(np.mean(modelo.energy_history[n_term:]) / n)
        m_prod = float(np.mean(modelo.magnetization_history[n_term:]) / n)
        print(
            f"  ⟨E/N⟩ (MCS≥{n_term}) = {e_prod:.4f}   "
            f"⟨m/N⟩ (MCS≥{n_term}) = {m_prod:.4f}   "
            f"E_final/N = {modelo.E / n:.4f}   m_final/N = {modelo.m / n:.4f}"
        )
        modelos.append((caso["titulo"], modelo))

    print("\nGenerando gráficas...")
    graficar_energia(
        modelos,
        rutas["graficas"] / "energia_vs_mcs.png",
        MCS,
        titulo_extra=etiqueta,
    )
    graficar_magnetizacion(
        modelos,
        rutas["graficas"] / "magnetizacion_vs_mcs.png",
        MCS,
        titulo_extra=etiqueta,
    )

    print("Generando video...")
    video = crear_video(
        modelos,
        rutas["video"] / "evolucion_espines.mp4",
        stride=stride_video,
        fps=20,
        titulo_extra=etiqueta,
        geometria=geometria,
    )
    print(f"Video: {video}")
    print(f"Listo → {rutas['base']}")


def ejecutar_magnetizacion_vs_t(geometria: str) -> None:
    """Barrido en T: |m|(T), Tc y video principal."""
    rutas = preparar_salidas_vs_t(geometria)
    etiqueta = etiqueta_red(geometria)
    tc_teo = tc_teorica(geometria)
    lado = lado_geometria(geometria)
    t_max = t_max_geometria(geometria)

    temps_previstas = _malla_temperaturas(
        T_MIN, t_max, N_TEMPERATURAS, geometria
    )
    grabador = GrabadorVideoTemperatura(
        carpeta_video=rutas["video"],
        tc_teorica=tc_teo,
        t_max_vista=float(temps_previstas[-1]),
        n_total=len(temps_previstas),
        fps=VIDEO_T_FPS,
        titulo_extra=etiqueta,
        geometria=geometria,
    )

    def on_paso(i, T_paso, m, _chi, red, temps, mags, chis):
        grabador.agregar_paso(i, T_paso, m, red, temps, mags, chis)

    print(f"{etiqueta}")
    print(f"Salida → {rutas['base']}")
    print(f"Video: {VIDEO_T_FPS} fps (1 frame por paso de T).\n")

    resultado = barrido_temperatura(
        geometria=geometria,
        t_max=t_max,
        L_red=lado,
        on_paso=on_paso,
    )

    print("\nEnsamblando video principal...")
    video = grabador.finalizar(rutas["video"] / "magnetizacion_hasta_tc.mp4")

    print("Generando gráfica magnetización vs temperatura...")
    ruta_fig = rutas["graficas"] / "magnetizacion_vs_temperatura.png"
    graficar_magnetizacion_vs_t(
        resultado.temperaturas,
        resultado.magnetizacion,
        resultado.susceptibilidad,
        resultado.tc_estimada,
        resultado.tc_teorica,
        ruta_fig,
        titulo_extra=etiqueta,
    )
    print(f"Gráfica: {ruta_fig}")

    print("Generando gráfica energía vs temperatura...")
    ruta_e = rutas["graficas"] / "energia_vs_temperatura.png"
    graficar_energia_vs_t(
        resultado.temperaturas,
        resultado.energia,
        resultado.calor_especifico,
        resultado.tc_estimada,
        resultado.tc_teorica,
        ruta_e,
        titulo_extra=etiqueta,
    )
    print(f"Gráfica: {ruta_e}")

    idx = resultado.indice_tc
    guardar_instantanea(
        resultado.redes_finales[idx],
        f"Red cerca de Tc ≈ {resultado.tc_estimada:.3f} ({geometria})",
        rutas["redes"] / "red_cerca_tc.png",
        geometria=geometria,
    )

    print(
        f"\nPromedio de producción: MCS ≥ {resultado.mcs_termalizacion}\n"
        f"Tc de referencia ({geometria}) ≈ {resultado.tc_teorica:.4f}\n"
        f"Tc estimada (máx. χ) ≈ {resultado.tc_estimada:.4f}\n"
        f"Puntos de T: {len(resultado.temperaturas)}\n"
        f"Video: {video}\n"
        f"Listo → {rutas['base']}"
    )


def ejecutar_tc_vs_tamano(geometria: str) -> None:
    """Barrido L=2,4,…,L_MAX: Tc(L) vs referencia, parada al 2 %."""
    print(f"{etiqueta_red(geometria)}")
    ejecutar_comparacion_tamano(geometria)
