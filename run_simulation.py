"""Punto de entrada: simulación a T fija, barrido en T, o ambos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite ejecutar `python run_simulation.py` sin instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

from src.analisis_temperatura import _malla_temperaturas, barrido_temperatura, tc_onsager
from src.config import (
    ANALISIS_VS_T,
    CASOS,
    GRAFICAS_T_FIJA,
    GRAFICAS_VS_T,
    H,
    KB,
    L,
    MCS,
    N_TEMPERATURAS,
    REDES_T_FIJA,
    REDES_VS_T,
    RESULTADOS,
    SEED,
    SIM_T_FIJA,
    T,
    T_MAX,
    T_MIN,
    VIDEO_T_FIJA,
    VIDEO_T_FPS,
    VIDEO_VS_T,
    carpetas_t_fija,
    carpetas_vs_t,
    preparar_carpetas,
)
from src.ising import Ising2D
from src.visualizacion import (
    GrabadorVideoTemperatura,
    crear_video,
    guardar_instantanea,
    graficar_energia,
    graficar_magnetizacion,
    graficar_magnetizacion_vs_t,
)


def ejecutar_simulacion_actual() -> None:
    """Simulación a T fija: tres inicializaciones, gráficas y video MCS."""
    preparar_carpetas(*carpetas_t_fija())
    stride_video = max(1, MCS // 250)

    print(f"Cuadrícula {L}×{L}  |  MCS={MCS}  |  T={T}  |  h={H}")
    print(f"Salida → {SIM_T_FIJA}")
    print(f"Guardando red cada {stride_video} paso(s) para el video.\n")

    modelos: list[tuple[str, Ising2D]] = []
    for i, caso in enumerate(CASOS):
        modelo = Ising2D(
            L=L,
            J=caso["J"],
            h=H,
            T=T,
            kB=KB,
            inicializacion=caso["inicializacion"],
            seed=SEED + i,
        )
        guardar_instantanea(
            modelo.spins,
            f"{caso['titulo']} — inicial",
            REDES_T_FIJA / f"red_{caso['clave']}_inicial.png",
        )
        print(f"Simulando: {caso['titulo']}  (J={caso['J']:+.1f}) ...")
        modelo.simular(MCS, guardar_red_cada=stride_video)
        guardar_instantanea(
            modelo.spins,
            f"{caso['titulo']} — final (MCS={MCS})",
            REDES_T_FIJA / f"red_{caso['clave']}_final.png",
        )
        e_media, m_media = modelo.promedios()
        n = modelo.Nspin
        print(
            f"  ⟨E/N⟩ = {e_media / n:.4f}   ⟨m/N⟩ = {m_media / n:.4f}   "
            f"E_final/N = {modelo.E / n:.4f}   m_final/N = {modelo.m / n:.4f}"
        )
        modelos.append((caso["titulo"], modelo))

    print("\nGenerando gráficas...")
    graficar_energia(modelos, GRAFICAS_T_FIJA / "energia_vs_mcs.png", MCS)
    graficar_magnetizacion(modelos, GRAFICAS_T_FIJA / "magnetizacion_vs_mcs.png", MCS)

    print("Generando video de la cuadrícula...")
    video = crear_video(
        modelos,
        VIDEO_T_FIJA / "evolucion_espines.mp4",
        stride=stride_video,
        fps=20,
    )
    print(f"Video: {video}")
    print(f"Resultados (T fija) en: {SIM_T_FIJA}")


def ejecutar_analisis_temperatura() -> None:
    """Barrido en T: |m|(T), estimación de Tc y un solo video principal."""
    preparar_carpetas(*carpetas_vs_t())

    temps_previstas = _malla_temperaturas(T_MIN, T_MAX, N_TEMPERATURAS)
    grabador = GrabadorVideoTemperatura(
        carpeta_video=VIDEO_VS_T,
        tc_teorica=tc_onsager(),
        t_max_vista=float(temps_previstas[-1]),
        n_total=len(temps_previstas),
        fps=VIDEO_T_FPS,
    )

    def on_paso(i, T_paso, m, _chi, red, temps, mags, chis):
        grabador.agregar_paso(i, T_paso, m, red, temps, mags, chis)

    print(f"Salida → {ANALISIS_VS_T}")
    print(f"Video: {VIDEO_T_FPS} fps (1 frame por paso de T).\n")
    resultado = barrido_temperatura(on_paso=on_paso)

    print("\nEnsamblando video principal...")
    video = grabador.finalizar(VIDEO_VS_T / "magnetizacion_hasta_tc.mp4")

    print("Generando gráfica magnetización vs temperatura...")
    ruta_fig = GRAFICAS_VS_T / "magnetizacion_vs_temperatura.png"
    graficar_magnetizacion_vs_t(
        resultado.temperaturas,
        resultado.magnetizacion,
        resultado.susceptibilidad,
        resultado.tc_estimada,
        resultado.tc_teorica,
        ruta_fig,
    )
    print(f"Gráfica: {ruta_fig}")

    idx = resultado.indice_tc
    guardar_instantanea(
        resultado.redes_finales[idx],
        f"Red cerca de Tc ≈ {resultado.tc_estimada:.3f}",
        REDES_VS_T / "red_cerca_tc.png",
    )

    print(
        f"\nTc Onsager (teoría) ≈ {resultado.tc_teorica:.4f}\n"
        f"Tc estimada (máx. χ) ≈ {resultado.tc_estimada:.4f}\n"
        f"Puntos de T: {len(resultado.temperaturas)}\n"
        f"Video: {video}\n"
        f"Resultados (vs T) en: {ANALISIS_VS_T}"
    )


def _menu_interactivo() -> str:
    print(
        "\n=== Modelo de Ising 2D — Monte Carlo Metropolis ===\n"
        "  1) Simulación a temperatura fija  →  resultados/simulacion_temperatura_fija/\n"
        "  2) Análisis magnetización vs T    →  resultados/analisis_magnetizacion_vs_t/\n"
        "  3) Todo el proyecto (1 + 2)\n"
    )
    while True:
        opcion = input("Elige una opción [1/2/3]: ").strip()
        if opcion in {"1", "2", "3"}:
            return {"1": "actual", "2": "temperatura", "3": "todo"}[opcion]
        print("Opción no válida. Usa 1, 2 o 3.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulación Ising 2D: T fija, barrido en T, o ambos.",
    )
    parser.add_argument(
        "--modo",
        choices=("actual", "temperatura", "todo"),
        default=None,
        help=(
            "actual = simulación a T fija; "
            "temperatura = |m|(T) y Tc; "
            "todo = ambos"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    modo = args.modo if args.modo is not None else _menu_interactivo()

    preparar_carpetas(RESULTADOS)

    if modo in ("actual", "todo"):
        print("\n--- 1) Simulación a temperatura fija ---\n")
        ejecutar_simulacion_actual()

    if modo in ("temperatura", "todo"):
        print("\n--- 2) Análisis magnetización vs temperatura ---\n")
        ejecutar_analisis_temperatura()


if __name__ == "__main__":
    main()
