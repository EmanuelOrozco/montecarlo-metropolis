"""Punto de entrada: tres soluciones Ising 2D (aleatoria, ferro y antiferro)."""

from __future__ import annotations

import sys
from pathlib import Path

# Permite ejecutar `python run_simulation.py` sin instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

from src.config import CASOS, GRAFICAS, H, KB, L, MCS, REDES, SEED, SOLUCION, T, VIDEO
from src.ising import Ising2D
from src.visualizacion import (
    crear_video,
    guardar_instantanea,
    graficar_energia,
    graficar_magnetizacion,
)


def _preparar_carpetas() -> None:
    for ruta in (SOLUCION, GRAFICAS, VIDEO, REDES):
        ruta.mkdir(parents=True, exist_ok=True)


def main() -> None:
    _preparar_carpetas()
    stride_video = max(1, MCS // 250)

    print(f"Cuadrícula {L}×{L}  |  MCS={MCS}  |  T={T}  |  h={H}")
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
            REDES / f"red_{caso['clave']}_inicial.png",
        )
        print(f"Simulando: {caso['titulo']}  (J={caso['J']:+.1f}) ...")
        modelo.simular(MCS, guardar_red_cada=stride_video)
        guardar_instantanea(
            modelo.spins,
            f"{caso['titulo']} — final (MCS={MCS})",
            REDES / f"red_{caso['clave']}_final.png",
        )
        e_media, m_media = modelo.promedios()
        n = modelo.Nspin
        print(
            f"  ⟨E/N⟩ = {e_media / n:.4f}   ⟨m/N⟩ = {m_media / n:.4f}   "
            f"E_final/N = {modelo.E / n:.4f}   m_final/N = {modelo.m / n:.4f}"
        )
        modelos.append((caso["titulo"], modelo))

    print("\nGenerando gráficas...")
    graficar_energia(modelos, GRAFICAS / "energia_vs_mcs.png", MCS)
    graficar_magnetizacion(modelos, GRAFICAS / "magnetizacion_vs_mcs.png", MCS)

    print("Generando video de la cuadrícula...")
    video = crear_video(
        modelos,
        VIDEO / "evolucion_espines.mp4",
        stride=stride_video,
        fps=20,
    )
    print(f"Video: {video}")
    print(f"Solución en: {SOLUCION}")


if __name__ == "__main__":
    main()
