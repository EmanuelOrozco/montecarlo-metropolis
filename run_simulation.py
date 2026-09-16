"""Punto de entrada para redes Ising 2D y cúbica simple 3D."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

from src.config import GEOMETRIAS, RESULTADOS, preparar_carpetas
from src.ejecucion import (
    ejecutar_magnetizacion_vs_t,
    ejecutar_tc_vs_tamano,
    ejecutar_temperatura_fija,
)
from src.ising import REDES, etiqueta_red


def _menu_interactivo() -> tuple[str, str]:
    print(
        "\n=== Modelo de Ising 2D/3D — Monte Carlo Metropolis ===\n"
        "Red:\n"
        "  1) Cuadrada      (2D, 4 vecinos)\n"
        "  2) Triangular    (2D, 6 vecinos)\n"
        "  3) Cúbica simple (3D, 6 vecinos)\n"
        "  4) Todas las redes\n"
    )
    while True:
        r = input("Elige la red [1/2/3/4]: ").strip()
        if r in {"1", "2", "3", "4"}:
            red = {
                "1": "cuadrada",
                "2": "triangular",
                "3": "cubica",
                "4": "todas",
            }[r]
            break
        print("Opción no válida.")

    print(
        "\nAnálisis:\n"
        "  1) Temperatura fija          →  .../temperatura_fija/\n"
        "  2) Magnetización vs T (y Tc) →  .../magnetizacion_vs_t/\n"
        "  3) Comparación Tc vs tamaño  →  .../tc_vs_tamano/\n"
        "  4) T fija + magnetización vs T\n"
    )
    while True:
        m = input("Elige el análisis [1/2/3/4]: ").strip()
        if m in {"1", "2", "3", "4"}:
            modo = {
                "1": "actual",
                "2": "temperatura",
                "3": "tamano",
                "4": "todo",
            }[m]
            break
        print("Opción no válida.")

    return red, modo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ising Metropolis: redes cuadrada, triangular o cúbica simple.",
    )
    parser.add_argument(
        "--red",
        choices=(*REDES, "ambas", "todas"),
        default=None,
        help="Geometría de la red (o ambas).",
    )
    parser.add_argument(
        "--modo",
        choices=("actual", "temperatura", "tamano", "todo"),
        default=None,
        help="actual = T fija; temperatura = |m|(T); tamano = Tc vs L; todo = T fija + |m|(T).",
    )
    return parser.parse_args(argv)


def _geometrias(red: str) -> tuple[str, ...]:
    if red in {"ambas", "todas"}:
        return GEOMETRIAS
    return (red,)


def _ejecutar(geometria: str, modo: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {etiqueta_red(geometria)}")
    print(f"{'=' * 60}")

    if modo in ("actual", "todo"):
        print("\n--- Temperatura fija ---\n")
        ejecutar_temperatura_fija(geometria)

    if modo in ("temperatura", "todo"):
        print("\n--- Magnetización vs temperatura ---\n")
        ejecutar_magnetizacion_vs_t(geometria)

    if modo == "tamano":
        print("\n--- Comparación Tc vs tamaño de celda ---\n")
        ejecutar_tc_vs_tamano(geometria)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.red is None or args.modo is None:
        red, modo = _menu_interactivo()
        if args.red is not None:
            red = args.red
        if args.modo is not None:
            modo = args.modo
    else:
        red, modo = args.red, args.modo

    preparar_carpetas(RESULTADOS)

    for geometria in _geometrias(red):
        _ejecutar(geometria, modo)

    print(f"\nResultados en: {RESULTADOS}")


if __name__ == "__main__":
    main()
