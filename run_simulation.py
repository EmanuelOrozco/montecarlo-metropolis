"""Punto de entrada: elige red (cuadrada/triangular) y modo de análisis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

from src.config import GEOMETRIAS, RESULTADOS, preparar_carpetas
from src.ejecucion import ejecutar_magnetizacion_vs_t, ejecutar_temperatura_fija
from src.ising import REDES, etiqueta_red


def _menu_interactivo() -> tuple[str, str]:
    print(
        "\n=== Modelo de Ising 2D — Monte Carlo Metropolis ===\n"
        "Red:\n"
        "  1) Cuadrada   (4 vecinos)\n"
        "  2) Triangular (6 vecinos)\n"
        "  3) Ambas redes\n"
    )
    while True:
        r = input("Elige la red [1/2/3]: ").strip()
        if r in {"1", "2", "3"}:
            red = {"1": "cuadrada", "2": "triangular", "3": "ambas"}[r]
            break
        print("Opción no válida.")

    print(
        "\nAnálisis:\n"
        "  1) Temperatura fija          →  .../temperatura_fija/\n"
        "  2) Magnetización vs T (y Tc) →  .../magnetizacion_vs_t/\n"
        "  3) Ambos análisis\n"
    )
    while True:
        m = input("Elige el análisis [1/2/3]: ").strip()
        if m in {"1", "2", "3"}:
            modo = {"1": "actual", "2": "temperatura", "3": "todo"}[m]
            break
        print("Opción no válida.")

    return red, modo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ising 2D Metropolis: red cuadrada o triangular.",
    )
    parser.add_argument(
        "--red",
        choices=(*REDES, "ambas"),
        default=None,
        help="Geometría de la red (o ambas).",
    )
    parser.add_argument(
        "--modo",
        choices=("actual", "temperatura", "todo"),
        default=None,
        help="actual = T fija; temperatura = |m|(T); todo = ambos.",
    )
    return parser.parse_args(argv)


def _geometrias(red: str) -> tuple[str, ...]:
    if red == "ambas":
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
