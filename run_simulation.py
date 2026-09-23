"""Punto de entrada Ising / Heisenberg clásico (Metropolis)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

from src.config import (
    GEOMETRIAS,
    GEOMETRIAS_HEISENBERG,
    INICIALIZACIONES,
    MODELOS,
    RESULTADOS,
    preparar_carpetas,
)
from src.ejecucion import (
    ejecutar_magnetizacion_vs_t,
    ejecutar_tc_vs_tamano,
    ejecutar_temperatura_fija,
)
from src.heisenberg import ejecutar_heisenberg_completo
from src.ising import REDES, etiqueta_red
from src.materials.estructura import etiqueta_estructura
from src.materials.mp_client import MATERIALES_MP


def _menu_interactivo() -> tuple[str, str, str, str]:
    print(
        "\n=== Monte Carlo Metropolis — Ising / Heisenberg clásico ===\n"
        "Modelo:\n"
        "  1) Ising (espines ±1)\n"
        "  2) Heisenberg clásico (vectores unitarios)\n"
    )
    while True:
        m = input("Elige el modelo [1/2]: ").strip()
        if m in {"1", "2"}:
            modelo = "ising" if m == "1" else "heisenberg"
            break
        print("Opción no válida.")

    if modelo == "ising":
        print(
            "\nRed:\n"
            "  1) Cuadrada      (2D)\n"
            "  2) Triangular    (2D)\n"
            "  3) Cúbica simple (SC)\n"
            "  4) BCC\n"
            "  5) FCC\n"
            "  6) Fe mp-13      (BCC, Materials Project)\n"
            "  7) Ni mp-23      (FCC, Materials Project)\n"
            "  8) Co mp-102     (FCC, Materials Project)\n"
            "  9) Todas\n"
        )
        mapa = {
            "1": "cuadrada",
            "2": "triangular",
            "3": "cubica",
            "4": "bcc",
            "5": "fcc",
            "6": "fe_mp13",
            "7": "ni_mp23",
            "8": "co_mp102",
            "9": "todas",
        }
    else:
        print(
            "\nRed (bulk 3D):\n"
            "  1) Cúbica simple (SC)\n"
            "  2) BCC\n"
            "  3) FCC\n"
            "  4) Fe mp-13  (BCC, Materials Project)\n"
            "  5) Ni mp-23  (FCC, Materials Project)\n"
            "  6) Co mp-102 (FCC, Materials Project)\n"
            "  7) Todos los materiales MP\n"
        )
        mapa = {
            "1": "cubica",
            "2": "bcc",
            "3": "fcc",
            "4": "fe_mp13",
            "5": "ni_mp23",
            "6": "co_mp102",
            "7": "materiales",
        }

    while True:
        r = input(f"Elige la red [{'/'.join(mapa)}]: ").strip()
        if r in mapa:
            red = mapa[r]
            break
        print("Opción no válida.")

    print(
        "\nInicialización de espines:\n"
        "  1) Aleatoria\n"
        "  2) Ferromagnética\n"
        "  3) Antiferromagnética\n"
    )
    while True:
        i = input("Elige inicialización [1/2/3]: ").strip()
        if i in {"1", "2", "3"}:
            inicializacion = {
                "1": "aleatorio",
                "2": "ferromagnetico",
                "3": "antiferromagnetico",
            }[i]
            break
        print("Opción no válida.")

    print(
        "\nAnálisis:\n"
        "  1) Temperatura fija\n"
        "  2) Magnetización vs T (y Tc)\n"
        "  3) Tc vs tamaño / caja mínima (+ calibración J en Heisenberg)\n"
        "  4) Pipeline completo\n"
    )
    while True:
        a = input("Elige el análisis [1/2/3/4]: ").strip()
        if a in {"1", "2", "3", "4"}:
            modo = {"1": "actual", "2": "temperatura", "3": "tamano", "4": "todo"}[a]
            break
        print("Opción no válida.")

    return modelo, red, modo, inicializacion


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Metropolis: Ising (±1) o Heisenberg clásico (vectores unitarios). "
            "Resultados en resultados/<modelo>/<geometria>/..."
        ),
    )
    parser.add_argument("--modelo", choices=MODELOS, default=None)
    parser.add_argument(
        "--red",
        choices=(*REDES, "ambas", "todas", "materiales"),
        default=None,
    )
    parser.add_argument(
        "--modo",
        choices=("actual", "temperatura", "tamano", "todo"),
        default=None,
    )
    parser.add_argument(
        "--init",
        choices=INICIALIZACIONES,
        default=None,
        help="Estado inicial de los espines.",
    )
    return parser.parse_args(argv)


def _geometrias(red: str, modelo: str) -> tuple[str, ...]:
    if red in {"ambas", "todas"}:
        return GEOMETRIAS if modelo == "ising" else GEOMETRIAS_HEISENBERG
    if red == "materiales":
        return tuple(MATERIALES_MP.keys())
    if modelo == "heisenberg" and red not in GEOMETRIAS_HEISENBERG:
        raise ValueError(
            f"Heisenberg solo admite {GEOMETRIAS_HEISENBERG}, recibido {red!r}"
        )
    return (red,)


def _ejecutar_ising(geometria: str, modo: str) -> None:
    etiqueta = (
        etiqueta_estructura(geometria)
        if geometria in MATERIALES_MP
        else etiqueta_red(geometria)
    )
    print(f"\n{'=' * 60}")
    print(f"  Ising — {etiqueta}")
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


def _ejecutar_heisenberg(geometria: str, modo: str, inicializacion: str) -> None:
    etiqueta = etiqueta_estructura(geometria)
    print(f"\n{'=' * 60}")
    print(f"  Heisenberg clásico — {etiqueta}")
    print(f"  init={inicializacion} | kB en eV/K | T en K (≥0.1)")
    print(f"{'=' * 60}")
    ejecutar_heisenberg_completo(
        geometria, inicializacion=inicializacion, modo=modo
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.modelo is None or args.red is None or args.modo is None:
        modelo, red, modo, inicializacion = _menu_interactivo()
        if args.modelo is not None:
            modelo = args.modelo
        if args.red is not None:
            red = args.red
        if args.modo is not None:
            modo = args.modo
        if args.init is not None:
            inicializacion = args.init
    else:
        modelo, red, modo = args.modelo, args.red, args.modo
        inicializacion = args.init or "ferromagnetico"

    preparar_carpetas(RESULTADOS, RESULTADOS / "ising", RESULTADOS / "heisenberg")

    for geometria in _geometrias(red, modelo):
        if modelo == "ising":
            _ejecutar_ising(geometria, modo)
        else:
            _ejecutar_heisenberg(geometria, modo, inicializacion)

    print(f"\nResultados en: {RESULTADOS}/<ising|heisenberg>/<geometria>/...")


if __name__ == "__main__":
    main()
