"""Gráficas de energía/magnetización y video de la cuadrícula de espines."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from .config import COLOR_ABAJO, COLOR_ARRIBA
from .ising import Ising2D


def _cmap_espines() -> ListedColormap:
    return ListedColormap([COLOR_ABAJO, COLOR_ARRIBA])


def _leyenda_espines() -> list[Patch]:
    return [
        Patch(facecolor=COLOR_ARRIBA, edgecolor="black", label="Espín arriba  (+1)"),
        Patch(facecolor=COLOR_ABAJO, edgecolor="black", label="Espín abajo   (−1 ≡ 0)"),
    ]


def _valores_por_espin(historias: list[list[float]], nspin: int) -> list[np.ndarray]:
    return [np.asarray(h, dtype=float) / nspin for h in historias]


def paso_estabilizacion(valores: np.ndarray, ventana: int = 25) -> int:
    """Primer paso en el que la media móvil ya no se aparta del régimen de equilibrio."""
    y = np.asarray(valores, dtype=float)
    n = len(y)
    if n < 4:
        return max(n - 1, 0)

    ventana = int(np.clip(ventana, 5, max(5, n // 4)))
    cola = y[-max(ventana, n // 5) :]
    mu = float(np.mean(cola))
    sigma = float(np.std(cola))
    amplitud = float(np.ptp(y))
    banda = max(4.0 * sigma, 0.04 * max(amplitud, 1e-6), 0.015)

    if abs(float(y[0]) - mu) <= banda:
        return 0

    kernel = np.ones(ventana) / ventana
    media_movil = np.convolve(y, kernel, mode="valid")
    dentro = np.abs(media_movil - mu) <= banda

    estable = n - 1
    for i in range(len(dentro)):
        if np.all(dentro[i:]):
            estable = i + ventana - 1
            break
    return int(np.clip(estable, 0, n - 1))


def ventana_hasta_equilibrio(series: list[np.ndarray]) -> int:
    """Corta el eje X tras ver el transitorio y un tramo claro de meseta."""
    n = min(len(s) for s in series)
    t_eq = max(paso_estabilizacion(s[:n]) for s in series)
    extra = max(60, int(0.8 * t_eq))
    return int(min(n - 1, max(t_eq + extra, 80)))


def _graficar_series(
    modelos: list[tuple[str, Ising2D]],
    historias: list[np.ndarray],
    ruta: Path,
    ylabel: str,
    titulo: str,
) -> None:
    xmax = ventana_hasta_equilibrio(historias)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    t_eq_max = 0
    for (nombre, _), y in zip(modelos, historias):
        pasos = np.arange(len(y))
        (linea,) = ax.plot(pasos[: xmax + 1], y[: xmax + 1], lw=1.6, label=nombre)
        t_eq = paso_estabilizacion(y)
        t_eq_max = max(t_eq_max, t_eq)
        if 0 < t_eq <= xmax:
            ax.plot(t_eq, y[t_eq], "o", color=linea.get_color(), ms=6, zorder=5)
            ax.axvline(t_eq, color=linea.get_color(), ls="--", lw=0.9, alpha=0.45)

    ax.set_xlim(0, xmax)
    ax.set_xlabel("Pasos de Monte Carlo")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.legend()
    ax.grid(True, alpha=0.35)
    if t_eq_max > 0:
        ax.annotate(
            f"equilibrio ≈ paso {t_eq_max}",
            xy=(t_eq_max, ax.get_ylim()[0]),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=9,
            color="0.25",
        )
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def graficar_energia(
    modelos: list[tuple[str, Ising2D]],
    ruta: Path,
    MCS: int | None = None,
) -> None:
    del MCS
    nspin = modelos[0][1].Nspin
    series = _valores_por_espin([m.energy_history for _, m in modelos], nspin)
    _graficar_series(
        modelos,
        series,
        ruta,
        ylabel=r"Energía por sitio  $E/N$",
        titulo="Energía por sitio hasta la estabilización",
    )


def graficar_magnetizacion(
    modelos: list[tuple[str, Ising2D]],
    ruta: Path,
    MCS: int | None = None,
) -> None:
    del MCS
    nspin = modelos[0][1].Nspin
    series = _valores_por_espin([m.magnetization_history for _, m in modelos], nspin)
    _graficar_series(
        modelos,
        series,
        ruta,
        ylabel=r"Magnetización por sitio  $m/N$",
        titulo="Magnetización por sitio hasta la estabilización",
    )


def guardar_instantanea(spins: np.ndarray, titulo: str, ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(spins, cmap=_cmap_espines(), vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(titulo)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(handles=_leyenda_espines(), loc="upper right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


def crear_video(
    modelos: list[tuple[str, Ising2D]],
    ruta: Path,
    stride: int,
    fps: int = 20,
) -> Path:
    """Video con las tres cuadrículas evolucionando en paralelo."""
    n_frames = min(len(m.grid_history) for _, m in modelos)
    cmap = _cmap_espines()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    imagenes = []
    for ax, (titulo, modelo) in zip(axes, modelos):
        im = ax.imshow(
            modelo.grid_history[0],
            cmap=cmap,
            vmin=-1,
            vmax=1,
            interpolation="nearest",
            animated=True,
        )
        ax.set_title(titulo)
        ax.set_xticks([])
        ax.set_yticks([])
        imagenes.append(im)

    fig.legend(
        handles=_leyenda_espines(),
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02),
    )
    titulo_paso = fig.suptitle("Paso de Monte Carlo: 0", fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])

    def actualizar(frame: int):
        for im, (_, modelo) in zip(imagenes, modelos):
            im.set_data(modelo.grid_history[frame])
        paso = 0 if frame == 0 else frame * stride
        titulo_paso.set_text(f"Paso de Monte Carlo: {paso}")
        return [*imagenes, titulo_paso]

    anim = FuncAnimation(fig, actualizar, frames=n_frames, interval=1000 / fps, blit=False)

    ruta.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(ruta, writer=writer, dpi=110)
        salida = ruta
    except (RuntimeError, ValueError, FileNotFoundError):
        salida = ruta.with_suffix(".gif")
        anim.save(salida, writer=PillowWriter(fps=min(fps, 12)), dpi=90)

    plt.close(fig)
    return salida
