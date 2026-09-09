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
    """Video principal de las tres cuadrículas; limpia frames temporales al terminar."""
    n_frames = min(len(m.grid_history) for _, m in modelos)
    cmap = _cmap_espines()
    carpeta_frames = ruta.parent / "_frames_mcs_tmp"
    carpeta_frames.mkdir(parents=True, exist_ok=True)
    for viejo in carpeta_frames.glob("frame_*.png"):
        viejo.unlink()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    imagenes = []
    for ax, (titulo, modelo) in zip(axes, modelos):
        im = ax.imshow(
            modelo.grid_history[0],
            cmap=cmap,
            vmin=-1,
            vmax=1,
            interpolation="nearest",
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

    salida: Path | None = None
    try:
        for frame in range(n_frames):
            for im, (_, modelo) in zip(imagenes, modelos):
                im.set_data(modelo.grid_history[frame])
            paso = 0 if frame == 0 else frame * stride
            titulo_paso.set_text(f"Paso de Monte Carlo: {paso}")
            fig.savefig(carpeta_frames / f"frame_{frame:04d}.png", dpi=100)

        salida = _ensamblar_video_desde_frames(carpeta_frames, n_frames, ruta, fps=float(fps))
    finally:
        plt.close(fig)
        _limpiar_carpeta_frames(carpeta_frames)

    if salida is None:
        return _crear_video_funcanimation(modelos, ruta, stride, fps)
    return salida


def _limpiar_carpeta_frames(carpeta: Path) -> None:
    """Elimina PNGs temporales y la carpeta si queda vacía."""
    if not carpeta.exists():
        return
    for f in carpeta.glob("frame_*.png"):
        f.unlink(missing_ok=True)
    try:
        carpeta.rmdir()
    except OSError:
        pass


def _crear_video_funcanimation(
    modelos: list[tuple[str, Ising2D]],
    ruta: Path,
    stride: int,
    fps: int,
) -> Path:
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
        anim.save(ruta, writer=FFMpegWriter(fps=fps, bitrate=1800), dpi=110)
        salida = ruta
    except (RuntimeError, ValueError, FileNotFoundError):
        salida = ruta.with_suffix(".gif")
        anim.save(salida, writer=PillowWriter(fps=min(fps, 12)), dpi=90)
    plt.close(fig)
    return salida


def graficar_magnetizacion_vs_t(
    temperaturas: np.ndarray,
    magnetizacion: np.ndarray,
    susceptibilidad: np.ndarray,
    tc_estimada: float,
    tc_teorica: float,
    ruta: Path,
) -> None:
    """|m|/N y susceptibilidad frente a T, con marcas de Tc."""
    fig, (ax_m, ax_chi) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_m.plot(temperaturas, magnetizacion, "o-", lw=1.8, ms=5, color="#C0392B", label=r"$\langle |m|/N\rangle$")
    ax_m.axvline(tc_teorica, color="#27AE60", ls="--", lw=1.4, label=rf"$T_c$ Onsager ≈ {tc_teorica:.3f}")
    ax_m.axvline(tc_estimada, color="#8E44AD", ls=":", lw=1.6, label=rf"$T_c$ estimada (máx. χ) ≈ {tc_estimada:.3f}")
    ax_m.set_ylabel(r"Magnetización  $\langle |m|/N\rangle$")
    ax_m.set_title("Magnetización en función de la temperatura")
    ax_m.legend(loc="upper right", fontsize=9)
    ax_m.grid(True, alpha=0.35)
    ax_m.set_ylim(-0.05, 1.05)

    ax_chi.plot(temperaturas, susceptibilidad, "s-", lw=1.6, ms=4.5, color="#2980B9", label=r"$\chi$")
    ax_chi.axvline(tc_teorica, color="#27AE60", ls="--", lw=1.4)
    ax_chi.axvline(tc_estimada, color="#8E44AD", ls=":", lw=1.6)
    ax_chi.set_xlabel(r"Temperatura  $T$")
    ax_chi.set_ylabel(r"Susceptibilidad  $\chi$")
    ax_chi.set_title("Susceptibilidad (pico ≈ temperatura crítica)")
    ax_chi.legend(loc="upper right", fontsize=9)
    ax_chi.grid(True, alpha=0.35)

    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=160)
    plt.close(fig)



def _ensamblar_video_desde_frames(
    carpeta_frames: Path,
    n_frames: int,
    ruta_salida: Path,
    fps: float,
) -> Path | None:
    """Une PNGs numerados con ffmpeg (o Pillow). Mucho más rápido que FuncAnimation."""
    import subprocess

    if n_frames <= 0:
        return None

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    patron = str(carpeta_frames / "frame_%04d.png")

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps:.6g}",
            "-i",
            patron,
            "-frames:v",
            str(n_frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(ruta_salida),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return ruta_salida
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        pass

    try:
        from PIL import Image

        salida = ruta_salida.with_suffix(".gif")
        imgs = [
            Image.open(carpeta_frames / f"frame_{i:04d}.png")
            for i in range(n_frames)
        ]
        dur_ms = max(int(1000 / max(fps, 1e-3)), 200)
        imgs[0].save(
            salida,
            save_all=True,
            append_images=imgs[1:],
            duration=dur_ms,
            loop=0,
        )
        for im in imgs:
            im.close()
        return salida
    except Exception:
        return None



class GrabadorVideoTemperatura:
    """Un PNG por paso de T; al final ensambla solo el video principal y limpia temporales."""

    def __init__(
        self,
        carpeta_video: Path,
        tc_teorica: float,
        t_max_vista: float,
        n_total: int,
        fps: float = 4.0,
        dpi: int = 100,
    ) -> None:
        self.carpeta_video = carpeta_video
        self.carpeta_frames = carpeta_video / "_frames_tmp"
        self.tc_teorica = tc_teorica
        self.t_max_vista = t_max_vista
        self.n_total = n_total
        self.fps = max(float(fps), 0.1)
        self.dpi = dpi
        self.n_frames = 0

        self.carpeta_frames.mkdir(parents=True, exist_ok=True)
        for viejo in self.carpeta_frames.glob("frame_*.png"):
            viejo.unlink()

        cmap = _cmap_espines()
        self.fig = plt.figure(figsize=(12.5, 5.2))
        self.ax_red = self.fig.add_subplot(1, 2, 1)
        self.ax_cur = self.fig.add_subplot(1, 2, 2)

        self.im = self.ax_red.imshow(
            np.ones((2, 2)),
            cmap=cmap,
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        self.ax_red.set_xticks([])
        self.ax_red.set_yticks([])
        self.titulo_red = self.ax_red.set_title("")

        (self.linea_hist,) = self.ax_cur.plot(
            [], [], "o-", color="#C0392B", lw=2.0, ms=5, label=r"$|m|/N$"
        )
        (self.punto,) = self.ax_cur.plot([], [], "o", color="#F39C12", ms=12, zorder=6)
        self.linea_t = self.ax_cur.axvline(
            1e-3, color="#E67E22", ls="-", lw=2.0, alpha=0.9, label="T actual"
        )
        self.ax_cur.axvline(
            tc_teorica,
            color="#27AE60",
            ls="--",
            lw=1.3,
            label=rf"$T_c$ Onsager ≈ {tc_teorica:.3f}",
        )
        self.linea_tc_est = self.ax_cur.axvline(
            tc_teorica,
            color="#8E44AD",
            ls=":",
            lw=1.5,
            label=r"$T_c$ estimada",
        )
        self.ax_cur.set_xlim(1e-3, t_max_vista * 1.02)
        self.ax_cur.set_ylim(-0.05, 1.05)
        self.ax_cur.set_xlabel(r"Temperatura  $T$")
        self.ax_cur.set_ylabel(r"$\langle |m|/N\rangle$")
        self.ax_cur.set_title("Magnetización vs $T$")
        self.ax_cur.legend(loc="upper right", fontsize=8)
        self.ax_cur.grid(True, alpha=0.35)

        self.texto_t = self.fig.text(
            0.5,
            0.02,
            "",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#1A5276",
        )
        self.fig.suptitle("Evolución hacia la temperatura crítica", fontsize=13)
        self.fig.tight_layout(rect=[0, 0.06, 1, 0.93])

    def agregar_paso(
        self,
        indice: int,
        T: float,
        m: float,
        red: np.ndarray,
        temps: np.ndarray,
        mags: np.ndarray,
        chis: np.ndarray,
    ) -> None:
        """Guarda solo el frame PNG del paso (sin videos intermedios)."""
        tc_est = float(temps[int(np.argmax(chis))]) if len(chis) else T

        self.im.set_data(red)
        self.titulo_red.set_text(f"Red final\npaso {indice + 1}/{self.n_total}")
        self.linea_hist.set_data(temps, mags)
        self.punto.set_data([T], [m])
        self.linea_t.set_xdata([T, T])
        self.linea_tc_est.set_xdata([tc_est, tc_est])
        self.ax_cur.set_xlim(max(float(temps[0]) * 0.5, 1e-4), self.t_max_vista * 1.02)
        self.texto_t.set_text(
            f"T = {T:.4f}   |m|/N = {m:.3f}   "
            f"paso {indice + 1}/{self.n_total}   Tc est.≈{tc_est:.3f}"
        )

        ruta_frame = self.carpeta_frames / f"frame_{self.n_frames:04d}.png"
        self.fig.savefig(ruta_frame, dpi=self.dpi)
        self.n_frames += 1

    def finalizar(self, ruta_final: Path) -> Path:
        try:
            salida = _ensamblar_video_desde_frames(
                self.carpeta_frames,
                self.n_frames,
                ruta_final,
                fps=self.fps,
            )
        finally:
            plt.close(self.fig)
            _limpiar_carpeta_frames(self.carpeta_frames)

        if salida is None:
            raise RuntimeError("No se pudo ensamblar el video de temperatura")
        return salida


def crear_video_temperatura(
    temperaturas: np.ndarray,
    magnetizacion: np.ndarray,
    redes_finales: list[np.ndarray],
    tc_estimada: float,
    tc_teorica: float,
    ruta: Path,
    fps: float = 4.0,
) -> Path:
    """Arma solo el video principal a partir del resultado completo."""
    del tc_estimada
    grabador = GrabadorVideoTemperatura(
        carpeta_video=ruta.parent,
        tc_teorica=tc_teorica,
        t_max_vista=float(temperaturas[-1]),
        n_total=len(temperaturas),
        fps=fps,
    )
    chis_dummy = np.ones_like(temperaturas)
    for i in range(len(temperaturas)):
        grabador.agregar_paso(
            i,
            float(temperaturas[i]),
            float(magnetizacion[i]),
            redes_finales[i],
            temperaturas[: i + 1],
            magnetizacion[: i + 1],
            chis_dummy[: i + 1],
        )
    return grabador.finalizar(ruta)
