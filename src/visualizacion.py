"""Gráficas de energía/magnetización y video de la cuadrícula de espines."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from .config import COLOR_ABAJO, COLOR_ARRIBA, MCS_TERMALIZACION
from .ising import IsingBase


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


def ventana_hasta_equilibrio(
    series: list[np.ndarray],
    mcs_promedio: int = MCS_TERMALIZACION,
) -> int:
    """Corta el eje X tras ver el transitorio, el corte de 200 MCS y un tramo de meseta."""
    n = min(len(s) for s in series)
    t_eq = max(paso_estabilizacion(s[:n]) for s in series)
    extra = max(60, int(0.8 * t_eq))
    return int(min(n - 1, max(t_eq + extra, mcs_promedio + 80, 80)))


def _graficar_series(
    modelos: list[tuple[str, IsingBase]],
    historias: list[np.ndarray],
    ruta: Path,
    ylabel: str,
    titulo: str,
    mcs_promedio: int = MCS_TERMALIZACION,
) -> None:
    xmax = ventana_hasta_equilibrio(historias, mcs_promedio=mcs_promedio)
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
        if len(y) > mcs_promedio:
            mu = float(np.mean(y[mcs_promedio : xmax + 1]))
            ax.hlines(
                mu,
                mcs_promedio,
                xmax,
                colors=linea.get_color(),
                linestyles=":",
                lw=1.15,
                alpha=0.85,
            )

    if 0 < mcs_promedio <= xmax:
        ax.axvline(
            mcs_promedio,
            color="0.25",
            ls="-.",
            lw=1.3,
            label=f"promedio MCS ≥ {mcs_promedio}",
        )

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
    modelos: list[tuple[str, IsingBase]],
    ruta: Path,
    MCS: int | None = None,
    titulo_extra: str = "",
) -> None:
    del MCS
    nspin = modelos[0][1].Nspin
    series = _valores_por_espin([m.energy_history for _, m in modelos], nspin)
    titulo = "Energía por sitio hasta la estabilización"
    if titulo_extra:
        titulo = f"{titulo} — {titulo_extra}"
    _graficar_series(
        modelos,
        series,
        ruta,
        ylabel=r"Energía por sitio  $E/N$",
        titulo=titulo,
    )


def graficar_magnetizacion(
    modelos: list[tuple[str, IsingBase]],
    ruta: Path,
    MCS: int | None = None,
    titulo_extra: str = "",
) -> None:
    del MCS
    nspin = modelos[0][1].Nspin
    series = _valores_por_espin([m.magnetization_history for _, m in modelos], nspin)
    titulo = "Magnetización por sitio hasta la estabilización"
    if titulo_extra:
        titulo = f"{titulo} — {titulo_extra}"
    _graficar_series(
        modelos,
        series,
        ruta,
        ylabel=r"Magnetización por sitio  $m/N$",
        titulo=titulo,
    )


def _coords_triangulares(L: int) -> tuple[np.ndarray, np.ndarray]:
    i, j = np.indices((L, L))
    xs = (j + 0.5 * (i % 2)).astype(float).ravel()
    ys = (i * np.sqrt(3) / 2).astype(float).ravel()
    return xs, ys


def _colores_de_spins(spins: np.ndarray) -> np.ndarray:
    flat = np.asarray(spins).ravel()
    return np.where(flat > 0, COLOR_ARRIBA, COLOR_ABAJO)


def _tamano_marcador(L: int, panel_ancho: float = 4.0) -> float:
    return max(12.0, 1800.0 * panel_ancho / (L * L))


def _configurar_eje_triangular(ax, L: int) -> None:
    """Límites y aspecto para que los triángulos se vean equiláteros."""
    margin = 0.65
    xmax = (L - 1) + 0.5 + margin
    ymax = (L - 1) * (np.sqrt(3) / 2) + margin
    ax.set_xlim(-margin, xmax)
    ax.set_ylim(ymax, -margin)  # y hacia abajo, sin invert_yaxis extra
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _dibujar_red(ax, spins: np.ndarray, geometria: str = "cuadrada"):
    """Dibuja la red y devuelve el artista actualizable (AxesImage o PathCollection)."""
    if geometria != "triangular":
        im = ax.imshow(
            spins,
            cmap=_cmap_espines(),
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    L = spins.shape[0]
    xs, ys = _coords_triangulares(L)
    sc = ax.scatter(
        xs,
        ys,
        c=_colores_de_spins(spins),
        s=_tamano_marcador(L, panel_ancho=5.0),
        marker="o",
        edgecolors="0.2",
        linewidths=0.4,
        zorder=3,
    )
    _configurar_eje_triangular(ax, L)
    return sc


def _actualizar_red(artista, spins: np.ndarray, geometria: str = "cuadrada") -> None:
    if geometria == "triangular":
        artista.set_facecolor(_colores_de_spins(spins))
    else:
        artista.set_data(spins)


def guardar_instantanea(
    spins: np.ndarray,
    titulo: str,
    ruta: Path,
    geometria: str = "cuadrada",
) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    _dibujar_red(ax, spins, geometria=geometria)
    ax.set_title(titulo)
    ax.legend(handles=_leyenda_espines(), loc="upper right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


def crear_video(
    modelos: list[tuple[str, IsingBase]],
    ruta: Path,
    stride: int,
    fps: int = 20,
    titulo_extra: str = "",
    geometria: str = "cuadrada",
) -> Path:
    """Video principal de las tres redes; limpia frames temporales al terminar."""
    if not geometria:
        geometria = getattr(modelos[0][1], "geometria", "cuadrada")

    n_frames = min(len(m.grid_history) for _, m in modelos)
    carpeta_frames = ruta.parent / "_frames_mcs_tmp"
    carpeta_frames.mkdir(parents=True, exist_ok=True)
    for viejo in carpeta_frames.glob("frame_*.png"):
        viejo.unlink()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    artistas = []
    for ax, (titulo, modelo) in zip(axes, modelos):
        art = _dibujar_red(ax, modelo.grid_history[0], geometria=geometria)
        ax.set_title(titulo)
        artistas.append(art)

    fig.legend(
        handles=_leyenda_espines(),
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02),
    )
    prefijo = f"{titulo_extra}  |  " if titulo_extra else ""
    titulo_paso = fig.suptitle(f"{prefijo}Paso de Monte Carlo: 0", fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])

    salida: Path | None = None
    try:
        for frame in range(n_frames):
            for art, (_, modelo) in zip(artistas, modelos):
                _actualizar_red(art, modelo.grid_history[frame], geometria)
            paso = 0 if frame == 0 else frame * stride
            titulo_paso.set_text(f"{prefijo}Paso de Monte Carlo: {paso}")
            fig.savefig(carpeta_frames / f"frame_{frame:04d}.png", dpi=100)

        salida = _ensamblar_video_desde_frames(carpeta_frames, n_frames, ruta, fps=float(fps))
    finally:
        plt.close(fig)
        _limpiar_carpeta_frames(carpeta_frames)

    if salida is None:
        return _crear_video_funcanimation(modelos, ruta, stride, fps, geometria=geometria)
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
    modelos: list[tuple[str, IsingBase]],
    ruta: Path,
    stride: int,
    fps: int,
    geometria: str = "cuadrada",
) -> Path:
    n_frames = min(len(m.grid_history) for _, m in modelos)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    artistas = []
    for ax, (titulo, modelo) in zip(axes, modelos):
        art = _dibujar_red(ax, modelo.grid_history[0], geometria=geometria)
        ax.set_title(titulo)
        artistas.append(art)
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
        for art, (_, modelo) in zip(artistas, modelos):
            _actualizar_red(art, modelo.grid_history[frame], geometria)
        paso = 0 if frame == 0 else frame * stride
        titulo_paso.set_text(f"Paso de Monte Carlo: {paso}")
        return [*artistas, titulo_paso]

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
    titulo_extra: str = "",
) -> None:
    """|m|/N y susceptibilidad frente a T, con marcas de Tc."""
    fig, (ax_m, ax_chi) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_m.plot(temperaturas, magnetizacion, "o-", lw=1.8, ms=5, color="#C0392B", label=r"$\langle |m|/N\rangle$")
    ax_m.axvline(tc_teorica, color="#27AE60", ls="--", lw=1.4, label=rf"$T_c$ teórica ≈ {tc_teorica:.3f}")
    ax_m.axvline(tc_estimada, color="#8E44AD", ls=":", lw=1.6, label=rf"$T_c$ estimada (máx. χ) ≈ {tc_estimada:.3f}")
    ax_m.set_ylabel(r"Magnetización  $\langle |m|/N\rangle$")
    titulo_m = "Magnetización en función de la temperatura"
    if titulo_extra:
        titulo_m = f"{titulo_m} — {titulo_extra}"
    ax_m.set_title(titulo_m)
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


def graficar_energia_vs_t(
    temperaturas: np.ndarray,
    energia: np.ndarray,
    calor_especifico: np.ndarray,
    tc_estimada: float,
    tc_teorica: float,
    ruta: Path,
    titulo_extra: str = "",
) -> None:
    """E/N y calor específico frente a T, con marcas de Tc."""
    fig, (ax_e, ax_c) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_e.plot(temperaturas, energia, "o-", lw=1.8, ms=5, color="#1F618D", label=r"$\langle E/N\rangle$")
    ax_e.axvline(tc_teorica, color="#27AE60", ls="--", lw=1.4, label=rf"$T_c$ teórica ≈ {tc_teorica:.3f}")
    ax_e.axvline(tc_estimada, color="#8E44AD", ls=":", lw=1.6, label=rf"$T_c$ estimada (máx. χ) ≈ {tc_estimada:.3f}")
    ax_e.set_ylabel(r"Energía  $\langle E/N\rangle$")
    titulo_e = "Energía en función de la temperatura"
    if titulo_extra:
        titulo_e = f"{titulo_e} — {titulo_extra}"
    ax_e.set_title(titulo_e)
    ax_e.legend(loc="upper left", fontsize=9)
    ax_e.grid(True, alpha=0.35)

    ax_c.plot(temperaturas, calor_especifico, "s-", lw=1.6, ms=4.5, color="#D35400", label=r"$C$")
    ax_c.axvline(tc_teorica, color="#27AE60", ls="--", lw=1.4)
    ax_c.axvline(tc_estimada, color="#8E44AD", ls=":", lw=1.6)
    ax_c.set_xlabel(r"Temperatura  $T$")
    ax_c.set_ylabel(r"Calor específico  $C$")
    ax_c.set_title(r"Calor específico $C=\mathrm{Var}(E)/(N T^2)$ (pico cerca de $T_c$)")
    ax_c.legend(loc="upper right", fontsize=9)
    ax_c.grid(True, alpha=0.35)

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
        titulo_extra: str = "",
        geometria: str = "cuadrada",
    ) -> None:
        self.carpeta_video = carpeta_video
        self.carpeta_frames = carpeta_video / "_frames_tmp"
        self.tc_teorica = tc_teorica
        self.t_max_vista = t_max_vista
        self.n_total = n_total
        self.fps = max(float(fps), 0.1)
        self.dpi = dpi
        self.n_frames = 0
        self.titulo_extra = titulo_extra
        self.geometria = geometria

        self.carpeta_frames.mkdir(parents=True, exist_ok=True)
        for viejo in self.carpeta_frames.glob("frame_*.png"):
            viejo.unlink()

        self.fig = plt.figure(figsize=(12.5, 5.4))
        self.ax_red = self.fig.add_subplot(1, 2, 1)
        self.ax_cur = self.fig.add_subplot(1, 2, 2)

        # Placeholder inicial; se reemplaza en el primer agregar_paso.
        self.artista_red = _dibujar_red(
            self.ax_red,
            np.ones((2, 2), dtype=int),
            geometria=geometria,
        )
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
            label=rf"$T_c$ teórica ≈ {tc_teorica:.3f}",
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
        subt = "Evolución hacia la temperatura crítica"
        if titulo_extra:
            subt = f"{titulo_extra} — {subt}"
        self.fig.suptitle(subt, fontsize=13)
        self.fig.tight_layout(rect=[0, 0.06, 1, 0.93])
        self._eje_red_listo = False

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

        if not self._eje_red_listo:
            self.ax_red.clear()
            self.artista_red = _dibujar_red(self.ax_red, red, geometria=self.geometria)
            self.titulo_red = self.ax_red.set_title("")
            self._eje_red_listo = True
        else:
            _actualizar_red(self.artista_red, red, self.geometria)

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
    geometria: str = "cuadrada",
) -> Path:
    """Arma solo el video principal a partir del resultado completo."""
    del tc_estimada
    grabador = GrabadorVideoTemperatura(
        carpeta_video=ruta.parent,
        tc_teorica=tc_teorica,
        t_max_vista=float(temperaturas[-1]),
        n_total=len(temperaturas),
        fps=fps,
        geometria=geometria,
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
