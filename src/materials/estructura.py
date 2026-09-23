"""Validación y clasificación cristalográfica de estructuras Materials Project."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mp_client import asegurar_estructura, material_id_de_geometria


@dataclass(frozen=True, slots=True)
class AnalisisEstructura:
    material_id: str
    formula: str
    tipo_red: str
    simbolo_espacial: str
    sitios_por_celda: int
    coordinacion: int
    a_angstrom: float
    bipartita: bool
    valida: bool
    mensaje: str

    def etiqueta_corta(self) -> str:
        return f"{self.tipo_red}, z={self.coordinacion}"

    def etiqueta_grafica(self) -> str:
        return (
            f"{self.formula} ({self.material_id}, {self.tipo_red}, "
            f"z={self.coordinacion}, a={self.a_angstrom:.3f} Å)"
        )


def _simbolo_espacial(texto: str) -> str:
    if "symbol='" in texto:
        return texto.split("symbol='")[1].split("'")[0]
    if "Im-3m" in texto:
        return "Im-3m"
    if "Fm-3m" in texto:
        return "Fm-3m"
    return texto[:32] if texto else "desconocido"


def _coordinacion_y_d2(frac_coords: list[list[float]]) -> tuple[float, int]:
    base = np.asarray(frac_coords, dtype=float)
    origen = base[0]
    candidatos: list[float] = []
    from itertools import product

    for destino in base:
        for desp in product((-1, 0, 1), repeat=3):
            if np.allclose(destino, origen) and desp == (0, 0, 0):
                continue
            vector = np.asarray(desp, dtype=float) + destino - origen
            candidatos.append(float(np.dot(vector, vector)))
    d2 = min(candidatos)
    z = sum(1 for v in candidatos if abs(v - d2) < 1e-10)
    return d2, z


def clasificar_tipo_red(sitios: int, coordinacion: int, d2: float) -> str:
    if sitios == 1 and coordinacion == 6 and abs(d2 - 1.0) < 1e-6:
        return "SC"
    if sitios == 2 and coordinacion == 8 and abs(d2 - 0.75) < 1e-6:
        return "BCC"
    if sitios == 4 and coordinacion == 12 and abs(d2 - 0.5) < 1e-6:
        return "FCC"
    if coordinacion == 8:
        return "BCC-like"
    if coordinacion == 12:
        return "FCC-like"
    if coordinacion == 6:
        return "SC-like"
    return f"desconocida(z={coordinacion})"


def analizar_estructura_mp(geometria: str) -> AnalisisEstructura:
    """Descarga/cachea, clasifica y verifica consistencia de la estructura MP."""
    material_id = material_id_de_geometria(geometria)
    data = asegurar_estructura(material_id)
    frac = data["frac_coords"]
    d2, z = _coordinacion_y_d2(frac)
    tipo = clasificar_tipo_red(len(frac), z, d2)
    simbolo = _simbolo_espacial(str(data.get("symmetry", "")))

    esperado = {
        "mp-13": ("BCC", 2, 8, "Im-3m"),
        "mp-23": ("FCC", 4, 12, "Fm-3m"),
        "mp-102": ("FCC", 4, 12, "Fm-3m"),
    }.get(material_id)

    valida = True
    mensajes: list[str] = []
    if esperado:
        tipo_e, sitios_e, z_e, sym_e = esperado
        if tipo != tipo_e:
            valida = False
            mensajes.append(f"tipo {tipo} ≠ esperado {tipo_e}")
        if len(frac) != sitios_e:
            valida = False
            mensajes.append(f"sitios {len(frac)} ≠ {sitios_e}")
        if z != z_e:
            valida = False
            mensajes.append(f"z={z} ≠ {z_e}")
        if sym_e not in simbolo and sym_e not in str(data.get("symmetry", "")):
            mensajes.append(f"aviso simetría: {simbolo} (esperada {sym_e})")
    if abs(float(data["a"]) - float(data["b"])) > 1e-4 or abs(
        float(data["a"]) - float(data["c"])
    ) > 1e-4:
        mensajes.append("celda no cúbica métricamente")

    if valida and not mensajes:
        mensajes.append("estructura verificada (topología y coordinación OK)")

    return AnalisisEstructura(
        material_id=material_id,
        formula=str(data.get("formula_pretty") or data.get("formula")),
        tipo_red=tipo,
        simbolo_espacial=simbolo,
        sitios_por_celda=len(frac),
        coordinacion=z,
        a_angstrom=float(data["a"]),
        bipartita=(tipo == "BCC"),
        valida=valida,
        mensaje="; ".join(mensajes),
    )


def etiqueta_estructura(geometria: str) -> str:
    """Etiqueta rica para títulos de gráficas (incluye BCC/FCC)."""
    from ..materials.mp_client import MATERIALES_MP

    if geometria in MATERIALES_MP:
        analisis = analizar_estructura_mp(geometria)
        return analisis.etiqueta_grafica()
    return {
        "cuadrada": "Cuadrada (2D, z=4)",
        "triangular": "Triangular (2D, z=6)",
        "cubica": "SC (3D, z=6)",
        "bcc": "BCC (3D, z=8)",
        "fcc": "FCC (3D, z=12)",
    }.get(geometria, geometria)
