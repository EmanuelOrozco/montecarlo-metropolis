"""Cliente Materials Project: descarga y cachea estructuras cristalinas."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from ..config import ROOT

DATA_DIR = ROOT / "data"
ENV_PATH = ROOT / ".env"

# Clave local del proyecto → material_id de Materials Project.
MATERIALES_MP = {
    "fe_mp13": "mp-13",
    "ni_mp23": "mp-23",
    "co_mp102": "mp-102",
}


def _api_key() -> str:
    load_dotenv(ENV_PATH)
    key = os.getenv("MP_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Falta MP_API_KEY. Copia .env.example a .env y pega tu API key "
            "de https://next-gen.materialsproject.org/api"
        )
    return key


def ruta_cache(material_id: str) -> Path:
    return DATA_DIR / f"{material_id}.json"


def cargar_estructura_cache(material_id: str) -> dict:
    ruta = ruta_cache(material_id)
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No hay cache local para {material_id} en {ruta}. "
            "Ejecuta asegurar_estructura(...) primero."
        )
    return json.loads(ruta.read_text(encoding="utf-8"))


def descargar_estructura(material_id: str, *, conventional: bool = True) -> dict:
    """Consulta Materials Project y guarda un JSON mínimo reutilizable."""
    from mp_api.client import MPRester

    with MPRester(_api_key()) as mpr:
        estructura = mpr.get_structure_by_material_id(
            material_id,
            conventional_unit_cell=conventional,
        )
        docs = mpr.materials.summary.search(
            material_ids=[material_id],
            fields=["material_id", "formula_pretty", "symmetry", "nsites"],
        )
        resumen = docs[0] if docs else None

    payload = {
        "material_id": material_id,
        "formula": estructura.composition.reduced_formula,
        "formula_pretty": getattr(resumen, "formula_pretty", None),
        "symmetry": str(getattr(resumen, "symmetry", "")),
        "nsites": len(estructura),
        "lattice_matrix": estructura.lattice.matrix.tolist(),
        "frac_coords": [site.frac_coords.tolist() for site in estructura],
        "species": [site.species_string for site in estructura],
        "a": float(estructura.lattice.a),
        "b": float(estructura.lattice.b),
        "c": float(estructura.lattice.c),
        "angles": list(estructura.lattice.angles),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ruta_cache(material_id).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


def asegurar_estructura(material_id: str, *, forzar: bool = False) -> dict:
    """Usa cache local; solo llama a la API si falta o se fuerza la descarga."""
    ruta = ruta_cache(material_id)
    if ruta.is_file() and not forzar:
        return cargar_estructura_cache(material_id)
    return descargar_estructura(material_id)


def material_id_de_geometria(geometria: str) -> str:
    try:
        return MATERIALES_MP[geometria]
    except KeyError as exc:
        raise ValueError(
            f"Geometría de material desconocida {geometria!r}; "
            f"use una de {tuple(MATERIALES_MP)}"
        ) from exc
