"""Parámetros de simulación y rutas: Ising y Heisenberg clásico."""

from __future__ import annotations

from pathlib import Path

# --- Constantes físicas ---
# Constante de Boltzmann en eV/K (Heisenberg físico).
KB_EV = 8.617333262145e-5
# Ising académico usa unidades reducidas kB = 1.
KB_ISING = 1.0
# Alias histórico para el Ising en unidades reducidas.
KB = KB_ISING

T = 1.0
H = 0.1
MCS = 1000
L = 10
L_CUBICA = 8
SEED = 35000

J_FERRO = 1.0
J_ANTIFERRO = -1.0

COLOR_ARRIBA = "#E74C3C"
COLOR_ABAJO = "#1F4E79"
VISTA_3D_ELEVACION = 22.0
VISTA_3D_AZIMUT = 38.0
ROTACION_3D_POR_FRAME = 0.35

ROOT = Path(__file__).resolve().parents[1]
RESULTADOS = ROOT / "resultados"
MODELOS = ("ising", "heisenberg")
INICIALIZACIONES = ("aleatorio", "ferromagnetico", "antiferromagnetico")

GEOMETRIAS_2D = ("cuadrada", "triangular")
GEOMETRIAS_3D = ("cubica", "bcc", "fcc", "fe_mp13", "ni_mp23", "co_mp102")
GEOMETRIAS = (*GEOMETRIAS_2D, *GEOMETRIAS_3D)
# Heisenberg se aplica a redes 3D bulk (incluye Materials Project).
GEOMETRIAS_HEISENBERG = GEOMETRIAS_3D

# Temperaturas de Curie experimentales (K) para calibrar J.
TC_EXPERIMENTAL_K = {
    "fe_mp13": 1043.0,
    "ni_mp23": 631.0,
    "co_mp102": 1394.0,
    "bcc": 1043.0,
    "fcc": 631.0,
}

# Barrido Ising (unidades reducidas).
T_MIN = 0.001
T_MAX = 5.0
T_MAX_CUBICA = 7.0
T_MAX_BCC = 9.0
T_MAX_FCC = 13.0
T_MAX_FE_MP13 = 9.0
T_MAX_NI_MP23 = 13.0
T_MAX_CO_MP102 = 13.0
N_TEMPERATURAS = 55
MCS_T = 800
MCS_TERMALIZACION = 200
H_T = 0.0
CASO_TEMPERATURA = "ferromagnetico"
VIDEO_T_FPS = 4

# Heisenberg: T en kelvin, nunca desde 0.
T_MIN_HEISENBERG_K = 0.1
N_TEMPERATURAS_H = 40
MCS_T_H = 400
MCS_TERMALIZACION_H = 100
MCS_CAJA_H = 250

# Comparación Tc vs L / caja mínima.
L_MIN_COMP = 2
L_MAX_COMP = 30
L_MAX_COMP_CUBICA = 12
L_MAX_COMP_CENTRADAS = 8
L_PASO_COMP = 2
ERROR_REL_MAX = 0.02
# Criterio de estabilización de caja: cambio relativo entre L consecutivos.
DELTA_TC_CAJA_MAX = 0.01
N_TEMPS_GRUESA = 16
N_TEMPS_FINA = 11
L_MIN_PARADA = 8

CASOS = (
    {
        "clave": "aleatorio",
        "titulo": "Espines aleatorios",
        "inicializacion": "aleatorio",
        "J": J_FERRO,
    },
    {
        "clave": "ferromagnetico",
        "titulo": "Ferromagnético",
        "inicializacion": "ferromagnetico",
        "J": J_FERRO,
    },
    {
        "clave": "antiferromagnetico",
        "titulo": "Antiferromagnético",
        "inicializacion": "antiferromagnetico",
        "J": J_ANTIFERRO,
    },
)


def validar_modelo(modelo: str) -> str:
    if modelo not in MODELOS:
        raise ValueError(f"modelo debe ser uno de {MODELOS}, recibido: {modelo!r}")
    return modelo


def lado_geometria(geometria: str) -> int:
    return L_CUBICA if geometria in GEOMETRIAS_3D else L


def t_max_geometria(geometria: str) -> float:
    return {
        "cubica": T_MAX_CUBICA,
        "bcc": T_MAX_BCC,
        "fcc": T_MAX_FCC,
        "fe_mp13": T_MAX_FE_MP13,
        "ni_mp23": T_MAX_NI_MP23,
        "co_mp102": T_MAX_CO_MP102,
    }.get(geometria, T_MAX)


def l_max_comparacion(geometria: str) -> int:
    if geometria in {"bcc", "fcc", "fe_mp13", "ni_mp23", "co_mp102"}:
        return L_MAX_COMP_CENTRADAS
    return L_MAX_COMP_CUBICA if geometria == "cubica" else L_MAX_COMP


def tc_experimental_k(geometria: str) -> float | None:
    return TC_EXPERIMENTAL_K.get(geometria)


def dir_modelo(modelo: str) -> Path:
    return RESULTADOS / validar_modelo(modelo)


def dir_geometria(geometria: str, modelo: str = "ising") -> Path:
    if geometria not in GEOMETRIAS:
        raise ValueError(f"geometria debe ser una de {GEOMETRIAS}, recibido: {geometria!r}")
    return dir_modelo(modelo) / geometria


def dir_t_fija(geometria: str, modelo: str = "ising") -> Path:
    return dir_geometria(geometria, modelo) / "temperatura_fija"


def dir_vs_t(geometria: str, modelo: str = "ising") -> Path:
    return dir_geometria(geometria, modelo) / "magnetizacion_vs_t"


def dir_tamano(geometria: str, modelo: str = "ising") -> Path:
    return dir_geometria(geometria, modelo) / "tc_vs_tamano"


def dir_calibracion(geometria: str, modelo: str = "heisenberg") -> Path:
    return dir_geometria(geometria, modelo) / "calibracion_j_caja"


def _rutas_analisis(base: Path) -> dict[str, Path]:
    return {
        "base": base,
        "graficas": base / "graficas",
        "redes": base / "redes",
        "video": base / "video",
    }


def rutas_t_fija(geometria: str, modelo: str = "ising") -> dict[str, Path]:
    return _rutas_analisis(dir_t_fija(geometria, modelo))


def rutas_vs_t(geometria: str, modelo: str = "ising") -> dict[str, Path]:
    return _rutas_analisis(dir_vs_t(geometria, modelo))


def rutas_tamano(geometria: str, modelo: str = "ising") -> dict[str, Path]:
    base = dir_tamano(geometria, modelo)
    return {"base": base, "graficas": base / "graficas", "tablas": base / "tablas"}


def rutas_calibracion(geometria: str, modelo: str = "heisenberg") -> dict[str, Path]:
    base = dir_calibracion(geometria, modelo)
    return {"base": base, "graficas": base / "graficas", "tablas": base / "tablas"}


def preparar_carpetas(*rutas: Path) -> None:
    for ruta in rutas:
        ruta.mkdir(parents=True, exist_ok=True)


def preparar_salidas_t_fija(geometria: str, modelo: str = "ising") -> dict[str, Path]:
    rutas = rutas_t_fija(geometria, modelo)
    preparar_carpetas(*rutas.values())
    return rutas


def preparar_salidas_vs_t(geometria: str, modelo: str = "ising") -> dict[str, Path]:
    rutas = rutas_vs_t(geometria, modelo)
    preparar_carpetas(*rutas.values())
    return rutas


def preparar_salidas_tamano(geometria: str, modelo: str = "ising") -> dict[str, Path]:
    rutas = rutas_tamano(geometria, modelo)
    preparar_carpetas(*rutas.values())
    return rutas


def preparar_salidas_calibracion(
    geometria: str, modelo: str = "heisenberg"
) -> dict[str, Path]:
    rutas = rutas_calibracion(geometria, modelo)
    preparar_carpetas(*rutas.values())
    return rutas
