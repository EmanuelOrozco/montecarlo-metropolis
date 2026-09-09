"""Parámetros de la simulación y rutas de resultados por geometría de red."""

from pathlib import Path

KB = 1.0
T = 1.0
H = 0.1
MCS = 1000
L = 10
SEED = 35000

J_FERRO = 1.0
J_ANTIFERRO = 1.0

COLOR_ARRIBA = "#E74C3C"
COLOR_ABAJO = "#1F4E79"

ROOT = Path(__file__).resolve().parents[1]
RESULTADOS = ROOT / "resultados"

# Geometrías disponibles: cada una tiene la misma estructura de salidas.
GEOMETRIAS = ("cuadrada", "triangular")

# Barrido de temperatura (|m|(T) y Tc).
T_MIN = 0.001
T_MAX = 5.0
N_TEMPERATURAS = 55
MCS_T = 800
# Se descartan los primeros MCS (termalización) y el promedio se toma del resto.
MCS_TERMALIZACION = 200
H_T = 0.0
CASO_TEMPERATURA = "ferromagnetico"
VIDEO_T_FPS = 4

# Comparación de Tc vs tamaño de celda (L = 2, 4, …, L_MAX_COMP).
L_MIN_COMP = 2
L_MAX_COMP = 30
L_PASO_COMP = 2
ERROR_REL_MAX = 0.02
N_TEMPS_GRUESA = 16
N_TEMPS_FINA = 11
# No se corta por error en L muy pequeños: χ es ruidosa y el 2 % sería accidental.
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


def dir_geometria(geometria: str) -> Path:
    if geometria not in GEOMETRIAS:
        raise ValueError(f"geometria debe ser una de {GEOMETRIAS}, recibido: {geometria!r}")
    return RESULTADOS / geometria


def dir_t_fija(geometria: str) -> Path:
    return dir_geometria(geometria) / "temperatura_fija"


def dir_vs_t(geometria: str) -> Path:
    return dir_geometria(geometria) / "magnetizacion_vs_t"


def rutas_t_fija(geometria: str) -> dict[str, Path]:
    base = dir_t_fija(geometria)
    return {
        "base": base,
        "graficas": base / "graficas",
        "redes": base / "redes",
        "video": base / "video",
    }


def rutas_vs_t(geometria: str) -> dict[str, Path]:
    base = dir_vs_t(geometria)
    return {
        "base": base,
        "graficas": base / "graficas",
        "redes": base / "redes",
        "video": base / "video",
    }


def preparar_carpetas(*rutas: Path) -> None:
    for ruta in rutas:
        ruta.mkdir(parents=True, exist_ok=True)


def preparar_salidas_t_fija(geometria: str) -> dict[str, Path]:
    rutas = rutas_t_fija(geometria)
    preparar_carpetas(*rutas.values())
    return rutas


def dir_tamano(geometria: str) -> Path:
    return dir_geometria(geometria) / "tc_vs_tamano"


def rutas_tamano(geometria: str) -> dict[str, Path]:
    base = dir_tamano(geometria)
    return {
        "base": base,
        "graficas": base / "graficas",
        "tablas": base / "tablas",
    }


def preparar_salidas_tamano(geometria: str) -> dict[str, Path]:
    rutas = rutas_tamano(geometria)
    preparar_carpetas(*rutas.values())
    return rutas


def preparar_salidas_vs_t(geometria: str) -> dict[str, Path]:
    rutas = rutas_vs_t(geometria)
    preparar_carpetas(*rutas.values())
    return rutas
