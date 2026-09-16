"""Parámetros de la simulación y rutas de resultados por geometría de red."""

from pathlib import Path

KB = 1.0
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

# Geometrías disponibles: dos redes 2D y una red 3D.
GEOMETRIAS_2D = ("cuadrada", "triangular")
GEOMETRIAS_3D = ("cubica", "bcc", "fcc")
GEOMETRIAS = (*GEOMETRIAS_2D, *GEOMETRIAS_3D)

# Barrido de temperatura (|m|(T) y Tc).
T_MIN = 0.001
T_MAX = 5.0
T_MAX_CUBICA = 7.0
T_MAX_BCC = 9.0
T_MAX_FCC = 13.0
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
L_MAX_COMP_CUBICA = 12
L_MAX_COMP_CENTRADAS = 8
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


def lado_geometria(geometria: str) -> int:
    """Lado predeterminado; 3D usa menos sitios para limitar el coste."""
    return L_CUBICA if geometria in GEOMETRIAS_3D else L


def t_max_geometria(geometria: str) -> float:
    """Cada barrido debe superar la Tc de referencia de su red."""
    return {
        "cubica": T_MAX_CUBICA,
        "bcc": T_MAX_BCC,
        "fcc": T_MAX_FCC,
    }.get(geometria, T_MAX)


def l_max_comparacion(geometria: str) -> int:
    """Límite seguro del barrido de tamaño según la dimensión."""
    if geometria in {"bcc", "fcc"}:
        return L_MAX_COMP_CENTRADAS
    return L_MAX_COMP_CUBICA if geometria == "cubica" else L_MAX_COMP


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
