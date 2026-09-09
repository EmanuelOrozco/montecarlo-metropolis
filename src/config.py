"""Parámetros de la simulación.

Se conservan kB, T, J, h y MCS de avance.py. La cadena 1D de Nspin=25
se sustituye por una cuadrícula LxL para visualizar los espines.
"""

from pathlib import Path

KB = 1.0
T = 1.0
H = 0.1
MCS = 1000
L = 10
SEED = 35000

# Acoplamiento: J > 0 ferromagnético, J < 0 antiferromagnético.
J_FERRO = 1.0
J_ANTIFERRO = 1.0

# Espín arriba (+1) y espín abajo (−1, equivalente a 0 en la visualización).
COLOR_ARRIBA = "#E74C3C"
COLOR_ABAJO = "#1F4E79"

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Resultados: una carpeta por análisis, misma estructura interna.
# ---------------------------------------------------------------------------
RESULTADOS = ROOT / "resultados"

# Modo 1 — simulación a temperatura fija (evolución en MCS).
SIM_T_FIJA = RESULTADOS / "simulacion_temperatura_fija"
GRAFICAS_T_FIJA = SIM_T_FIJA / "graficas"
REDES_T_FIJA = SIM_T_FIJA / "redes"
VIDEO_T_FIJA = SIM_T_FIJA / "video"

# Modo 2 — barrido de magnetización vs temperatura (y Tc).
ANALISIS_VS_T = RESULTADOS / "analisis_magnetizacion_vs_t"
GRAFICAS_VS_T = ANALISIS_VS_T / "graficas"
REDES_VS_T = ANALISIS_VS_T / "redes"
VIDEO_VS_T = ANALISIS_VS_T / "video"

# Barrido de temperatura para magnetización |m|(T) y estimación de Tc.
# Tc teórica (Onsager, Ising 2D, J=kB=1, h=0): 2 / ln(1+√2) ≈ 2.269
T_MIN = 0.001
T_MAX = 5.0
N_TEMPERATURAS = 55  # más puntos; la malla se densifica al subir T
MCS_T = 800
FRACCION_TERMALIZACION = 0.4
H_T = 0.0  # campo nulo para comparar con Onsager
CASO_TEMPERATURA = "ferromagnetico"  # clave de CASOS

# Video de temperatura: 1 frame por paso de T.
VIDEO_T_FPS = 4

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


def carpetas_t_fija() -> tuple[Path, ...]:
    return (SIM_T_FIJA, GRAFICAS_T_FIJA, REDES_T_FIJA, VIDEO_T_FIJA)


def carpetas_vs_t() -> tuple[Path, ...]:
    return (ANALISIS_VS_T, GRAFICAS_VS_T, REDES_VS_T, VIDEO_VS_T)


def preparar_carpetas(*rutas: Path) -> None:
    for ruta in rutas:
        ruta.mkdir(parents=True, exist_ok=True)
