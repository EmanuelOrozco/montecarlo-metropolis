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
SOLUCION = ROOT / "solucion"
GRAFICAS = SOLUCION / "graficas"
VIDEO = SOLUCION / "video"
REDES = SOLUCION / "redes"

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
