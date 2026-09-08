# Modelo de Ising 2D con Monte Carlo Metropolis

Simulación del modelo de Ising en una red cuadrada con el algoritmo de Metropolis. Se corren tres configuraciones iniciales: espines aleatorios, ferromagnética y antiferromagnética.

Las gráficas muestran **energía por sitio** (`E/N`) y **magnetización por sitio** (`m/N`) hasta que el sistema se estabiliza.

## Requisitos

- Python 3.10 o superior
- `ffmpeg` (opcional, para guardar el video en MP4; si no está instalado, se exporta GIF)

Las librerías de Python se instalan con `requirements.txt` (`numpy` y `matplotlib`).

## Instalación

Desde la raíz de este repositorio:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En algunos sistemas el comando `pip` no está disponible; usa `python -m pip`.

## Ejecución

Con el entorno virtual activado:

```bash
python run_simulation.py
```

La salida se escribe en `solucion/`:

| Ruta | Contenido |
| --- | --- |
| `solucion/graficas/` | `energia_vs_mcs.png`, `magnetizacion_vs_mcs.png` |
| `solucion/video/` | `evolucion_espines.mp4` (o `.gif`) |
| `solucion/redes/` | instantáneas inicial y final de cada configuración |

## Estructura

```
.
├── README.md
├── requirements.txt
├── run_simulation.py    # punto de entrada
├── src/                 # modelo, parámetros y visualización
└── solucion/            # resultados de la simulación
    ├── graficas/
    ├── video/
    └── redes/
```

## Parámetros

Se editan en `src/config.py`:

- `L`: lado de la cuadrícula (`N = L × L` sitios)
- `T`, `H`, `MCS`: temperatura, campo externo y pasos de Monte Carlo
- `SEED`: semilla aleatoria
- `J_FERRO` / `J_ANTIFERRO`: constante de intercambio de cada caso

En las visualizaciones, **rojo** es espín arriba (`+1`) y **azul** es espín abajo (`-1`).
