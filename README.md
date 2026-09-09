# Modelo de Ising 2D con Monte Carlo Metropolis

Simulación del modelo de Ising en una red cuadrada con el algoritmo de Metropolis. Incluye dos análisis independientes:

1. **Temperatura fija** — tres configuraciones iniciales (aleatoria, ferromagnética, antiferromagnética), gráficas de energía/magnetización vs MCS y video de la red.
2. **Magnetización vs temperatura** — barrido de \(T\) desde ~0.001, curva \(\langle|m|/N\rangle(T)\), estimación de \(T_c\) y video hasta el régimen crítico.

## Requisitos

- Python 3.10 o superior
- `ffmpeg` (opcional, para guardar el video en MP4; si no está instalado, se exporta GIF)

Las librerías de Python se instalan con `requirements.txt` (`numpy` y `matplotlib`).

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Ejecución

```bash
python run_simulation.py                 # menú interactivo
python run_simulation.py --modo actual       # solo T fija
python run_simulation.py --modo temperatura  # solo |m|(T) y Tc
python run_simulation.py --modo todo         # ambos
```

| Modo | Carpeta de salida |
| --- | --- |
| `actual` | `resultados/simulacion_temperatura_fija/` |
| `temperatura` | `resultados/analisis_magnetizacion_vs_t/` |
| `todo` | ambas |

## Estructura del proyecto

```
.
├── README.md
├── requirements.txt
├── run_simulation.py              # punto de entrada
├── src/                           # código fuente
│   ├── ising.py                   # Metropolis 2D
│   ├── config.py                  # parámetros y rutas de salida
│   ├── visualizacion.py           # gráficas y videos
│   └── analisis_temperatura.py    # barrido M(T) y Tc
└── resultados/                    # todo lo generado por la simulación
    ├── simulacion_temperatura_fija/
    │   ├── graficas/              # E y m vs MCS
    │   ├── redes/                 # instantáneas inicial/final
    │   └── video/                 # evolucion_espines.mp4
    └── analisis_magnetizacion_vs_t/
        ├── graficas/              # |m| y χ vs T
        ├── redes/                 # red cerca de Tc
        └── video/                 # magnetizacion_hasta_tc.mp4
```

Cada análisis usa la misma organización interna: `graficas/`, `redes/` y `video/`.

### Contenido típico

**`simulacion_temperatura_fija/`**

| Archivo | Descripción |
| --- | --- |
| `graficas/energia_vs_mcs.png` | Energía por sitio hasta el equilibrio |
| `graficas/magnetizacion_vs_mcs.png` | Magnetización por sitio hasta el equilibrio |
| `redes/red_*_inicial.png` / `*_final.png` | Espines al inicio y al final |
| `video/evolucion_espines.mp4` | Evolución de las tres redes |

**`analisis_magnetizacion_vs_t/`**

| Archivo | Descripción |
| --- | --- |
| `graficas/magnetizacion_vs_temperatura.png` | \(\|m\|\) y susceptibilidad vs \(T\) |
| `redes/red_cerca_tc.png` | Configuración cerca de la \(T_c\) estimada |
| `video/magnetizacion_hasta_tc.mp4` | Recorrido de redes y curva \(M(T)\) |

Los frames temporales para armar videos se eliminan al terminar.

La \(T_c\) teórica de Onsager (Ising 2D, \(J=k_B=1\), \(h=0\)) es \(T_c = 2/\ln(1+\sqrt{2}) \approx 2.269\). En la simulación se estima como la temperatura donde \(\chi\) es máxima.

## Parámetros

Se editan en `src/config.py`:

- `L`: lado de la cuadrícula (`N = L × L`)
- `T`, `H`, `MCS`: temperatura, campo y pasos (modo T fija)
- `SEED`: semilla aleatoria
- `J_FERRO` / `J_ANTIFERRO`: constante de intercambio
- `T_MIN`, `T_MAX`, `N_TEMPERATURAS`, `MCS_T`, `H_T`: barrido en temperatura
- `VIDEO_T_FPS`: velocidad del video vs \(T\)
- `CASO_TEMPERATURA`: configuración del barrido (`ferromagnetico` por defecto)

En las visualizaciones, **rojo** es espín arriba (`+1`) y **azul** es espín abajo (`-1`).
