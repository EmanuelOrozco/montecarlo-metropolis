# Modelo de Ising 2D/3D con Monte Carlo Metropolis

Simulación del modelo de Ising en redes **cuadrada 2D**, **triangular 2D**,
**cúbica simple (SC)**, **cúbica centrada en el cuerpo (BCC)** y **cúbica
centrada en las caras (FCC)** mediante el algoritmo de Metropolis.

1. **Temperatura fija** — tres configuraciones iniciales, gráficas \(E/N\) y \(m/N\) vs MCS, video de la red.
2. **Magnetización vs temperatura** — barrido de \(T\), estimación de \(T_c\) y video hasta el régimen crítico.
3. **Comparación \(T_c\) vs tamaño** — celdas \(2\times 2,4\times 4,\ldots\) hasta \(L_{\max}\) (en `config.py`) frente a \(T_c\) teórica; se detiene si el error relativo \(\le 2\,\%\).

## Requisitos

- Python 3.10+
- `ffmpeg` (opcional; si falta, el video se guarda como GIF)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Ejecución

```bash
python run_simulation.py                          # menú interactivo
python run_simulation.py --red cuadrada --modo actual
python run_simulation.py --red cubica --modo actual
python run_simulation.py --red cubica --modo temperatura
python run_simulation.py --red bcc --modo tamano
python run_simulation.py --red fcc --modo actual
python run_simulation.py --red cuadrada --modo tamano
python run_simulation.py --red todas --modo todo
```

| Flag | Valores |
| --- | --- |
| `--red` | `cuadrada`, `triangular`, `cubica`, `bcc`, `fcc`, `todas` |
| `--modo` | `actual` (T fija), `temperatura` (\(\|m\|(T)\)), `tamano` (\(T_c\) vs \(L\)), `todo` |

## Estructura

```
.
├── run_simulation.py          # menú / CLI
├── src/
│   ├── ising/                 # dominio: modelos por geometría
│   │   ├── base.py            # Metropolis común para dimensión d
│   │   ├── cuadrada.py        # 2D, 4 vecinos
│   │   ├── triangular.py      # 2D, 6 vecinos
│   │   ├── cubica.py          # SC: 3D, 6 vecinos
│   │   ├── cubicas_centradas.py # BCC (z=8) y FCC (z=12)
│   │   └── topologia_cubica.py  # bases, coordenadas y tablas de vecinos
│   ├── config.py              # parámetros y rutas
│   ├── ejecucion.py           # pipelines (T fija y vs T)
│   ├── analisis_temperatura.py
│   ├── comparacion_tc.py      # Tc(L) vs teórica, parada al 2 %
│   └── visualizacion.py
├── tests/                     # invariantes 2D/3D
└── resultados/
    ├── cuadrada/
    │   ├── temperatura_fija/          # graficas / redes / video
    │   ├── magnetizacion_vs_t/        # |m|(T), E(T), video
    │   └── tc_vs_tamano/              # tabla y gráfica Tc(L)
    ├── triangular/
    │   ├── temperatura_fija/
    │   └── magnetizacion_vs_t/
    ├── cubica/
    │   ├── temperatura_fija/
    │   ├── magnetizacion_vs_t/
    │   └── tc_vs_tamano/
    ├── bcc/                   # misma estructura de análisis
    └── fcc/                   # misma estructura de análisis
```

Las salidas se agrupan por geometría y análisis; los barridos de tamaño
añaden una carpeta `tablas/`.

## Arquitectura

- `src/ising/base.py` contiene el algoritmo de Metropolis independiente de la
  dimensión: calcula \(N=L^d\), crea la forma de la red y conserva los
  historiales.
- Cada geometría implementa únicamente su energía y vecindad. BCC usa una
  base de 2 sitios por celda y FCC una base de 4; sus tablas periódicas se
  construyen una vez y se reutilizan.
- `src/ejecucion.py` orquesta los casos de uso; `src/visualizacion.py` se
  encarga exclusivamente de gráficas y videos.
- `src/config.py` separa los tamaños 2D (`L`) y 3D (`L_CUBICA`) para controlar
  el coste \(O(L^d)\).

Las pruebas se ejecutan con:

```bash
python -m unittest discover -v
```

## Temperaturas críticas (teoría, \(h=0\), \(J=k_B=1\))

| Red | Dimensión | \(T_c\) de referencia |
| --- | --- | --- |
| Cuadrada (Onsager) | 2D | \(2/\ln(1+\sqrt{2}) \approx 2.269\) |
| Triangular | 2D | \(4/\ln(3) \approx 3.641\) |
| Cúbica simple | 3D | \(4.511524\), estimación numérica |
| BCC | 3D | \(6.3558\), estimación numérica |
| FCC | 3D | \(9.794\), estimación numérica |

Las dos referencias 2D son exactas. Los modelos 3D no tienen solución
analítica exacta conocida, por lo que se comparan con valores numéricos
aceptados.

En la simulación se estima \(T_c\) como la temperatura donde la susceptibilidad \(\chi\) es máxima:

\[
\chi = \frac{\langle m^2\rangle - \langle m\rangle^2}{N\,T}
\]

Eso localiza la transición porque \(\chi\) diverge (en el límite termodinámico) en \(T_c\).

En `--modo tamano` se estima \(T_c(L)\) para
\(L=2,4,\ldots,L_{\max}\) mediante una búsqueda gruesa y refinamiento del
pico de \(\chi\). El máximo predeterminado es 30 en 2D, 12 en SC y 8 en
BCC/FCC. Se compara
con la referencia correspondiente usando
\(|T_c(L)-T_c|/T_c\), y el barrido se detiene al llegar al 2 % (solo desde
\(L\ge 8\), para no cortar por ruido en celdas muy pequeñas).

## Termalización y promedios

El criterio de equilibrio de producción es un **corte fijo en 200 pasos de Monte Carlo**:

1. Se descartan los primeros 200 MCS (transitorio / termalización).
2. Los promedios \(\langle E/N\rangle\), \(\langle |m|/N\rangle\), \(\chi\) y \(C\) se calculan solo con MCS ≥ 200.

En las gráficas vs MCS, la media móvil marca de forma visual cuándo la curva se aplana; el valor reportado de equilibrio usa siempre el promedio a partir del paso 200 (`MCS_TERMALIZACION` en `src/config.py`).

El calor específico se obtiene de las fluctuaciones de energía:

\[
C = \frac{\mathrm{Var}(E)}{N T^2}
\]

## Parámetros

En `src/config.py`: `L`, `L_CUBICA`, `T`, `H`, `MCS`, `SEED`, `T_MIN`,
`T_MAX`, `T_MAX_CUBICA`, `T_MAX_BCC`, `T_MAX_FCC`, `N_TEMPERATURAS`, `MCS_T`,
`MCS_TERMALIZACION`, `L_MAX_COMP`, `L_MAX_COMP_CUBICA` y
`ERROR_REL_MAX`.

**Rojo** = espín arriba (`+1`); **azul** = espín abajo (`-1`). Las
instantáneas triangulares usan offset hexagonal. Las redes cúbicas representan
todos sus sitios en ejes \(x,y,z\), con transparencia y rotación gradual de
cámara. Para \(L^3\) celdas: SC tiene \(N=L^3\), BCC \(N=2L^3\) y FCC
\(N=4L^3\).
