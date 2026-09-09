# Modelo de Ising 2D con Monte Carlo Metropolis

Simulación del modelo de Ising en redes **cuadrada** (4 vecinos) y **triangular** (6 vecinos) con el algoritmo de Metropolis. Cada geometría admite los mismos análisis:

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
python run_simulation.py --red cuadrada --modo tamano
python run_simulation.py --red ambas --modo todo
```

| Flag | Valores |
| --- | --- |
| `--red` | `cuadrada`, `triangular`, `ambas` |
| `--modo` | `actual` (T fija), `temperatura` (\(\|m\|(T)\)), `tamano` (\(T_c\) vs \(L\)), `todo` |

## Estructura

```
.
├── run_simulation.py          # menú / CLI
├── src/
│   ├── ising/                 # modelos por geometría
│   │   ├── base.py            # Metropolis compartido
│   │   ├── cuadrada.py        # 4 vecinos
│   │   └── triangular.py      # 6 vecinos
│   ├── config.py              # parámetros y rutas
│   ├── ejecucion.py           # pipelines (T fija y vs T)
│   ├── analisis_temperatura.py
│   ├── comparacion_tc.py      # Tc(L) vs teórica, parada al 2 %
│   └── visualizacion.py
└── resultados/
    ├── cuadrada/
    │   ├── temperatura_fija/          # graficas / redes / video
    │   ├── magnetizacion_vs_t/        # |m|(T), E(T), video
    │   └── tc_vs_tamano/              # tabla y gráfica Tc(L)

    └── triangular/
        ├── temperatura_fija/
        └── magnetizacion_vs_t/
```

Cada análisis usa la misma carpeta interna: `graficas/`, `redes/`, `video/`.

## Temperaturas críticas (teoría, \(h=0\), \(J=k_B=1\))

| Red | \(T_c\) |
| --- | --- |
| Cuadrada (Onsager) | \(2/\ln(1+\sqrt{2}) \approx 2.269\) |
| Triangular | \(4/\ln(3) \approx 3.641\) |

En la simulación se estima \(T_c\) como la temperatura donde la susceptibilidad \(\chi\) es máxima:

\[
\chi = \frac{\langle m^2\rangle - \langle m\rangle^2}{N\,T}
\]

Eso localiza la transición porque \(\chi\) diverge (en el límite termodinámico) en \(T_c\).

En `--modo tamano` se estima \(T_c(L)\) para \(L=2,4,\ldots,L_{\max}\) (por defecto hasta 30; búsqueda gruesa + refinamiento del pico de \(\chi\)) y se compara con la \(T_c\) exacta. El error relativo es \(|T_c(L)-T_c|/T_c\); el barrido se detiene al llegar al 2 % (solo a partir de \(L\ge 8\), para no cortar por ruido en celdas muy pequeñas).

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

En `src/config.py`: `L`, `T`, `H`, `MCS`, `SEED`, `T_MIN`, `T_MAX`, `N_TEMPERATURAS`, `MCS_T`, `MCS_TERMALIZACION`, `H_T`, `VIDEO_T_FPS`, `CASO_TEMPERATURA`, `L_MIN_COMP`, `L_MAX_COMP`, `ERROR_REL_MAX`.

**Rojo** = espín arriba (`+1`); **azul** = espín abajo (`-1`). Las instantáneas triangulares se dibujan con offset hexagonal.
