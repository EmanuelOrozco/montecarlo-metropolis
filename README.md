# Modelo de Ising 2D con Monte Carlo Metropolis

Simulación del modelo de Ising en redes **cuadrada** (4 vecinos) y **triangular** (6 vecinos) con el algoritmo de Metropolis. Cada geometría admite los mismos análisis:

1. **Temperatura fija** — tres configuraciones iniciales, gráficas \(E/N\) y \(m/N\) vs MCS, video de la red.
2. **Magnetización vs temperatura** — barrido de \(T\), estimación de \(T_c\) y video hasta el régimen crítico.

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
python run_simulation.py --red triangular --modo temperatura
python run_simulation.py --red ambas --modo todo
```

| Flag | Valores |
| --- | --- |
| `--red` | `cuadrada`, `triangular`, `ambas` |
| `--modo` | `actual` (T fija), `temperatura` (\(\|m\|(T)\)), `todo` |

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
│   └── visualizacion.py
└── resultados/
    ├── cuadrada/
    │   ├── temperatura_fija/          # graficas / redes / video
    │   └── magnetizacion_vs_t/
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

En la simulación se estima \(T_c\) como la temperatura donde la susceptibilidad \(\chi\) es máxima.

## Parámetros

En `src/config.py`: `L`, `T`, `H`, `MCS`, `SEED`, `T_MIN`, `T_MAX`, `N_TEMPERATURAS`, `MCS_T`, `H_T`, `VIDEO_T_FPS`, `CASO_TEMPERATURA`.

**Rojo** = espín arriba (`+1`); **azul** = espín abajo (`-1`). Las instantáneas triangulares se dibujan con offset hexagonal.
