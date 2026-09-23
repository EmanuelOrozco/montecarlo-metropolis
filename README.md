# Monte Carlo Metropolis: Ising y Heisenberg clásico

Simulación de espines en redes 2D/3D y estructuras de Materials Project
con el algoritmo de **Metropolis**, en dos modelos:

- **Ising** — espines \(\pm 1\), unidades reducidas (\(k_B=1\)).
- **Heisenberg clásico** — espines **vectores unitarios**, \(k_B\) en **eV/K**,
  temperatura en **kelvin** (\(T \ge 0.1\,\mathrm{K}\)), sistemas **bulk** con PBC.

Los resultados se separan por modelo:

```
resultados/
  ising/<geometria>/{temperatura_fija,magnetizacion_vs_t,tc_vs_tamano}/
  heisenberg/<geometria>/{calibracion_j_caja,temperatura_fija,magnetizacion_vs_t}/
```

## Ejecución rápida

```bash
# Ising (rutas bajo resultados/ising/)
python run_simulation.py --modelo ising --red fe_mp13 --modo temperatura

# Heisenberg: valida MP, busca caja mínima, calibra J y barre T
python run_simulation.py --modelo heisenberg --red fe_mp13 --modo todo --init ferromagnetico
python run_simulation.py --modelo heisenberg --red ni_mp23 --modo todo --init aleatorio
python run_simulation.py --modelo heisenberg --red co_mp102 --modo todo --init ferromagnetico
```

| Flag | Valores |
| --- | --- |
| `--modelo` | `ising`, `heisenberg` |
| `--red` | redes 2D/3D / MP (`fe_mp13`, `ni_mp23`, `co_mp102`) / `todas` / `materiales` |
| `--modo` | `actual`, `temperatura`, `tamano`, `todo` |
| `--init` | `aleatorio`, `ferromagnetico`, `antiferromagnetico` |

### Heisenberg: caja mínima y J

1. Se valida la estructura MP (BCC/FCC, z, simetría) y se muestra en títulos.
2. Se barre \(L=2,4,\ldots\) hasta que \(T_c^\star\) se estabiliza (\(\Delta T_c/T_c \le 1\%\)).
3. Se calibra \(J = k_B T_C^\mathrm{exp} / T_c^\star\) (eV) para reproducir la Curie experimental.
4. El barrido físico usa \(T\) desde **0.1 K**.

Espines aleatorios: muestreo gaussiano 3D normalizado (uniforme en la esfera, sin sesgo).

---

# Modelo de Ising (detalle)

Simulación del modelo de Ising en redes **cuadrada 2D**, **triangular 2D**,
**cúbica simple (SC)**, **cúbica centrada en el cuerpo (BCC)**, **cúbica
centrada en las caras (FCC)** y **Fe (mp-13)**, **Ni (mp-23)** y **Co (mp-102)** de Materials Project mediante
el algoritmo de Metropolis.

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
python run_simulation.py --red fe_mp13 --modo temperatura
python run_simulation.py --red ni_mp23 --modo actual
python run_simulation.py --red co_mp102 --modo temperatura
python run_simulation.py --red cuadrada --modo tamano
python run_simulation.py --red todas --modo todo
```

| Flag | Valores |
| --- | --- |
| `--red` | `cuadrada`, `triangular`, `cubica`, `bcc`, `fcc`, `fe_mp13`, `ni_mp23`, `co_mp102`, `todas` |
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


## Materials Project (Fe, Ni, Co)

Geometrías descargadas de Materials Project (celda convencional, cache en `data/`):

| Clave | MP ID | Elemento | Red | Sitios/celda | z | a (Å) |
| --- | --- | --- | --- | --- | --- | --- |
| `fe_mp13` | [mp-13](https://next-gen.materialsproject.org/materials/mp-13) | Fe | BCC Im-3m | 2 | 8 | ≈ 2.863 |
| `ni_mp23` | [mp-23](https://next-gen.materialsproject.org/materials/mp-23) | Ni | FCC Fm-3m | 4 | 12 | ≈ 3.475 |
| `co_mp102` | [mp-102](https://next-gen.materialsproject.org/materials/mp-102) | Co | FCC Fm-3m | 4 | 12 | ≈ 3.513 |

1. Copia `.env.example` → `.env` y define `MP_API_KEY` (el archivo `.env` no se versiona).
2. Si el JSON local falta, el cliente descarga la estructura con `mp-api`.
3. Las Tc de referencia del Ising (J=1) son las del BCC/FCC numérico; no son las Curie experimentales.

Dependencias extra: `mp-api`, `python-dotenv`.

## Temperaturas críticas (teoría, \(h=0\), \(J=k_B=1\))

| Red | Dimensión | \(T_c\) de referencia |
| --- | --- | --- |
| Cuadrada (Onsager) | 2D | \(2/\ln(1+\sqrt{2}) \approx 2.269\) |
| Triangular | 2D | \(4/\ln(3) \approx 3.641\) |
| Cúbica simple | 3D | \(4.511524\), estimación numérica |
| BCC | 3D | \(6.3558\), estimación numérica |
| FCC | 3D | \(9.794\), estimación numérica |
| Fe (mp-13, BCC) | 3D | \(6.3558\), estimación numérica BCC |

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
