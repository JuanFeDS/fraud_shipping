# 🔍 Exploración inicial — Resumen ejecutivo

> Hallazgos principales del análisis exploratorio del dataset de prevención de fraude.
> Detalle completo y gráficos en [`notebooks/01.exploracion.ipynb`](../../notebooks/01.exploracion.ipynb).

## 🎯 Contexto

El objetivo es predecir si una transacción es fraudulenta **maximizando la ganancia**: cada transacción legítima aprobada deja un **25%** de su monto y cada fraude aprobado pierde el **100%**.

## 🧾 El dataset en números

| | |
|---|---|
| 📦 Transacciones | 150.000 |
| 🧩 Variables | 19 (15 anonimizadas `a`–`p`, `fecha`, `monto`, `score`, `fraude`) |
| 🚨 Fraudes | 7.500 (**5%** exacto) |
| 📅 Periodo | 8 de marzo – 21 de abril de 2020 (45 días) |
| 🔁 Duplicados | 0 |

## 💡 Hallazgos principales

### 1. ⚖️ Clases muy desbalanceadas
Hay 19 transacciones legítimas por cada fraude: un modelo que aprueba todo tendría 95% de accuracy sin detectar un solo fraude. Además, un 5% tan redondo sugiere que el dataset fue **muestreado**, así que la tasa real en producción podría ser distinta.

### 2. 🏆 `score` es la variable más discriminante
Los fraudes se concentran entre 80 y 100, mientras que las transacciones legítimas se distribuyen de forma uniforme. No se sabe qué mide (podría ser un score del usuario o de la transacción), así que se analizó su riesgo de **data leakage**:
- La relación con el fraude es gradual e imperfecta: incluso con score 95–100 solo ~32% son fraudes. No reproduce la etiqueta.
- Tampoco es monótona: los scores bajos (0–24) tienen más fraude que los medios (25–64). Parece medir un riesgo más amplio que el fraude.
- El resto de las variables explica solo el 16% del score (R² = 0,16), así que se calcula con información que no está en el dataset, como pasa con cualquier otra variable anonimizada.

Sin señales de leakage, se incluye como feature bajo el supuesto de que está disponible al momento de decidir.

### 3. 🕳️ Los nulos cuentan una historia
- `o` tiene **72,6% de nulos**, y ese nulo es la categoría **más segura** (2,1% de fraude). En cambio, `o = N` llega a **21,8%**, más de 4 veces la tasa global.
- Los nulos de `b`/`c`, `d`/`m` y `f`/`l` aparecen siempre juntos en las mismas filas, lo que sugiere pares de variables que provienen de una misma fuente.

### 4. 👤 El fraude se concentra en perfiles con "poco historial"
`f`, `l`, `m`, `d`, `e` y `h` toman valores más bajos en los fraudes. Si representan antigüedad o actividad de la cuenta, el patrón apunta a **usuarios nuevos o poco activos**.

### 5. 💰 Los fraudes pesan más en montos altos
La distribución de `monto` de los fraudes tiene una cola más pesada por encima de ~100. Como cada fraude aprobado cuesta el **100% del monto**, estos casos son los que más impactan la ganancia.

### 6. 🚩 Categorías minoritarias = mayor riesgo

| Variable | Categoría riesgosa | Tasa de fraude | Resto |
|---|---|---|---|
| `o` | `N` | 21,8% | 2,1% – 6,5% |
| `n` | `0` | 16,1% | 3,8% |
| `p` | `N` | 7,6% | 2,9% |
| `a` | `1`, `2`, `3` | ~8% – 9% | 4,5% |

### 7. 🌎 Dos países concentran casi todo el volumen
Brasil (74%) y Argentina (21%) suman el 95% de las transacciones y tienen tasas cercanas a la media (5,5% y 3,7%). Rusia (8,2%) y España (7,3%) están por encima, y Uruguay (1,0%) y México (1,3%) muy por debajo.

### 8. 📈 El fraude no es estable en el tiempo
La tasa diaria oscila entre **2,2%** y **6,6%**, con un pico sostenido cerca de 6,5% entre el 26 de marzo y el 2 de abril. El periodo coincide con el inicio de las cuarentenas por COVID-19, un contexto atípico de consumo.

## 🛠️ Implicaciones para el modelado

- 📏 **Evaluar por ganancia**, no por accuracy, y optimizar el umbral de decisión con la función de costo del negocio.
- 🕰️ Verificar si el comportamiento depende del tiempo antes de elegir el esquema de validación (en el baseline, la CV aleatoria y la temporal dieron resultados equivalentes).
- 🕳️ Mantener los **nulos como información** (sobre todo en `o`) en lugar de imputarlos a ciegas.
- 🔤 Codificar `j` (8.324 categorías) con frequency/target encoding y agrupar los países poco frecuentes de `g`.
- 📐 Transformar a escala log las variables de cola larga (`c`, `e`, `f`, `monto`) si se usan modelos lineales.
- 📡 En producción, **monitorear el drift** de la tasa de fraude y de las variables principales.

## ⚠️ Supuestos y limitaciones

- Las variables están anonimizadas: interpretaciones como "`g` es el país" o "`f`, `l` y `m` miden historial del usuario" son **hipótesis**, no hechos.
- Se asume que todas las variables, incluido `score`, están disponibles al momento de decidir sobre la transacción.
- Los 45 días de datos no permiten capturar estacionalidad y corresponden a un periodo atípico.
