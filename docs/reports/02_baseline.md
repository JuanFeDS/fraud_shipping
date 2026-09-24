# 🤖 Modelo baseline — Resumen ejecutivo

> Primer acercamiento para medir la calidad predictiva de los datos, sin feature engineering ni búsqueda de hiperparámetros.
> Detalle completo y gráficos en [`notebooks/02.baseline.ipynb`](../../notebooks/02.baseline.ipynb).

## 🎯 Objetivo

Responder una pregunta concreta antes de invertir en el modelo final: **¿los datos tienen señal suficiente para superar a la referencia existente?** El éxito se mide en **ganancia**: +25% del monto por cada transacción legítima aprobada y −100% del monto por cada fraude aprobado.

## 🧪 Diseño del experimento

| | |
|---|---|
| 📏 Métrica principal | Ganancia, expresada como % de la ganancia máxima posible (aprobar solo legítimas) |
| 🔁 Validación | CV aleatoria estratificada, 5 folds, con predicciones out-of-fold |
| 🎚️ Umbral de decisión | El que maximiza la ganancia en la curva out-of-fold de cada modelo |
| 🌳 Modelo | LightGBM con parámetros por defecto (maneja nulos y categóricas de forma nativa) |

**Modelos comparados:**
- ⚪ **Aprobar todo**: referencia sin modelo.
- 🔵 **Score**: la variable `score` usada directamente para ordenar por riesgo.
- 🟠 **LightGBM** (principal): todas las variables.
- 🟢 **LightGBM sin score** (sensibilidad): mide cuánto depende el modelo de `score`.

## 📊 Resultados

| Modelo | AUC-ROC | AUC-PR | % ganancia máxima | % rechazadas | % fraudes detectados | Umbral |
|---|---|---|---|---|---|---|
| ⚪ Aprobar todo | — | — | 63,4 ± 1,9 | 0 | 0 | — |
| 🔵 Score | 0,726 | 0,177 | 67,4 ± 1,2 | 8,5 | 38,5 | 0,90 |
| 🟠 **LightGBM** | **0,872** | **0,437** | **77,7 ± 1,7** | **5,3** | **44,6** | 0,22 |
| 🟢 LightGBM sin score | 0,834 | 0,362 | 76,4 ± 1,1 | 5,3 | 38,9 | 0,20 |

## 💡 Hallazgos principales

### 1. ✅ Los datos tienen señal predictiva suficiente
Un LightGBM sin ningún ajuste llega al **77,7%** de la ganancia máxima, frente al **67,4%** del score: **+10 puntos**, y lo supera en **los 5 folds**. De la ganancia que se pierde por fraude al aprobar todo, el modelo recupera el **39%** y el score solo el 11%.

### 2. 🎯 Mejor en las dos dimensiones del problema
LightGBM **detecta más fraudes** (44,6% frente a 38,5%) y **rechaza menos transacciones legítimas** (5,3% frente a 8,5%) al mismo tiempo. No hay que elegir entre seguridad y experiencia del cliente.

### 3. 💵 Aprobar todo ya es un piso alto
Como el 95% de las transacciones son legítimas, no hacer nada ya obtiene el **63%** de la ganancia máxima. El margen real para mejorar es el 37% restante, y es ahí donde se mide el aporte del modelo.

### 4. 🎚️ El umbral importa tanto como el modelo
- El umbral óptimo de LightGBM (**0,22**) es casi el teórico (**0,20**, que sale de aprobar mientras `p · monto < (1 − p) · 0,25 · monto`): sus probabilidades están razonablemente calibradas.
- La curva de ganancia es **plana alrededor del óptimo**, así que un error en el umbral cuesta poco.
- El score, en cambio, necesita un umbral de **0,90** encontrado empíricamente, y su curva es más sensible.

### 5. 🔁 Las transacciones se comportan como independientes del tiempo
Se comparó la CV aleatoria con una **CV temporal** (ventana expansiva semanal):

| Esquema | AUC-ROC | % ganancia máxima |
|---|---|---|
| Aleatoria | 0,872 ± 0,007 | 77,5 ± 1,5 |
| Temporal | 0,866 ± 0,004 | 75,6 ± 2,6 |

El ranking es equivalente, y la diferencia en ganancia queda dentro de la variación entre folds. Incluso entrenado con solo 2 semanas, el modelo predice semanas no vistas con un AUC de 0,861. Esto justifica usar CV aleatoria, que aprovecha todos los datos.

### 6. 🛡️ Robusto al supuesto sobre `score`
Sin `score`, el modelo pierde solo **1,3 puntos** y sigue superando ampliamente a la referencia. Si en producción `score` no estuviera disponible a tiempo, la solución seguiría siendo válida.

### 7. ⚠️ `j` necesita revisión
`j` es la variable más importante del modelo (~29%), seguida de `score` (~27%) y `o` (~27%). Con **8.324 categorías**, existe el riesgo de que LightGBM esté **memorizando categorías** vistas en entrenamiento, y la importancia por ganancia tiende a inflarse en estos casos.

## 🛠️ Próximos pasos

- 🔤 Validar el efecto de `j`: comparar el modelo con y sin ella, o con frequency/target encoding.
- 🧱 Feature engineering: nulos informativos, agrupación de países poco frecuentes y variables derivadas de la fecha.
- 🎛️ Búsqueda de hiperparámetros con la ganancia como métrica objetivo.
- 📐 Revisar la calibración de las probabilidades.
- 🗂️ Modularizar el código y registrar los experimentos con MLflow.

## ⚠️ Supuestos y limitaciones

- Se asume que todas las variables, incluido `score`, están **disponibles al momento de decidir**. El análisis de leakage de la exploración no encontró señales de que `score` reproduzca la etiqueta.
- El umbral se elige sobre las mismas predicciones out-of-fold con las que se evalúa. Al ser un único parámetro elegido sobre 150.000 transacciones, el sesgo optimista es mínimo.
- La proporción de fraude del dataset (5% exacto) sugiere un muestreo. Si la tasa real en producción es distinta, habrá que revisar el umbral.
