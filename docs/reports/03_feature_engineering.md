# 🧱 Feature engineering — Resumen ejecutivo

> Construcción y evaluación de variables nuevas a partir de los hallazgos de la exploración y del baseline.
> Detalle completo y gráficos en [`notebooks/03.feature_engineering.ipynb`](../../notebooks/03.feature_engineering.ipynb).

## 🎯 Objetivo

Construir variables que hagan explícita la información que la exploración y el baseline dejaron a la vista, y decidir cuáles vale la pena **medir contra el baseline**. Hay además una pregunta abierta del baseline: si la importancia de `j` (~29%) es señal real o **memorización** de sus 8.324 categorías.

## 🧪 Cómo se evaluó cada feature

| | |
|---|---|
| 📊 Relación con el fraude | Tasa de fraude por tramo o categoría, comparada con la tasa global (5%) |
| 🏆 Ranking univariado | AUC out-of-fold de la tasa de fraude por tramo (deciles, categorías y el nulo como tramo propio), con la misma vara para originales y nuevas y una variable aleatoria como referencia de ruido |
| 🔗 Redundancia | Correlación de Spearman de cada feature nueva con la variable original más cercana |
| 🔁 Folds | Los mismos 5 folds estratificados del baseline, para calcular las tasas sin usar la etiqueta de la propia transacción |

## 🧩 Features construidas

18 variables en 6 bloques:

| Bloque | Features |
|---|---|
| 🕐 Tiempo | `hora`, `dia_semana` |
| 🔤 Variable `j` | `j_frecuencia`, `j_tasa_fraude` (out-of-fold), `j_monto_relativo`, `j_transacciones_1h`, `j_transacciones_24h` |
| 🚩 Nulos y valores especiales | `bc_nulo`, `f_negativo`, `d_tope`, `e_cero`, `monto_entero` |
| 🌎 País | `g_agrupado` (países con menos de 100 transacciones → "Otros") |
| 👤 Ratios de historial | `ratio_f_l`, `ratio_m_l`, `ratio_h_l`, `ratio_d_m` |
| 🧩 Perfil | `perfil_onp` (combinación de `o`, `n` y `p`) |

## 💡 Hallazgos principales

### 1. 🥇 `perfil_onp` es la variable con más señal del dataset
Con un **AUC univariado de 0,799** supera a `score` (0,785) y a `o` (0,754), y es la única feature nueva por encima de todas las originales. El riesgo se acumula: el perfil `N_0_N` (`o = N`, `n = 0`, `p = N`) tiene **35,7% de fraude** sobre 4.626 transacciones, 7 veces la tasa global, mientras que `nulo_1_Y` concentra el 41% de las transacciones con solo 1,1%.

### 2. 🔤 `j` tiene señal real, pero el baseline la exageraba
- Calculada de forma ingenua, la tasa de fraude por categoría da un **AUC de 0,783**; calculada **out-of-fold, 0,645**. La diferencia viene de usar la etiqueta de la propia transacción, sobre todo en las categorías chicas. Esto confirma la sospecha de **memorización** del baseline.
- Aun así hay señal: el decil más riesgoso de `j_tasa_fraude` tiene **13,3%** de fraude, frente a 2,6% en el menos riesgoso.
- El fraude crece con el tamaño de la categoría: 2,8% en las que tienen una sola transacción y **7,0%** en las que tienen más de 1.000. El 27% de las categorías aparece una sola vez, así que siempre habrá categorías de validación nunca vistas en train.
- La actividad en 24 horas mide sobre todo el tamaño de la categoría (correlación de 0,89 con `j_frecuencia`), no ataques concentrados.

### 3. 🌙 La madrugada es el horario más riesgoso
Entre las 0 y las 4 la tasa de fraude va de **10% a 17%** (pico de 17,2% a las 2), de 2 a 3,5 veces la tasa global. Es un horario de poco volumen, así que su AUC univariado es moderado (0,554), pero la señal es clara e **independiente** del resto de las variables. El día de la semana no diferencia (4,5% a 5,4%).

### 4. 🚩 Los flags hacen explícito lo que el modelo ya ve
- `d_tope` es el más claro: `d = 50` en una de cada cuatro transacciones, con **2,8% de fraude frente a 5,8%**. Sugiere que `d` es una variable de historial truncada ("50 o más").
- `f_negativo` duplica la tasa (8,8%), pero solo en el 0,85% de las transacciones.
- **`monto_entero` no tiene señal** (4,1% frente a 5,0%): la hipótesis de montos redondos por pruebas de tarjetas no se sostiene.
- LightGBM ya maneja los nulos y puede cortar en `d = 50` o en `f < 0`, así que se espera poco aporte incremental.

### 5. 👤 Los ratios de historial no superan a sus componentes
Su AUC univariado va de 0,62 a 0,64, por debajo de `f` (0,681) y de `m` (0,651). `ratio_m_l` y `ratio_h_l` muestran una **forma de U** (hasta 10% de fraude en los extremos), consistente con "mucha actividad en una cuenta nueva", aunque es una hipótesis porque las variables están anonimizadas.

### 6. 🔗 Varias features repiten información existente

| Tipo | Features |
|---|---|
| 🔁 Redundantes (correlación > 0,75) | `ratio_f_l` (0,91 con `f`), `e_cero` (0,90 con `e`), `ratio_h_l` (0,79 con `h`), `d_tope` (0,77 con `d`) |
| ✨ Información nueva (correlación < 0,2) | `hora`, `j_tasa_fraude`, `j_frecuencia`, actividad reciente de `j`, `f_negativo`, `bc_nulo`, `monto_entero` |

### 7. 🔇 Variables sin señal
Al ranking se le sumó una **variable aleatoria** como piso de ruido, y quedó en un AUC de **0,500**. `monto_entero` (0,500), `f_negativo` (0,502) y la original **`k` (0,503)** quedan pegadas a ella, lo que confirma que `k` es ruido. `dia_semana` (0,510), `bc_nulo` (0,511) y `j_transacciones_1h` (0,518) apenas se separan del piso. Se mantienen en el notebook como registro del análisis.

## 🏅 Candidatas para el modelo

| Feature | Por qué |
|---|---|
| 🧩 `perfil_onp` | La más fuerte y no se reduce a una sola variable original (en el notebook 04 no aportó dentro del modelo y se quitó) |
| 🕐 `hora` | Señal clara e independiente del resto |
| 🔤 `j_tasa_fraude` y `j_frecuencia` | Reemplazan a la `j` cruda sin permitir que el modelo memorice categorías |
| 🌎 `g_agrupado` | Misma señal que `g`, pero evita aprender tasas de países con una o dos transacciones |

## 🛠️ Próximos pasos

- 🧪 Medir **conjuntos de features** contra el baseline con LightGBM, sobre los mismos folds y comparando fold a fold, para ver el aporte incremental real.
- 🔤 Comparar el modelo con y sin la `j` cruda, calculando la tasa de `j` **dentro de cada fold** para no filtrar la etiqueta.
- 🗂️ Mover la construcción de features a `src/` y registrar los experimentos en MLflow.

## ⚠️ Supuestos y limitaciones

- El **AUC univariado no mide interacciones ni aporte incremental**: una variable con poca señal individual puede sumar dentro del modelo, y una con mucha puede ser redundante. La decisión final se toma entrenando el modelo.
- Las interpretaciones de las variables anonimizadas (historial, antigüedad, categoría de producto para `j`) son **hipótesis**.
- Las tasas de fraude por perfil con menos de 30 transacciones (`N_0_Y`, `Y_0_Y`, `Y_0_N`) son inestables.
- Las features de actividad reciente solo miran hacia atrás, así que se pueden calcular al momento de decidir.
