# 🔬 Explicabilidad del modelo — Resumen ejecutivo

> Qué aprendió el modelo final y por qué decide lo que decide.
> Detalle completo y gráficos en [`notebooks/05.explicabilidad.ipynb`](../../notebooks/05.explicabilidad.ipynb).

## 🎯 Objetivo

Entender en qué se apoya el **LightGBM tuneado** del notebook 04 (conjunto `candidatas`, umbral 0,15): qué variables pesan, si las relaciones que aprendió tienen sentido de negocio, cuánto dinero vale cada variable y por qué comete sus errores más caros.

## 🧪 Cómo se explicó el modelo

| | |
|---|---|
| 🤖 Modelo | Parámetros, features y umbral del run `validacion_lightgbm_tuneado` de MLflow, o de las constantes del pipeline productivo, que tienen los mismos valores |
| 🔁 Datos | Entrenado con 4 de los 5 folds nuevos (semilla 7) y explicado sobre el quinto: **30.000 transacciones no vistas** |
| 🌍 SHAP (TreeExplainer) | Aporte de cada variable a la predicción de cada transacción, en log-odds, sobre una muestra estratificada de 10.000 |
| 📈 Dependencia parcial | Probabilidad media predicha al fijar una variable en cada valor |
| 💰 Permutación en ganancia | Puntos de la ganancia máxima que se pierden al desordenar cada variable (5 repeticiones) |

En el fold de explicación el modelo llega a un **AUC de 0,892** y al **77,9% de la ganancia máxima**: aprueba el 92,6% de las transacciones y detecta el 53,6% de los fraudes. Está dentro de la variación normal de la validación (78,9% ± 1,4).

## 💡 Hallazgos principales

### 1. 🌍 El modelo se apoya en `score`, el perfil `o`/`n`/`p`, el país y el monto

| Variable | Aporte SHAP medio | Pérdida de ganancia al permutar |
|---|---|---|
| `score` | 0,77 | **7,5 ± 1,2** |
| `perfil_onp` | 0,55 | 0,7 ± 0,8 |
| `o` | 0,31 | 0,5 ± 0,3 |
| `g_agrupado` | 0,20 | **2,0 ± 0,6** |
| `monto` | 0,18 | 0,4 ± 0,2 |
| `m` | 0,08–0,17 (rango del bloque de historial `l`, `m`, `e`, `d`, `f`) | **0,8 ± 0,3** |

- `score`, `perfil_onp` y `o` son las tres variables con más señal individual en la exploración y en el notebook 03. `perfil_onp` contiene a `o`, así que se leen como un bloque.
- `a`, `n` y `p` casi no aportan (≤ 0,02): su información ya está en `perfil_onp`.

### 2. 📈 Las relaciones aprendidas son coherentes con la exploración
- **`score` tiene forma de U**: la probabilidad media pasa de 3,6% con score 0 a 0,5% con 30–35, y sube a 15,5% con 95 y **28,9% con 100**. Solo con score ≥ 95 la variable por sí sola supera el umbral.
- **La madrugada sube el riesgo sin necesidad de segmentar**: entre las 0 y las 4 la probabilidad media es de 7,3–8,7%, frente a 4,3–4,5% durante el día.
- **Más monto, más riesgo**, de forma monótona: de 2,7% a 7,4% de probabilidad media entre los montos más bajos y los más altos.
- **`o` informado sube el riesgo**: nulo (−0,22), `Y` (+0,31) y `N` (+0,83). El perfil más frecuente, `nulo_1_Y` (41% de las transacciones), es el que más lo baja.

### 3. 💰 Importar en SHAP no es lo mismo que importar en dinero
- **`score` vale 7,5 puntos de la ganancia máxima** (de ~78): es, lejos, la variable más valiosa.
- **`g_agrupado` es cuarta en SHAP pero segunda en ganancia** (2,0 puntos). Una hipótesis es que separa países con tasas de fraude y montos distintos (Uruguay y Estados Unidos son los que más bajan el riesgo).
- **`hora` es el caso inverso**: mueve la predicción en la madrugada, pero no cambia la ganancia porque afecta a pocas transacciones.
- Con desvíos de 0,2 a 0,8 puntos entre repeticiones, **solo `score`, `g_agrupado` y `m` superan con claridad el ruido**.

### 4. 🔇 Algunas variables no aportan a la decisión
- `h` (−0,42 ± 0,18), `k`, `b` y `hora` dan **pérdidas negativas**: desordenarlas mejora levemente la ganancia en este fold.
- `k` recibe un aporte SHAP pequeño (0,05) aunque es ruido: el modelo corta en ella a veces, un sobreajuste leve sin impacto práctico.

### 5. 🔍 Los errores más caros dependen de `score`
Se tomó la transacción de mayor monto de cada tipo de resultado:

| Caso | Monto | `score` | Probabilidad | Qué pasó |
|---|---|---|---|---|
| ✅ Fraude detectado | 1.656 | 89 | 0,77 | Todas las señales apuntan al fraude: `score` alto, `o = N`, monto alto, perfil `N_1_N` y una `j` riesgosa |
| ❌ Fraude no detectado | 1.682 | 76 | 0,12 | Quedó justo por debajo del umbral. El monto y el perfil empujaban al fraude, pero el `score` medio aportó poco |
| ⚠️ Legítima rechazada | 2.878 | 94 | 0,24 | Tiene perfil de fraude: `score` muy alto, monto alto, perfil `Y_1_N` y país riesgoso (Brasil) |

El modelo falla cuando **un fraude tiene score medio** o cuando **una legítima tiene score muy alto**.

## 🧭 Conclusiones

- El modelo aprendió relaciones con sentido de negocio y consistentes con la exploración: la U de `score`, el riesgo de la madrugada, más riesgo con más monto y con `o` informado.
- **`score` es la pieza crítica**: es la variable más valiosa en ganancia y la raíz de los errores más caros. Que esté disponible al momento de decidir es el supuesto más importante del modelo, aunque el baseline mostró que sin ella se pierden solo ~1,3 puntos.
- Para decisiones de negocio conviene la **importancia por permutación en ganancia**, no solo la de SHAP.

## 🛠️ Próximos pasos

- 🧹 Validar con los 5 folds si quitar `h`, `k`, `b` y `hora` mantiene o mejora la ganancia, para simplificar el modelo.
- 📄 Consolidar los resultados en el informe final del challenge.

## ⚠️ Supuestos y limitaciones

- La explicación usa **un solo fold** (30.000 transacciones). Las conclusiones sobre variables con aporte cercano a cero no son definitivas.
- SHAP y la permutación **reparten el crédito entre variables correlacionadas** (`d`/`m`, `f`/`l`, `e`/`monto` y `perfil_onp`/`o`), así que su importancia individual subestima la del par.
- SHAP se calculó sobre una **muestra de 10.000** transacciones por tiempo de cómputo (~6 minutos sobre las 30.000).
- SHAP explica **qué usa el modelo, no causalidad**: que una variable suba el riesgo no significa que lo cause.
- Las interpretaciones de las variables anonimizadas siguen siendo hipótesis.
