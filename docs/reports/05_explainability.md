# 🔬 Explicabilidad del modelo — Resumen ejecutivo

> Qué aprendió el modelo final y por qué decide lo que decide.
> Detalle completo y gráficos en [`notebooks/05.explicabilidad.ipynb`](../../notebooks/05.explicabilidad.ipynb).

## 🎯 Objetivo

Entender en qué se apoya el **LightGBM tuneado** del notebook 04 (`candidatas` sin `perfil_onp`, umbral 0,20): qué variables pesan, si las relaciones que aprendió tienen sentido de negocio, cuánto dinero vale cada variable y por qué comete sus errores más caros.

## 🧪 Cómo se explicó el modelo

| | |
|---|---|
| 🤖 Modelo | Parámetros y features del run `validacion_lightgbm_tuneado_sin_perfil_onp` de MLflow, o de las constantes del pipeline productivo, que tienen los mismos valores. El umbral es siempre el de producción (0,20) |
| 🔁 Datos | Entrenado con 4 de los 5 folds de validación (semilla 7) y explicado sobre el quinto: **30.000 transacciones con las que no entrenó** (ese fold sí participó en la selección del notebook 04) |
| 🌍 SHAP (TreeExplainer) | Aporte de cada variable a la predicción de cada transacción, en log-odds, sobre una muestra estratificada de 10.000 |
| 📈 Dependencia parcial | Probabilidad media predicha al fijar una variable en cada valor |
| 💰 Permutación en ganancia | Puntos de la ganancia máxima que se pierden al desordenar cada variable (5 repeticiones) |

En el fold de explicación el modelo llega a un **AUC de 0,891** y al **78,7% de la ganancia máxima**: aprueba el 94,5% de las transacciones y detecta el 47,8% de los fraudes. Está en línea con la validación del pipeline (78,9% ± 1,3).

## 💡 Hallazgos principales

### 1. 🌍 El modelo se apoya en `score` y `o`, y después en el país y el monto

| Variable | Aporte SHAP medio | Pérdida de ganancia al permutar |
|---|---|---|
| `score` | 0,80 | **8,3 ± 1,0** |
| `o` | 0,78 | **6,3 ± 0,7** |
| `g_agrupado` | 0,21 | 1,3 ± 0,4 |
| `monto` | 0,18 | 1,2 ± 0,3 |
| `l` | 0,18 | 0,9 ± 0,3 |
| `m` | 0,13 | 1,4 ± 0,7 |
| Tasa de fraude de `j` | 0,14 | 0,8 ± 0,2 |

- **Sin `perfil_onp`, `o` concentra la señal del perfil** y pasa a ser la segunda variable en SHAP y en ganancia. Quitar el perfil no hizo perder información.
- `a`, `n` y `p` casi no aportan en SHAP (0,06 o menos): su información está mayormente en `o`.

### 2. 📈 Las relaciones aprendidas son coherentes con la exploración
- **`score` tiene forma de U**: la probabilidad media pasa de 3,5% con score 0 a 0,5% con 30–35, y sube a 15,8% con 95 y **30,0% con 100**. Solo con un score cercano a 100 la variable por sí sola supera el umbral de 0,20.
- **La madrugada sube el riesgo sin necesidad de segmentar**: entre las 0 y las 4 la probabilidad media es de 7,3–8,4%, frente a 4,3–4,5% durante el día.
- **Más monto, más riesgo**, de forma monótona: de 2,7% a 7,4% de probabilidad media entre los montos más bajos y los más altos.
- **`o` informado sube el riesgo**: nulo (−0,54), `Y` (+1,16) y `N` (+1,83), con efectos más grandes que cuando el modelo tenía el perfil.

### 3. 💰 Importar en SHAP no es lo mismo que importar en dinero
- **`score` vale 8,3 puntos de la ganancia máxima** (de ~79) y `o`, 6,3. Juntas concentran casi todo el valor.
- `g_agrupado` pesa en dinero porque separa países con riesgos muy distintos: Uruguay (−1,18) y Estados Unidos (−0,60) son los que más bajan el riesgo en SHAP.
- **`hora` mueve la predicción en la madrugada, pero vale poco en dinero en este fold** (0,5 ± 0,3), porque afecta a pocas transacciones. Con los cinco folds del notebook 04, quitarla cuesta 0,26 puntos.
- `score` y `o` superan el ruido con claridad; `g_agrupado`, `monto`, `l`, la tasa de `j` y `m` quedan entre 2 y 4 desvíos por encima de cero.

### 4. 🔇 Algunas variables no aportan a la decisión
- Solo `e` da una pérdida negativa (−0,17 ± 0,16), dentro del ruido. `h`, `j_frecuencia`, `c` y `k` quedan en cero o cerca.
- `k` recibe un aporte SHAP pequeño (0,06) aunque es ruido: el modelo corta en ella a veces, un sobreajuste leve sin impacto práctico.

### 5. 🔍 Los errores más caros dependen de `score`
Se tomó la transacción de mayor monto de cada tipo de resultado:

| Caso | Monto | `score` | Probabilidad | Qué pasó |
|---|---|---|---|---|
| ✅ Fraude detectado | 1.656 | 89 | 0,77 | Todas las señales apuntan al fraude: `o = N`, `score` alto, monto alto y una `j` riesgosa |
| ❌ Fraude no detectado | 1.682 | 76 | 0,12 | Quedó por debajo del umbral. El monto y `o = Y` empujaban al fraude, pero el `score` medio aportó poco y ser de Argentina restó |
| ⚠️ Legítima rechazada | 2.878 | 94 | 0,31 | Tiene perfil de fraude: `score` muy alto, `o = Y`, monto alto y país riesgoso (Brasil) |

El modelo falla cuando **un fraude tiene score medio** o cuando **una legítima tiene score muy alto**.

## 🧭 Conclusiones

- El modelo aprendió relaciones con sentido de negocio y consistentes con la exploración: la U de `score`, el riesgo de la madrugada, más riesgo con más monto y con `o` informado.
- **`score` es la pieza crítica**: es la variable más valiosa en ganancia y la raíz de los errores más caros. Que esté disponible al momento de decidir es el supuesto más importante del modelo, aunque el baseline mostró que sin ella se pierden solo ~1,3 puntos.
- **Quitar `perfil_onp` no hizo perder información**: `o` la absorbió.
- Para decisiones de negocio conviene la **importancia por permutación en ganancia**, idealmente sobre varios folds, no solo la de SHAP.

## 🛠️ Próximos pasos

- 🧹 Validar con los 5 folds si quitar `e`, `h`, `k` y `j_frecuencia` mantiene la ganancia, para simplificar el modelo.
- 🔍 Estudiar los fraudes con score medio, el principal punto ciego del modelo.

## ⚠️ Supuestos y limitaciones

- La explicación usa **un solo fold** (30.000 transacciones). Las conclusiones sobre variables con aporte cercano a cero no son definitivas.
- SHAP y la permutación **reparten el crédito entre variables correlacionadas** (`d`/`m`, `f`/`l`, `e`/`monto`), así que su importancia individual subestima la del par.
- SHAP se calculó sobre una **muestra de 10.000** transacciones por tiempo de cómputo (~6 minutos sobre las 30.000).
- SHAP explica **qué usa el modelo, no causalidad**: que una variable suba el riesgo no significa que lo cause.
- Las interpretaciones de las variables anonimizadas siguen siendo hipótesis.
