# 🧪 Experimentación y modelado — Resumen ejecutivo

> Búsqueda del modelo final: conjuntos de features, comparación de algoritmos, ajuste de hiperparámetros y entrenamiento sensible al monto.
> Detalle completo y gráficos en [`notebooks/04.experimentacion.ipynb`](../../notebooks/04.experimentacion.ipynb). Todos los runs quedan registrados en MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).

## 🎯 Objetivo

Superar al baseline (**77,7% de la ganancia máxima**) con cambios que se sostengan fold a fold, y elegir un modelo final con su umbral de decisión. Se cambia una cosa por vez para poder atribuir cada mejora.

## 🧪 Diseño de los experimentos

| | |
|---|---|
| 📏 Métrica principal | % de la ganancia máxima (+25% del monto por legítima aprobada, −100% por fraude aprobado) |
| 🔁 Validación | Los mismos 5 folds estratificados del baseline; cada experimento se compara **fold a fold** contra una referencia |
| 🎚️ Umbral | El que maximiza la ganancia en la curva out-of-fold |
| 🔤 Tasa de fraude de `j` | Se calcula **dentro de cada fold**, sin usar la etiqueta de la transacción que se predice |
| ✅ Validación del tuning | Folds nuevos (semilla 7) que no intervinieron en la búsqueda de hiperparámetros |

**Orden de los experimentos:**
1. 🎯 Baseline replicado.
2. 🧱 Conjuntos de features con LightGBM por defecto.
3. 🤖 Cinco algoritmos con el mejor conjunto.
4. 🎛️ Búsqueda de hiperparámetros con Optuna (50 pruebas).
5. 💰 Entrenamiento con pesos por monto.

## 📊 Resultados

| Etapa | Mejor configuración | % ganancia máxima | AUC-ROC | AUC-PR | Umbral |
|---|---|---|---|---|---|
| 🎯 Baseline | LightGBM, variables originales | 77,7 ± 0,9 | 0,872 | 0,436 | 0,22 |
| 🧱 Features | LightGBM, conjunto `candidatas` | 79,0 ± 1,0 | 0,889 | 0,467 | 0,16 |
| 🤖 Modelos | LightGBM (empate técnico con CatBoost y Random Forest) | 79,0 ± 1,0 | 0,889 | 0,467 | 0,16 |
| 🎛️ Tuning (folds de la búsqueda) | LightGBM tuneado | 79,7 | — | — | — |
| ✅ Tuning (folds nuevos) | LightGBM tuneado | **78,9 ± 1,4** | 0,890 | 0,474 | 0,15 |
| 💰 Pesos por monto | Sin pesos | 78,9 ± 1,4 | 0,890 | 0,474 | 0,15 |

## 💡 Hallazgos principales

### 1. 🔤 La `j` cruda era el principal problema del baseline
- Solo con quitarla, el AUC-ROC sube de **0,872 a 0,887**. LightGBM memorizaba sus 8.324 categorías y eso no generaliza.
- **Reemplazarla por su tasa de fraude y su frecuencia** mejora la ganancia en 0,75 puntos y lo hace **en los 5 folds**. Es el cambio individual más importante.
- Mientras la `j` cruda esté presente, el AUC queda en ~0,872 aunque se agreguen su tasa y su frecuencia.

### 2. 🧱 `candidatas` es el mejor conjunto de features
- **79,0% de la ganancia máxima**, +1,2 puntos sobre el baseline **en los 5 folds**, con AUC-PR de 0,467 frente a 0,436.
- Combina `perfil_onp`, `hora`, `g_agrupado` y `j` como tasa y frecuencia. Por separado, `perfil_onp` (+0,24), `hora` (+0,09) y `g_agrupado` (+0,09) aportan poco: LightGBM ya aprendía la interacción entre `o`, `n` y `p` por su cuenta.
- Agregar las otras 13 features nuevas (`todas_sin_j_cruda`) da el mismo resultado con 33 variables en lugar de 19. Se elige el conjunto más simple.

### 3. 🤖 LightGBM, CatBoost y Random Forest están técnicamente empatados

| Modelo | % ganancia máxima | AUC-ROC | AUC-PR | Folds mejores que LightGBM |
|---|---|---|---|---|
| 🟠 LightGBM | 79,0 | 0,889 | 0,467 | — |
| CatBoost | 78,8 | 0,890 | 0,476 | 2 |
| Random Forest | 78,8 | 0,864 | 0,446 | 2 |
| XGBoost | 77,3 | 0,879 | 0,442 | 1 |
| Regresión logística | 77,1 | 0,847 | 0,382 | 0 |

- La diferencia entre los tres primeros (0,2 puntos) es menor que el desvío entre folds (~1 punto).
- **CatBoost tiene el mejor ranking, pero tarda ~11 minutos por CV** frente a ~6 segundos de LightGBM. Con ese costo no es viable una búsqueda de hiperparámetros.
- **Random Forest gana lo mismo con un AUC bastante menor**: la ganancia depende de acertar en la zona de decisión y en los montos altos, no de ordenar bien todas las transacciones.
- La regresión logística llega casi al baseline de LightGBM: con buenas features, un modelo lineal ya captura la mayor parte de la señal.
- Se elige **LightGBM** por ganancia, velocidad y facilidad para tunear.

### 4. 🎛️ El tuning aporta una mejora real pero modesta
- Sobre los folds de la búsqueda, la mejor prueba llega a 79,7% (+0,7 puntos). Pero **39 de las 50 pruebas superan el 79%**: hay una zona amplia de configuraciones equivalentes, no un óptimo puntual.
- Las mejores configuraciones tienen una **tasa de aprendizaje baja** (~0,02) con más árboles (~480), **hojas grandes** (`num_leaves` ~170) que exigen **muchos casos por hoja** (`min_child_samples` ~260). Son árboles complejos que aprenden despacio, sin hojas tan chicas que permitan memorizar.
- **Con folds nuevos, la mejora baja a +0,4 puntos** (78,9% frente a 78,5%), mejor en 4 de 5 folds. Cerca de la mitad de la mejora de la búsqueda era optimismo por elegir y evaluar con los mismos folds.
- La partición en sí mueve el resultado ~0,5 puntos: las cifras de ganancia deben leerse con un margen de **±1 punto**.

### 5. 🎚️ El umbral óptimo queda por debajo del teórico, y no por calibración
- Las probabilidades están **bien calibradas**: en el decil más alto, 0,31 predicho frente a 0,31 observado.
- Aun así, el umbral óptimo (0,15) queda por debajo del teórico de 0,20. La regla `p < 0,20` supone que el monto no depende de si la transacción es fraude, pero **los fraudes son más caros** (monto medio de 73 frente a 42).
- La curva de ganancia es plana entre 0,15 y 0,20: usar el umbral teórico cuesta **0,6 puntos** con el modelo tuneado.

### 6. 💰 Entrenar con pesos por monto no mejora

| Esquema | % ganancia máxima | AUC-PR | Umbral |
|---|---|---|---|
| Sin pesos | 78,9 | 0,474 | 0,15 |
| Peso = raíz del monto | 78,8 | 0,471 | 0,18 |
| Peso = monto | 78,4 | 0,459 | 0,14 |

- Con la raíz del monto el resultado es el mismo, dentro del ruido. Con el monto directo empeora 0,5 puntos.
- La diferencia se decide en el **quintil de montos más altos**, que concentra más de la mitad de la ganancia. Ahí el modelo con peso por monto rechaza menos y detecta menos fraudes (1.294 frente a 1.366).
- Con montos de 0,02 a 3.696, unas pocas transacciones caras dominan el entrenamiento y el modelo aprende con más varianza. Como `monto` ya es una feature y las probabilidades están calibradas, los pesos no agregan información.

## 🏆 Modelo elegido

| | |
|---|---|
| 🌳 Algoritmo | LightGBM tuneado, sin pesos por monto |
| 🧱 Features | Conjunto `candidatas`: variables originales sin `g` ni `j`, más `g_agrupado`, `perfil_onp`, `hora`, `j_frecuencia` y la tasa de fraude de `j` calculada dentro de cada fold |
| 🎛️ Hiperparámetros | `n_estimators` 481, `learning_rate` 0,019, `num_leaves` 172, `min_child_samples` 262, `subsample` 0,96, `colsample_bytree` 0,96, `reg_alpha` 1,61, `reg_lambda` 3,03 |
| 🎚️ Umbral | 0,15 |
| 📏 Resultado (folds nuevos) | **78,9% de la ganancia máxima**, AUC-ROC 0,890, AUC-PR 0,474 |
| 📈 Mejora sobre el baseline | ~+1,2 puntos (77,7% → 78,9%) y sobre aprobar todo, ~+15,5 puntos (63,4% → 78,9%) |

## 🛠️ Próximos pasos

- 🔬 Explicar el modelo elegido: qué variables usa, con qué forma y cuánto vale cada una en ganancia (notebook 05).
- 🧹 Evaluar si quitar variables sin aporte simplifica el modelo sin perder ganancia.

## ⚠️ Supuestos y limitaciones

- La métrica es **ruidosa**: cambios triviales mueven la ganancia ~0,3 puntos porque cambian el umbral elegido. Solo se consideran mejoras las que se repiten en la mayoría de los folds.
- La validación es **aleatoria**, respaldada por la comparación con una CV temporal del [baseline](02_baseline.md): el ranking es equivalente (AUC-ROC 0,872 frente a 0,866) y la ganancia temporal queda ~1,9 puntos abajo, dentro de su desvío (75,6 ± 2,6). Esa comparación se hizo con el baseline, no con el modelo final, y la ganancia varía según la semana que se valida.
- El resultado depende de que `score` esté disponible al momento de decidir, igual que en el baseline.
- Los algoritmos distintos de LightGBM se compararon con sus **parámetros por defecto**. XGBoost, en particular, podría mejorar con ajuste.
